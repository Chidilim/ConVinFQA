"""Agent with procedural memory: wrong answers become lessons, retrieved for later questions.

Run from the repository root:
    python -m agent.ai_agent_procedural_memory                      # learn lessons on 20 train turns
    python -m agent.ai_agent_procedural_memory --max-turns 100      # learn on 100 train turns
    python -m agent.ai_agent_procedural_memory --split dev --max-turns 100
        # held-out test: load processed/lessons.json, write no new lessons

If LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, each turn is traced to
Langfuse with a "correct" score.
"""
import argparse
import json
import math
import operator
import os
import time
import uuid
from contextlib import ExitStack
from functools import lru_cache
from typing import Annotated, List, TypedDict, Literal,Optional
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,SystemMessage
from dotenv import load_dotenv
from agent.table_format import to_markdown
from langgraph.store.memory import InMemoryStore
load_dotenv() # at the top and not in main because llm called before the
# The store embeds each lesson's `applies_when` (the TYPE of calculation it is
# about), not the question it came from. A new question is compared against
# those conditions, so "fluctuation of UPS" no longer matches a UPS percentage
# lesson just because both mention UPS. Items without the field aren't indexed.
store = InMemoryStore(index = {
    "embed" : OpenAIEmbeddings(
        model = "text-embedding-3-small"
    ),
    "dims": 1536,
    "fields": ["applies_when"]}
                      )
LESSONS_NAMESPACE = ("vanilla_agent", "errors")
# Lessons scoring below this cosine similarity are not shown to the planner.
# 0.5 is a starting guess: tune it from the scores logged in `lessons_used`.
MIN_LESSON_SCORE = 0.5


@lru_cache(maxsize=2)
def load_records(split: str = "train"):
    """Read data/{split}_turn.json once and reuse it (train is 159 MB, so don't re-read it per node).

    lru_cache keys on `split`, so train and dev are each cached once.
    Each entry is ONE turn of a conversation. That turn's question is
    annotation.cur_dial[-1], and its expected answer is annotation.cur_program /
    annotation.exe_ans. (`qa` holds the conversation's LAST question, and is
    missing entirely on the 3,642 two-question records.)
    """
    with open(f"data/{split}_turn.json") as f:
        return json.load(f)


def get_expected(state):
    """The expected program and answer for this state's turn."""
    annotation = load_records(state["split"])[state["table_index"]]["annotation"]
    return annotation["cur_program"], annotation["exe_ans"]


def select_turns(records, max_turns: int) -> list[int]:
    """Indices of the first `max_turns` turns, extended to the end of the last conversation.

    Turns are stored in conversation order, so a new conversation starts where
    turn_ind is 0. Never cutting a conversation off keeps turn_index comparisons fair.
    """
    indices = []
    for idx, record in enumerate(records):
        if len(indices) >= max_turns and record["annotation"]["turn_ind"] == 0:
            break
        indices.append(idx)
    return indices

class StepRef(BaseModel):
    """An argument that reuses an earlier operation's result."""
    step: int = Field(description="Index of an earlier operation whose result to use: 0 = first operation, 1 = second, etc.")


class Number(BaseModel):
    """An argument that is a number taken from the table or text."""
    value: float = Field(description="A number copied from the table or text.")


class Calculation(BaseModel):
    operation: Literal["add","subtract", "multiply", "divide", "power"]
    # Two separate types, so the model must choose explicitly: {"step": 0} means
    # "the first operation's result", {"value": 0} means the number zero.
    # (With str|float|int, a bare 0 meant for "#0" was silently used as zero.)
    args: List[StepRef | Number] = Field(min_length=2, max_length=2)

class CalculationPlan(BaseModel):
    # For questions like "what were revenues in 2008?": the answer is one number
    # from the table/text, which two-argument operations can't express.
    lookup_value: Optional[float] = Field(None, description="Set this, and leave operations empty, when the answer is one number read directly from the table or text.")
    operations: List[Calculation]

class CalucationState(TypedDict):
    question: str
    pre_text: str               # report text before the table (sentences joined)
    table:str
    post_text: str              # report text after the table
    split: str                  # "train" or "dev": which data/{split}_turn.json the turn is from
    table_index: int           # position of this turn in data/{split}_turn.json
    conversation_id: str        # record id without the turn suffix; used as the checkpointer thread_id
    turn_index: int             # 0 = first question of the conversation
    # Earlier turns of this conversation: [{"question", "program", "answer"}, ...].
    # operator.add is a reducer: a node's {"history": [entry]} is APPENDED to the
    # list instead of replacing it, and the checkpointer keeps the list between
    # invoke() calls that share a thread_id.
    history: Annotated[list[dict], operator.add]
    plan: CalculationPlan
    results: list[float]
    final_answer: Optional[float]   # calculate returns a float, or None for an empty plan
    is_pass: Optional[bool]
    lessons_used: list[dict]        # [{"key", "score"}] of the lessons shown to the planner this turn
    lesson_status: Optional[str]    # what write_lesson did: "saved", "skipped: unfixable", "skipped: duplicate of <key>"


SYSTEM_PROMPT = """
You are a financial calculation planner.

You will receive:
1. Text from a financial report that appears before the table.
2. A financial table.
3. Text from the report that appears after the table.
4. A financial question.
5. Earlier turns of this conversation (questions, your plans,
   your answers).

Your task is to generate a sequence of mathematical
operations needed to answer the question.

Available operations:
- add(a, b): Add two numbers.
- subtract(a, b): Subtract b from a.
- multiply(a, b): Multiply two numbers.
- divide(a, b): Divide a by b.
- power(a, b): Raise a to the power of b.

Rules:
1. Extract numerical values from the provided table or text.
2. Select only operations needed to answer the question.
3. Do not perform calculations yourself.
4. Each operation must contain exactly two arguments.
5. Each argument is either a number from the table or text,
   written {"value": 5363}, or an earlier operation's result,
   written {"step": 0} (first operation), {"step": 1} (second), etc.
   Never write {"value": 0} to mean the first operation's result.
6. Reference only steps that occur earlier in the plan.
7. Preserve the correct order of operations.
8. Do not invent values missing from the table or text.
9. If the answer is a single value from the table or text,
   set lookup_value to it and leave operations empty.
   If the question cannot be answered, leave both empty.
10. Give percentages, percentage changes and ratios as decimals.
    Do NOT multiply by 100: a 20% increase is 0.2, not 20.
11. Follow-up questions ("what about...", "this change",
    "that total") refer to earlier turns. Use them to decide
    which values and years the question means. You may reuse
    an earlier answer as a {"value": ...}.

Example:
Question: What is the percentage increase from 100 to 120?

Operations:
- subtract({"value": 120}, {"value": 100})
- divide({"step": 0}, {"value": 100})
"""

#model from llm
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
#defining nodes:

#structured output
structured_llm = llm.with_structured_output(CalculationPlan)
    

def plan_calculation(state:CalucationState):
    #reading input
    question = state["question"]
    table = state["table"]

    # Find up to 3 lessons whose `applies_when` condition is closest to this
    # question (best first), then drop weak matches: a lesson that doesn't
    # apply can push the planner into extra steps.
    past = store.search(LESSONS_NAMESPACE, query=question, limit=3)
    past = [item for item in past if item.score is not None and item.score >= MIN_LESSON_SCORE]
    lessons_used = [{"key": item.key, "score": round(item.score, 3)} for item in past]

    # Only add the section when there are lessons, so the first questions
    # (empty store) get exactly the original prompt.
    lessons_section = ""
    if past:
        # Show the condition too, so the planner can check whether it applies
        lessons = "\n    ".join(f"- When {item.value['applies_when']}: {item.value['lesson']}" for item in past)
        lessons_section = f"""
    Lessons from past mistakes (follow one only if its "When" condition fits this question):
    {lessons}
    """

    # Same order as the report: text before, table, text after, then the question
    message = f"""
    Text before the table:
    {state["pre_text"]}

    Financial table:
    {table}

    Text after the table:
    {state["post_text"]}

    Earlier turns in this conversation:
    {format_history(state.get("history", []))}

    Question:
    {question}
    {lessons_section}"""

    #Systemprompt = how the system should behave and HumanMessage = input/information the system needs to complete the task
    plan = structured_llm.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=message)])
    
    #in langrapha, a node normally returns a dictionary that contains the state field it wants to updated, so only return what we want to change 
    return {"plan": plan, "lessons_used": lessons_used}
    
def eval(state:CalucationState):
    """Route to "right" if the answer matches the expected one, else "wrong".

    Only the answer is checked, with a tolerance:
    - exe_ans is rounded to 5 decimals (0.14136), so an exact != fails most divisions.
    - Comparing program text would fail correct plans: the agent may write
      206588.0 or "$206,588" where the dataset has 206588, and add(a, b) == add(b, a).
    """
    _, expected_answer = get_expected(state)
    answer = state["final_answer"]
    if answer is None or not math.isclose(answer, expected_answer, rel_tol=1e-3, abs_tol=1e-4):
        return "wrong"
    return "right"

    

REFLECTION_PROMPT = """
You are reviewing a financial calculation planner that answered a question wrongly.

You will receive:
1. The report text, the financial table and the earlier turns of the
   conversation that the planner saw, and the question.
2. The planner's operations and the answer they produced.
3. The expected program and the expected answer.
4. Lessons already in memory that were retrieved for this question.

In both programs, "#0" means the first operation's result, "#1" the second, etc.

What the planner can do (lessons must be instructions it can follow):
- Write operations add, subtract, multiply, divide, power, each with two
  arguments: a number from the table or text, or an earlier step's result.
- Set lookup_value when the answer is one number read from the table or text.
  Never write a lesson saying "don't calculate" without pointing to lookup_value.
- Read the earlier turns of the conversation, including its own earlier answers.

Your task:
1. Compare the planner's operations with the expected program step by step.
   Find the first step where they differ and say what was different, for example:
   wrong value taken from the table (wrong row or wrong year column), wrong
   operation, arguments in the wrong order, a missing or extra step, or the
   wrong scale (percent vs. fraction).
2. Explain the most likely reason the planner made this mistake.
3. Choose the category that fits the mistake best.
4. Write one lesson the planner can follow next time, and the condition
   (applies_when) under which it applies.
5. If a retrieved lesson already says the same thing, give its key in
   duplicate_of.
6. Decide whether to save the lesson.

Rules for the lesson:
- Write it as an instruction to the planner: "When ..., do ...".
- Make it general so it helps on other questions and tables. Do not quote
  numbers, years or company names from this example.
- If both programs give the same answer and differ only in form (for example,
  swapped arguments in add or multiply), say so, and make the lesson about
  matching the expected format.

Rules for applies_when:
- One sentence describing the TYPE of question or calculation, for example
  "the question asks for a percentage change between two periods" or
  "a follow-up question refers to a value from an earlier turn".
- No company names, metric names, numbers or years.

Set save to false when no instruction to the planner could have prevented
the mistake, for example when the expected answer looks wrong or the
question is too ambiguous to answer from what the planner saw.
"""


class Lesson(BaseModel):
    """What the reflection LLM returns: one reusable lesson from one mistake.

    Field order matters: the model fills fields in order, so it analyses the
    mistake first and only decides duplicate_of / save at the end.
    """
    mistake: str = Field(description="Where the planner's operations differ from the expected program, in one or two sentences.")
    root_cause: str = Field(description="The most likely reason the planner made this mistake.")
    category: Literal[
        "wrong_value", "wrong_operation", "wrong_order", "missing_step",
        "extra_step", "scale", "misread_followup", "other",
    ] = Field(description="The kind of mistake.")
    lesson: str = Field(description="A general instruction for the planner, with no numbers or names from this example.")
    applies_when: str = Field(description="One sentence: the type of question or calculation this lesson applies to. No company, metric, numbers or years.")
    duplicate_of: Optional[str] = Field(None, description="The key of a retrieved lesson that already says the same thing, or null.")
    save: bool = Field(description="False when no instruction to the planner could have prevented this mistake.")


# Same model, different output shape: this call returns a Lesson, not a CalculationPlan.
reflection_llm = llm.with_structured_output(Lesson)


def format_arg(arg) -> str:
    """One argument in the dataset's notation: StepRef(step=0) -> "#0", Number(value=5363.0) -> "5363"."""
    if isinstance(arg, StepRef):
        return f"#{arg.step}"
    return format_number(arg.value)


def format_number(value: float) -> str:
    """Whole numbers without ".0", so they read like the dataset (5363, not 5363.0)."""
    return str(int(value)) if value.is_integer() else str(value)


def format_program(plan) -> str:
    """Write a plan in the dataset's notation.

    Calculation: "subtract(5363, 7983), divide(#0, 7983)"
    Lookup:      "9362.2" (the dataset writes lookups as just the number)
    Neither:     "" (empty plan)
    """
    if not plan.operations and plan.lookup_value is not None:
        return format_number(plan.lookup_value)
    return ", ".join(f"{op.operation}({', '.join(format_arg(a) for a in op.args)})" for op in plan.operations)


def format_history(history) -> str:
    """Earlier turns as text for the prompt, one line per turn.

    Turn 1: what were revenues in 2008? -> plan: 9362.2 -> answer: 9362.2
    """
    if not history:
        return "(none: this is the first question in this conversation)"
    lines = []
    for n, turn in enumerate(history, start=1):
        answer = "no answer" if turn["answer"] is None else format_number(turn["answer"])
        lines.append(f"Turn {n}: {turn['question']} -> plan: {turn['program'] or '(empty)'} -> answer: {answer}")
    return "\n    ".join(lines)


def record_turn(state:CalucationState):
    """Last node of every turn: append this turn to the conversation history.

    It runs last (after write_lesson on wrong answers) so that, during a turn,
    every node sees only EARLIER turns. A one-item list works with the
    operator.add reducer on `history`.
    """
    return {"history": [{
        "question": state["question"],
        "program": format_program(state["plan"]),
        "answer": state["final_answer"],
    }]}


def write_lesson(state:CalucationState):
    #NAMESPACE IS A TUPLE THAT I DEFINE TO ORGANIZE MEMORIES
    #IT'S LIKE A FOLDER

    #Langgraph interprets namespace as "vanilla_agent\errors"
    #langraph namespaces don't automatically control user access what can access what
    namespace = LESSONS_NAMESPACE

    # The lessons the planner was shown, read back by key (no embedding call).
    # Reflection uses them to spot duplicates.
    retrieved = {}
    for used in state.get("lessons_used") or []:
        item = store.get(namespace, used["key"])
        if item is not None:
            retrieved[item.key] = item.value
    if retrieved:
        retrieved_text = "\n    ".join(
            f"- key={key}: When {value['applies_when']}: {value['lesson']}" for key, value in retrieved.items()
        )
    else:
        retrieved_text = "(none)"

    # Kept as one string. Splitting on "," would also split the
    # arguments inside each operation, e.g. "subtract(5363", " 7983)".
    expected_program, expected_answer = get_expected(state)
    # Read the plan from `state`. `final_state` only exists after invoke() returns.
    agent_program = format_program(state["plan"]) or "(empty plan)"

    message = f"""
    Text before the table:
    {state["pre_text"]}

    Financial table:
    {state["table"]}

    Text after the table:
    {state["post_text"]}

    Earlier turns in this conversation:
    {format_history(state.get("history", []))}

    Question:
    {state["question"]}

    Planner's operations:
    {agent_program}
    Planner's answer: {state["final_answer"]}

    Expected program:
    {expected_program}
    Expected answer: {expected_answer}

    Lessons retrieved for this question:
    {retrieved_text}
    """

    lesson = reflection_llm.invoke([SystemMessage(content=REFLECTION_PROMPT), HumanMessage(content=message)])

    # Only keep lessons that are fixable and new. A duplicate_of key the model
    # made up (not one of the retrieved keys) is ignored, so the lesson is saved.
    if not lesson.save:
        return {"is_pass": False, "lesson_status": "skipped: unfixable"}
    if lesson.duplicate_of in retrieved:
        return {"is_pass": False, "lesson_status": f"skipped: duplicate of {lesson.duplicate_of}"}

    # store.put(namespace, key, value): key must be unique within the namespace,
    # so each mistake gets a random id. value must be a dict.
    # The store embeds value["applies_when"] for search.
    value = lesson.model_dump(exclude={"save", "duplicate_of"})  # mistake, root_cause, category, lesson, applies_when
    store.put(namespace, str(uuid.uuid4()), {
        "question": state["question"],
        "conversation_id": state["conversation_id"],
        "turn_index": state["turn_index"],
        "agent_program": agent_program,
        "agent_answer": state["final_answer"],
        "expected_program": expected_program,
        "expected_answer": expected_answer,
        **value,
    })

    return {"is_pass": False, "lesson_status": "saved"}
    
def calculate(state:CalucationState):
    step_results = {} #dict to store result from each step.
    plan = state["plan"]

    # No operations: either a lookup (lookup_value is the answer) or the
    # question can't be answered (both empty -> None). If the model sets both
    # operations and lookup_value, the operations win.
    if not plan.operations:
        if plan.lookup_value is not None:
            return {"results": [plan.lookup_value], "final_answer": plan.lookup_value}
        return {"results": [], "final_answer": None}

    for i, operation in enumerate(plan.operations):
        func = operation.operation 
        arg1,arg2 = operation.args
        
        arg1 = resolve_arg(arg1, step_results)
        arg2 = resolve_arg(arg2, step_results)
        
        step_results[i] = perform_operations(func,arg1=arg1,arg2=arg2)
    
    return {"results": list(step_results.values()), "final_answer": list(step_results.values())[-1]}
        

def resolve_arg(arg, step_results):
    """Turn one plan argument into a number.

    The schema already separates the two kinds, so no text parsing is needed:
    - Number(value=5363.0) -> 5363.0
    - StepRef(step=0)      -> the first operation's result
    """
    if isinstance(arg, Number):
        return arg.value

    # The schema can't know how many steps exist, so check the reference here
    if arg.step not in step_results:
        raise ValueError(f"Step {arg.step} not available")
    return step_results[arg.step]
            
      
    
def perform_operations(func:str, arg1:float,arg2:float):
    
    if func == "add":
        return arg1 + arg2
    
    if func == "subtract":
        return arg1 - arg2
    
    if func == "divide":
        if arg2 == 0:
            raise ValueError("Cannot divide by zero")
        return arg1/arg2
    
    if func == "multiply":
        return arg1 * arg2
    
    if func == "power":
        return arg1 ** arg2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agent with procedural memory (lessons).")
    parser.add_argument("--split", choices=["train", "dev"], default="train",
                        help="which data/{split}_turn.json to run on")
    # Keep equal to the base agent's --max-turns for a fair comparison
    parser.add_argument("--max-turns", type=int, default=20,
                        help="run at least this many turns, then finish the current conversation")
    parser.add_argument("--lessons", choices=["learn", "frozen"], default=None,
                        help="learn: start empty and write lessons on wrong answers. "
                             "frozen: load processed/lessons.json and write none. "
                             "Default: learn on train, frozen on dev.")
    args = parser.parse_args()
    lessons_mode = args.lessons or ("learn" if args.split == "train" else "frozen")
    learn = lessons_mode == "learn"

    vanilla_financial_agent = StateGraph(CalucationState)
    vanilla_financial_agent.add_node("plan_calculation",plan_calculation)
    vanilla_financial_agent.add_node("calculate", calculate)
    vanilla_financial_agent.add_node("write_lesson", write_lesson)
    vanilla_financial_agent.add_node("record_turn", record_turn)

    vanilla_financial_agent.add_edge(START,"plan_calculation")
    vanilla_financial_agent.add_edge("plan_calculation", "calculate")
    # Both branches end in record_turn, so every turn is added to the history.
    # Frozen mode skips write_lesson: lessons are only read, so a held-out
    # split tests the lessons learned on train without learning from its labels.
    vanilla_financial_agent.add_conditional_edges("calculate", eval, {
        "wrong": "write_lesson" if learn else "record_turn",
        "right": "record_turn"
    })
    vanilla_financial_agent.add_edge("write_lesson", "record_turn")
    vanilla_financial_agent.add_edge("record_turn", END)

    # Built and compiled ONCE. The graph is the same for every question;
    # only the input state changes inside the loop.
    # Short-term memory: the checkpointer saves each thread's state after every
    # invoke() and restores it on the next invoke() with the same thread_id.
    # One thread = one conversation, so `history` carries over between turns.
    # The checkpoint stores the whole state, including the Pydantic `plan`;
    # LangGraph only restores types it is told are safe, so allow CalculationPlan.
    # __module__ is used because under `python -m` this module is "__main__".
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(CalculationPlan.__module__, "CalculationPlan")])
    compiled_graph = vanilla_financial_agent.compile(checkpointer=InMemorySaver(serde=serde))

    AGENT_NAME = "memory_agent"
    SAVE_EVERY = 10     # write lessons to disk every N examples
    TRAIN_LESSONS_PATH = "processed/lessons.json"   # lessons learned on train; frozen mode loads these
    # Learning on another split writes elsewhere, so it never overwrites the train lessons
    LESSONS_PATH = TRAIN_LESSONS_PATH if args.split == "train" else f"processed/lessons_{args.split}.json"
    RESULTS_PATH = f"processed/{AGENT_NAME}_{args.split}_results.json"
    RUN_ID = time.strftime("%Y%m%d-%H%M%S")   # tells repeated runs apart in Langfuse

    def load_lessons(path):
        """Put saved lessons back into the store, under their original keys.

        Each put embeds the lesson's applies_when (one embedding call each).
        Lessons saved before edit 4 have no applies_when, so they could never be
        found by search; they are skipped.
        """
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: run on train with --lessons learn first.")
        with open(path) as f:
            saved = json.load(f)
        loaded = 0
        for item in saved:
            if not item.get("applies_when"):
                continue
            key = item.pop("key")
            store.put(LESSONS_NAMESPACE, key, item)
            loaded += 1
        print(f"Loaded {loaded} lessons from {path} (skipped {len(saved) - loaded} without applies_when)")

    def save_lessons():
        """Write every lesson in the store to LESSONS_PATH (overwrites the file).

        InMemoryStore is lost when the script exits, so this is the only copy.
        search() without `query` makes no embedding call; the high limit is
        needed because the default limit is 10.
        """
        items = store.search(LESSONS_NAMESPACE, limit=100_000)
        os.makedirs(os.path.dirname(LESSONS_PATH), exist_ok=True)
        with open(LESSONS_PATH, "w") as f:
            json.dump([{"key": item.key, **item.value} for item in items], f, indent=2)
        print(f"Saved {len(items)} lessons to {LESSONS_PATH}")

    # Hand-written rules seeded into memory before the run, so they are retrieved
    # like learned lessons. The fixed key means re-running overwrites, not duplicates.
    # "applies_when" is required: it's the field the store embeds for search.
    store.put(LESSONS_NAMESPACE, "rule-percent-as-decimal", {
        "applies_when": "the question asks for a percentage, a percentage change, or a ratio",
        "category": "scale",
        "source": "manual rule",
        "mistake": "Multiplied a percentage or ratio by 100 (gave 14.1 instead of 0.141).",
        "root_cause": "The dataset stores percentages as decimals, but the usual habit is to show them out of 100.",
        "lesson": "When a question asks for a percentage, percentage change, or ratio, stop after the division and do not multiply by 100.",
    })
    if not learn:
        load_lessons(TRAIN_LESSONS_PATH)
    print(f"Lessons mode: {lessons_mode}")

    # Tracing is optional: without Langfuse keys the run works as before.
    langfuse = None
    callbacks = []
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client, propagate_attributes
        from langfuse.langchain import CallbackHandler
        langfuse = get_client()          # reads the LANGFUSE_* keys and LANGFUSE_BASE_URL from .env
        callbacks = [CallbackHandler()]  # records each graph node and LLM call
        print(f"Tracing to Langfuse (run {RUN_ID})")

    records = load_records(args.split)
    rows = []  # one row per example, saved for metrics
    for n, idx in enumerate(select_turns(records, args.max_turns), start=1):
        record = records[idx]
        # "Single_JKHY/2009/page_28.pdf-3_2" -> conversation "Single_JKHY/2009/page_28.pdf-3", turn 2
        conversation_id = record["id"].rsplit("_", 1)[0]
        test_state = {
                "question": record["annotation"]["cur_dial"][-1],  # this turn's question
                # pre_text / post_text are lists of sentences in the dataset
                "pre_text": " ".join(record["pre_text"]),
                "table": to_markdown(record["table"]),
                "post_text": " ".join(record["post_text"]),
                "split": args.split,
                "table_index": idx,
                "conversation_id": conversation_id,
                "turn_index": record["annotation"]["turn_ind"],
                # No "history" key: the checkpointer supplies this thread's history.
                # The fields below are reset so the previous turn's values don't leak in.
                "plan": None,
                "results": None,
                "final_answer": None,
                "is_pass": None,
                "lessons_used": [],
                "lesson_status": None,
            }
        expected_program, expected_answer = get_expected(test_state)
        row = {
            "index": idx,
            "conversation_id": conversation_id,
            "turn_index": test_state["turn_index"],
            "question": test_state["question"],
            "expected_program": expected_program,
            "expected_answer": expected_answer,
            "agent_program": None,
            "agent_answer": None,
            "correct": False,
            "error": None,
            "lessons_used": [],     # [{"key", "score"}] shown to the planner
            "lesson_status": None,  # None when the answer was right
        }

        # ExitStack enters the Langfuse contexts only when tracing is on,
        # so the same loop body works with and without Langfuse.
        with ExitStack() as stack:
            span = None
            if langfuse:
                # One span per turn = one trace per turn. The score below is
                # attached to THIS span's trace, even if invoke() raises.
                span = stack.enter_context(langfuse.start_as_current_observation(
                    as_type="agent", name=AGENT_NAME, input={"question": test_state["question"]}))
                # Trace-level attributes, set right after the span opens so the
                # graph's spans (created later) inherit them.
                stack.enter_context(propagate_attributes(
                    trace_name=AGENT_NAME,
                    # One session per conversation per run, so the turns of a
                    # conversation are shown together
                    session_id=f"{AGENT_NAME}-{args.split}-{RUN_ID}-{conversation_id}",
                    tags=[AGENT_NAME, args.split, f"lessons-{lessons_mode}"],
                    metadata={"run_id": RUN_ID, "table_index": str(idx),
                              "turn_index": str(test_state["turn_index"])},
                ))

            try:
                # thread_id picks the conversation whose saved history to restore
                final_state = compiled_graph.invoke(test_state, config={
                    "configurable": {"thread_id": conversation_id},
                    "callbacks": callbacks,
                })
            except Exception as e:
                # One bad example (invalid "#" reference, divide by zero, API error)
                # shouldn't stop the run. It counts as wrong, like in base_agent.py.
                row["error"] = repr(e)
                print(f"\n[{idx}] ERROR: {row['question']}\n  {e!r}")
            else:
                # Runs only when invoke() succeeded.
                # Score with the graph's own eval, so the metric uses the same rule
                # that decided whether to write a lesson.
                row["agent_program"] = format_program(final_state["plan"])
                row["agent_answer"] = final_state["final_answer"]
                row["correct"] = eval(final_state) == "right"
                row["lessons_used"] = final_state.get("lessons_used") or []
                row["lesson_status"] = final_state.get("lesson_status")
                print(f"\n[{idx}] {'RIGHT' if row['correct'] else 'WRONG'}: {row['question']}")
                print(f"  plan: {row['agent_program'] or '(empty)'}")
                print(f"  answer={row['agent_answer']}  expected={expected_answer}")
                print(f"  lessons used: {row['lessons_used'] or 'none'}")
                if row["lesson_status"]:
                    print(f"  lesson: {row['lesson_status']}")

            if span is not None:
                span.update(output={
                    "program": row["agent_program"], "answer": row["agent_answer"],
                    "expected_program": expected_program, "expected_answer": expected_answer,
                    "lessons_used": row["lessons_used"], "lesson_status": row["lesson_status"],
                    "error": row["error"],
                })
                # 1 = right, 0 = wrong or error. Filter on it in Langfuse to see failures.
                span.score_trace(name="correct", value=1 if row["correct"] else 0, data_type="BOOLEAN")
                # Direct link to this turn's trace (None if the project id can't be fetched)
                print(f"  trace: {langfuse.get_trace_url(trace_id=span.trace_id)}")
        rows.append(row)

        # Outside try/except, so saving still happens on a step that failed.
        # Frozen mode never changes the lessons, so there is nothing to save.
        if learn and n % SAVE_EVERY == 0:
            save_lessons()

    if learn:
        # Final save, so the examples after the last multiple of SAVE_EVERY are kept
        save_lessons()
    if langfuse:
        langfuse.flush()   # send queued traces before the script exits

    correct = sum(r["correct"] for r in rows)
    errors = sum(r["error"] is not None for r in rows)
    accuracy = correct / len(rows)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "agent": "procedural memory",
            "split": args.split,
            "lessons_mode": lessons_mode,
            "run_id": RUN_ID,
            "num_examples": len(rows),
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy,
            "examples": rows,
        }, f, indent=2)

    print(f"\nAccuracy: {correct}/{len(rows)} = {accuracy:.1%}  ({errors} errors)")
    print(f"Saved per-example results to {RESULTS_PATH}")
    
    
    #TOMORROW: RUN ON DEV TEST AND COMPARE
    #TOMORROW: TRY THE MATH PYTHON CODE VERFIER.. COMPARE SPEED
    #TOMORROW: HOW TO CONNECT AGENT PER REVENUE PER STEP