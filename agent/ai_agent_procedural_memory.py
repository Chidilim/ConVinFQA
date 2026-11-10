import json
import math
import os
import uuid
from functools import lru_cache
from typing import List, Set, TypedDict, Literal,Optional
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph,START,END 
from langchain_core.messages import HumanMessage,SystemMessage
from dotenv import load_dotenv
from agent.table_format import parse_number, to_markdown
from langgraph.store.memory import InMemoryStore
load_dotenv() # at the top and not in main because llm called before the
store = InMemoryStore(index = {
    "embed" : OpenAIEmbeddings(
        model = "text-embedding-3-small"
    ), 
    "dims": 1536,
    "fields": ["question"]}
                      )


@lru_cache(maxsize=1)
def load_records():
    """Read train_turn.json once and reuse it (159 MB, so don't re-read it per node).

    Each entry is ONE turn of a conversation. That turn's question is
    annotation.cur_dial[-1], and its expected answer is annotation.cur_program /
    annotation.exe_ans. (`qa` holds the conversation's LAST question, and is
    missing entirely on the 3,642 two-question records.)
    """
    with open("data/train_turn.json") as f:
        return json.load(f)


def get_expected(state):
    """The expected program and answer for this state's turn."""
    annotation = load_records()[state["table_index"]]["annotation"]
    return annotation["cur_program"], annotation["exe_ans"]

class Calculation(BaseModel):
    operation: Literal["add","subtract", "multiply", "divide", "power"]
    args: List[str|float|int] =  Field(min_length = 2)

class CalculationPlan(BaseModel):
    operations: List[Calculation]

class CalucationState(TypedDict):
    question: str
    pre_text: str               # report text before the table (sentences joined)
    table:str
    post_text: str              # report text after the table
    table_index: int           # position of this turn in train_turn.json
    plan: CalculationPlan
    results: list[float]
    final_answer: Optional[float]   # calculate returns a float, or None for an empty plan
    is_pass: Optional[bool]
    

SYSTEM_PROMPT = """
You are a financial calculation planner.

You will receive:
1. Text from a financial report that appears before the table.
2. A financial table.
3. Text from the report that appears after the table.
4. A financial question.

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
5. To reuse a previous result, reference it as "#0" (first
   operation's result), "#1" (second), etc.
6. Reference only steps that occur earlier in the plan.
7. Preserve the correct order of operations.
8. Do not invent values missing from the table or text.
9. If the question cannot be answered, return an empty
   list of operations.
10. Give percentages, percentage changes and ratios as decimals.
    Do NOT multiply by 100: a 20% increase is 0.2, not 20.

Example:
Question: What is the percentage increase from 100 to 120?

Operations:
- subtract(120, 100)
- divide(#0, 100)
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

    # Find the 3 saved lessons whose question is most similar to this one.
    # The store embeds `question` (index "fields") on put; here it embeds the
    # query the same way and returns the closest matches, best first.
    past = store.search(("vanilla_agent", "errors"), query=question, limit=3)

    # Only add the section when there are lessons, so the first questions
    # (empty store) get exactly the original prompt.
    lessons_section = ""
    if past:
        lessons = "\n".join(f"- {item.value['lesson']}" for item in past)
        lessons_section = f"""
    Lessons from past mistakes on similar questions (follow them if they apply):
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

    Question:
    {question}
    {lessons_section}"""

    #Systemprompt = how the system should behave and HumanMessage = input/information the system needs to complete the task
    plan = structured_llm.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=message)])
    
    #in langrapha, a node normally returns a dictionary that contains the state field it wants to updated, so only return what we want to change 
    return {"plan": plan}
    
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
1. The question and the financial table the planner saw.
2. The planner's operations and the answer they produced.
3. The expected program and the expected answer.

In both programs, "#0" means the first operation's result, "#1" the second, etc.

Your task:
1. Compare the planner's operations with the expected program step by step.
   Find the first step where they differ and say what was different, for example:
   wrong value taken from the table (wrong row or wrong year column), wrong
   operation, arguments in the wrong order, a missing or extra step, or the
   wrong scale (percent vs. fraction).
2. Explain the most likely reason the planner made this mistake.
3. Write one lesson the planner can follow next time.

Rules for the lesson:
- Write it as an instruction to the planner: "When ..., do ...".
- Make it general so it helps on other questions and tables. Do not quote
  numbers, years or company names from this example.
- If both programs give the same answer and differ only in form (for example,
  swapped arguments in add or multiply), say so, and make the lesson about
  matching the expected format.
"""


class Lesson(BaseModel):
    """What the reflection LLM returns: one reusable lesson from one mistake."""
    mistake: str = Field(description="Where the planner's operations differ from the expected program, in one or two sentences.")
    root_cause: str = Field(description="The most likely reason the planner made this mistake.")
    lesson: str = Field(description="A general instruction for the planner, with no numbers or names from this example.")


# Same model, different output shape: this call returns a Lesson, not a CalculationPlan.
reflection_llm = llm.with_structured_output(Lesson)


def format_program(operations) -> str:
    """Write a plan in the dataset's notation, e.g. "subtract(5363, 7983), divide(#0, 7983)"."""
    return ", ".join(f"{op.operation}({', '.join(map(str, op.args))})" for op in operations)


def write_lesson(state:CalucationState):
    #NAMESPACE IS A TUPLE THAT I DEFINE TO ORGANIZE MEMORIES
    #IT'S LIKE A FOLDER

    #Langgraph interprets namespace as "vanilla_agent\errors"
    #langraph namespaces don't automatically control user access what can access what
    namespace = ("vanilla_agent", "errors")

    # Kept as one string. Splitting on "," would also split the
    # arguments inside each operation, e.g. "subtract(5363", " 7983)".
    expected_program, expected_answer = get_expected(state)
    # Read the plan from `state`. `final_state` only exists after invoke() returns.
    agent_program = format_program(state["plan"].operations) or "(empty plan)"

    message = f"""
    Text before the table:
    {state["pre_text"]}

    Financial table:
    {state["table"]}

    Text after the table:
    {state["post_text"]}

    Question:
    {state["question"]}

    Planner's operations:
    {agent_program}
    Planner's answer: {state["final_answer"]}

    Expected program:
    {expected_program}
    Expected answer: {expected_answer}
    """

    lesson = reflection_llm.invoke([SystemMessage(content=REFLECTION_PROMPT), HumanMessage(content=message)])

    # store.put(namespace, key, value): key must be unique within the namespace,
    # so each mistake gets a random id. value must be a dict.
    store.put(namespace, str(uuid.uuid4()), {
        "question": state["question"],
        "agent_program": agent_program,
        "agent_answer": state["final_answer"],
        "expected_program": expected_program,
        "expected_answer": expected_answer,
        **lesson.model_dump(),  # adds the mistake, root_cause and lesson fields
    })

    return {"is_pass": False}
    
def calculate(state:CalucationState):
    step_results = {} #dict to store result from each step.
    plan = state["plan"]

    # The prompt allows an empty plan when the question can't be answered.
    # Without this guard, step_results[len(...) - 1] would be step_results[-1] -> KeyError.
    if not plan.operations:
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

    An argument is either a number (5363), a number the LLM sent as text
    ("5,363", "$5,363", "28%"), or a reference to an earlier step's result ("#0").
    """
    if not isinstance(arg, str):
        return arg

    # The LLM sometimes copies values from the table as strings, with "$" or ",".
    # parse_number returns None for anything that isn't a number, e.g. "#0".
    number = parse_number(arg)
    if number is not None:
        return number

    if not arg.startswith("#"):
        raise ValueError(f"Invalid reference: {arg}")

    index_num = arg.find("#") + 1
    number_text = arg[index_num:]

    if not number_text.isdecimal():
        raise ValueError(f"Invalid reference: {arg}")

    step_num = int(number_text)

    if step_num not in step_results:
        raise ValueError(f"Step {step_num} not available")

    return step_results[step_num]          
            
      
    
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
    
    vanilla_financial_agent = StateGraph(CalucationState)
    vanilla_financial_agent.add_node("plan_calculation",plan_calculation)
    vanilla_financial_agent.add_node("calculate", calculate)
    vanilla_financial_agent.add_node("write_lesson", write_lesson)

    vanilla_financial_agent.add_edge(START,"plan_calculation")
    vanilla_financial_agent.add_edge("plan_calculation", "calculate")
    vanilla_financial_agent.add_conditional_edges("calculate", eval, {
        "wrong": "write_lesson",
        "right": END
    })
    vanilla_financial_agent.add_edge("write_lesson", END)
    
    # Built and compiled ONCE. The graph is the same for every question;
    # only the input state changes inside the loop.
    compiled_graph = vanilla_financial_agent.compile()

    NUM_EXAMPLES = 20   # small run to keep API calls down (~2-4 calls per example)
    SAVE_EVERY = 10     # write lessons to disk every N examples
    LESSONS_PATH = "processed/lessons.json"

    def save_lessons():
        """Write every lesson in the store to LESSONS_PATH (overwrites the file).

        InMemoryStore is lost when the script exits, so this is the only copy.
        search() without `query` makes no embedding call; the high limit is
        needed because the default limit is 10.
        """
        items = store.search(("vanilla_agent", "errors"), limit=100_000)
        os.makedirs(os.path.dirname(LESSONS_PATH), exist_ok=True)
        with open(LESSONS_PATH, "w") as f:
            json.dump([{"key": item.key, **item.value} for item in items], f, indent=2)
        print(f"Saved {len(items)} lessons to {LESSONS_PATH}")

    # Hand-written rules seeded into memory before the run, so they are retrieved
    # like learned lessons. The fixed key means re-running overwrites, not duplicates.
    # "question" is required: it's the field the store embeds for search.
    store.put(("vanilla_agent", "errors"), "rule-percent-as-decimal", {
        "question": "what was the percentage change in the value from one year to the next?",
        "source": "manual rule",
        "mistake": "Multiplied a percentage or ratio by 100 (gave 14.1 instead of 0.141).",
        "root_cause": "The dataset stores percentages as decimals, but the usual habit is to show them out of 100.",
        "lesson": "When a question asks for a percentage, percentage change, or ratio, stop after the division and do not multiply by 100.",
    })

    records = load_records()
    failures = []  # examples that raised, printed at the end
    for idx in range(NUM_EXAMPLES):
        record = records[idx]
        test_state = {
                "question": record["annotation"]["cur_dial"][-1],  # this turn's question
                # pre_text / post_text are lists of sentences in the dataset
                "pre_text": " ".join(record["pre_text"]),
                "table": to_markdown(record["table"]),
                "post_text": " ".join(record["post_text"]),
                "table_index": idx,
                "plan": None,
                "results": None,
                "final_answer": None,
                "is_pass": None,
            }
        try:
            final_state = compiled_graph.invoke(test_state)
        except Exception as e:
            # One bad example (invalid "#" reference, divide by zero, API error)
            # shouldn't stop the run. Log it and move on to the next example.
            failures.append({"index": idx, "question": test_state["question"], "error": repr(e)})
            print(f"\n[{idx}] ERROR: {test_state['question']}\n  {e!r}")
        else:
            # Runs only when invoke() succeeded.
            # write_lesson sets is_pass=False; a right answer leaves it None
            outcome = "wrong" if final_state["is_pass"] is False else "right"
            _, expected_answer = get_expected(final_state)
            print(f"\n[{idx}] {outcome.upper()}: {test_state['question']}")
            # Label steps #0, #1, ... so they match the references the plan uses
            for i, op in enumerate(final_state["plan"].operations):
                print(f"  #{i}: {op.operation}({', '.join(map(str, op.args))})")
            print(f"  answer={final_state['final_answer']}  expected={expected_answer}")

        # Outside try/except, so saving still happens on a step that failed
        if (idx + 1) % SAVE_EVERY == 0:
            save_lessons()

    # Final save, so the examples after the last multiple of SAVE_EVERY are kept
    save_lessons()
    print(f"\nDone: {NUM_EXAMPLES - len(failures)} ran, {len(failures)} failed")
    for failure in failures:
        print(f"  [{failure['index']}] {failure['error']}")
    
    
    #TOMORROW: RUN ON DEV TEST AND COMPARE
    #TOMORROW: TRY THE MATH PYTHON CODE VERFIER.. COMPARE SPEED
    #TOMORROW: 