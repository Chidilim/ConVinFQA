"""Baseline agent: the same planner + calculator as ai_agent_procedural_memory.py,
but WITHOUT memory (no lesson retrieval, no write_lesson, no seeded rules).

Use it to measure what memory adds. For a fair comparison, keep everything else
identical to the memory agent: same SYSTEM_PROMPT, same inputs (pre_text, table,
post_text), same model and temperature, same examples, same scoring tolerance.

Run from the repository root:
    python -m agent.base_agent                              # first 20 train turns
    python -m agent.base_agent --split dev --max-turns 100  # held-out dev turns

If LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, each turn is traced to
Langfuse with a "correct" score.
"""
import argparse
import json
import math
import operator
import os
import time
from contextlib import ExitStack
from functools import lru_cache
from typing import Annotated, List, TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from agent.table_format import to_markdown
load_dotenv()  # before ChatOpenAI is created, so it can read OPENAI_API_KEY


@lru_cache(maxsize=2)
def load_records(split: str = "train"):
    """Read data/{split}_turn.json once and reuse it (train is 159 MB, so don't re-read it).

    lru_cache keys on `split`, so train and dev are each cached once.
    Each entry is ONE turn of a conversation. That turn's question is
    annotation.cur_dial[-1], and its expected answer is annotation.cur_program /
    annotation.exe_ans.
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
    operation: Literal["add", "subtract", "multiply", "divide", "power"]
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
    table: str
    post_text: str              # report text after the table
    split: str                  # "train" or "dev": which data/{split}_turn.json the turn is from
    table_index: int            # position of this turn in data/{split}_turn.json
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


# Must stay identical to the memory agent's prompt, or the comparison
# measures prompt differences instead of memory.
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

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
structured_llm = llm.with_structured_output(CalculationPlan)


def plan_calculation(state: CalucationState):
    """Ask the LLM for a plan. Same message as the memory agent, minus the lessons section."""
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
    """
    plan = structured_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)])
    return {"plan": plan}


def calculate(state: CalucationState):
    """Run the plan step by step. Step i's result is stored as step_results[i] ("#i")."""
    step_results = {}
    plan = state["plan"]

    # No operations: either a lookup (lookup_value is the answer) or the
    # question can't be answered (both empty -> None). If the model sets both
    # operations and lookup_value, the operations win.
    if not plan.operations:
        if plan.lookup_value is not None:
            return {"results": [plan.lookup_value], "final_answer": plan.lookup_value}
        return {"results": [], "final_answer": None}

    for i, operation in enumerate(plan.operations):
        arg1, arg2 = operation.args
        # Turn {"step": 0} / {"value": 5363} into numbers before doing arithmetic
        arg1 = resolve_arg(arg1, step_results)
        arg2 = resolve_arg(arg2, step_results)
        step_results[i] = perform_operations(operation.operation, arg1=arg1, arg2=arg2)

    results = list(step_results.values())
    return {"results": results, "final_answer": results[-1]}   # the last step is the answer


def record_turn(state: CalucationState):
    """Last node of every turn: append this turn to the conversation history.

    It runs last so that, during a turn, every node sees only EARLIER turns.
    Returning a one-item list works with the operator.add reducer on `history`.
    """
    return {"history": [{
        "question": state["question"],
        "program": format_program(state["plan"]),
        "answer": state["final_answer"],
    }]}


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


def perform_operations(func: str, arg1: float, arg2: float):
    if func == "add":
        return arg1 + arg2
    if func == "subtract":
        return arg1 - arg2
    if func == "divide":
        if arg2 == 0:
            raise ValueError("Cannot divide by zero")
        return arg1 / arg2
    if func == "multiply":
        return arg1 * arg2
    if func == "power":
        return arg1 ** arg2


def is_correct(answer, expected_answer) -> bool:
    """Same rule as the memory agent's eval: match within a small tolerance.

    exe_ans is rounded to 5 decimals (0.14136), so exact equality would fail
    most divisions. An empty plan (answer None) counts as wrong.
    """
    return answer is not None and math.isclose(answer, expected_answer, rel_tol=1e-3, abs_tol=1e-4)


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


if __name__ == "__main__":
    # No eval node or write_lesson: the baseline never learns, so the graph
    # is plan -> calculate -> record_turn. Scoring happens in the loop below.
    graph = StateGraph(CalucationState)
    graph.add_node("plan_calculation", plan_calculation)
    graph.add_node("calculate", calculate)
    graph.add_node("record_turn", record_turn)
    graph.add_edge(START, "plan_calculation")
    graph.add_edge("plan_calculation", "calculate")
    graph.add_edge("calculate", "record_turn")
    graph.add_edge("record_turn", END)
    # Short-term memory: the checkpointer saves each thread's state after every
    # invoke() and restores it on the next invoke() with the same thread_id.
    # One thread = one conversation, so `history` carries over between turns.
    #
    # The checkpoint stores the whole state, including the Pydantic `plan`.
    # LangGraph only restores types it is told are safe, so allow CalculationPlan.
    # __module__ is used because under `python -m` this module is "__main__".
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(CalculationPlan.__module__, "CalculationPlan")])
    compiled_graph = graph.compile(checkpointer=InMemorySaver(serde=serde))

    parser = argparse.ArgumentParser(description="Run the baseline agent (no memory).")
    parser.add_argument("--split", choices=["train", "dev"], default="train",
                        help="which data/{split}_turn.json to run on")
    # Keep equal to the memory agent's --max-turns for a fair comparison
    parser.add_argument("--max-turns", type=int, default=20,
                        help="run at least this many turns, then finish the current conversation")
    args = parser.parse_args()

    AGENT_NAME = "base_agent"
    RESULTS_PATH = f"processed/{AGENT_NAME}_{args.split}_results.json"
    RUN_ID = time.strftime("%Y%m%d-%H%M%S")   # tells repeated runs apart in Langfuse

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
    for idx in select_turns(records, args.max_turns):
        record = records[idx]
        # "Single_JKHY/2009/page_28.pdf-3_2" -> conversation "Single_JKHY/2009/page_28.pdf-3", turn 2
        conversation_id = record["id"].rsplit("_", 1)[0]
        test_state = {
            "question": record["annotation"]["cur_dial"][-1],
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
                    tags=[AGENT_NAME, args.split],
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
                # An error (bad reference, divide by zero, API error) counts as wrong
                #Try block so an error doesn't stop the whole block.
                row["error"] = repr(e)
                print(f"\n[{idx}] ERROR: {row['question']}\n  {e!r}")
            else:
                row["agent_program"] = format_program(final_state["plan"])
                row["agent_answer"] = final_state["final_answer"]
                row["correct"] = is_correct(final_state["final_answer"], expected_answer)
                print(f"\n[{idx}] {'RIGHT' if row['correct'] else 'WRONG'}: {row['question']}")
                print(f"  plan: {row['agent_program'] or '(empty)'}")
                print(f"  answer={row['agent_answer']}  expected={expected_answer}")

            if span is not None:
                span.update(output={
                    "program": row["agent_program"], "answer": row["agent_answer"],
                    "expected_program": expected_program, "expected_answer": expected_answer,
                    "error": row["error"],
                })
                # 1 = right, 0 = wrong or error. Filter on it in Langfuse to see failures.
                span.score_trace(name="correct", value=1 if row["correct"] else 0, data_type="BOOLEAN")
                # Direct link to this turn's trace (None if the project id can't be fetched)
                print(f"  trace: {langfuse.get_trace_url(trace_id=span.trace_id)}")
        rows.append(row)

    if langfuse:
        langfuse.flush()   # send queued traces before the script exits

    correct = sum(r["correct"] for r in rows)
    errors = sum(r["error"] is not None for r in rows)
    accuracy = correct / len(rows)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "agent": "base (no memory)",
            "split": args.split,
            "run_id": RUN_ID,
            "num_examples": len(rows),
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy,
            "examples": rows,
        }, f, indent=2)

    print(f"\nAccuracy: {correct}/{len(rows)} = {accuracy:.1%}  ({errors} errors)")
    print(f"Saved per-example results to {RESULTS_PATH}")
