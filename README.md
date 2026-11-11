# ConvFinQA calculation agent

A LangGraph agent that answers numerical questions about financial reports from the
[ConvFinQA](https://github.com/czyssrs/ConvFinQA) dataset. It learns from its mistakes:
wrong answers become short lessons that are retrieved for similar questions later.
A baseline agent without memory is kept for comparison.

## How it works

```
question + report text + table + earlier turns
        │
        ▼
plan_calculation ──► calculate ──► eval ──right──────────────────► record_turn ──► END
 (LLM writes a       (Python runs     │                                ▲
  plan, or a          each step)      └─wrong──► write_lesson ─────────┘
  lookup_value)                                  (LLM compares with the expected
                                                  program and may store a lesson)
```

- **Planner**: `gpt-4.1-mini` returns a structured plan such as
  `subtract(206588, 181001), divide(#0, 181001)`. Each argument is either a number
  (`{"value": 206588}`) or an earlier step's result (`{"step": 0}`, written `#0`).
  When the answer is a single number from the table or text, it sets `lookup_value` instead.
- **Calculator**: plain Python does the arithmetic, so the LLM never computes numbers itself.
- **Eval**: compares the final answer with the dataset's `exe_ans` (with a small tolerance).
- **Conversation history (short-term memory)**: a LangGraph checkpointer (`InMemorySaver`)
  keeps each conversation's earlier questions, plans and answers, using the conversation id
  as `thread_id`. `record_turn` runs last, so during a turn the agent only sees earlier turns.
  The history holds the agent's own answers, so an early mistake can carry forward.
- **Lessons (long-term memory, memory agent only)**: an `InMemoryStore` with OpenAI
  embeddings.
  - Each lesson has an `applies_when` condition (the type of calculation, e.g. "the question
    asks for a percentage change between two periods"), and that is what gets embedded.
  - The planner gets up to 3 lessons whose condition is close to the new question, keeping
    only those with similarity ≥ `MIN_LESSON_SCORE` (0.5).
  - Reflection labels each mistake with a category and can decline to save a lesson that
    couldn't have prevented it, or that duplicates a retrieved one.
  - Lessons are saved to `processed/lessons.json` every 10 examples.

## Project layout

| Path | What it is |
|---|---|
| `agent/ai_agent_procedural_memory.py` | The agent with lesson memory: graph, prompts, and a run over train or dev turns |
| `agent/base_agent.py` | Baseline without lessons (same prompt, plan schema and conversation history), for comparison |
| `agent/table_format.py` | Cleans dataset tables and converts them to Markdown or row sentences |
| `processed/lessons.json` | Lessons learned in the latest train run; loaded by frozen-mode runs |
| `processed/*_results.json` | Per-example results and accuracy from each agent's latest run |
| `tests/` | pytest tests for the table formatter |
| `docs/decisions.md` | Design decisions |

## Setup

1. Python 3.10+, then install the dependencies:
   ```bash
   python -m pip install langgraph "langchain>=1,<2" langchain-openai langfuse python-dotenv pydantic pytest
   ```
   `langchain` itself is only needed by Langfuse's LangChain integration, which
   imports it to detect the LangChain version.
2. Download the ConvFinQA data into `data/` (needs `data/train_turn.json`, and
   `data/dev_turn.json` for held-out runs). The data is not in git because
   `train_turn.json` is 159 MB.
3. Create a `.env` file with your keys (it is gitignored):
   ```
   OPENAI_API_KEY=...
   # Optional: trace every turn to Langfuse
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_SECRET_KEY=...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   ```

## Run

Run from the repository root, as a module (`-m`), so `from agent.table_format import ...`
and the `data/` and `processed/` paths resolve:

Both agents take `--split train|dev` (default `train`) and `--max-turns N` (default 20).
A run takes at least N turns, then finishes the current conversation so none is cut off.

```bash
# 1. Learn lessons on train (writes processed/lessons.json)
python -m agent.ai_agent_procedural_memory --max-turns 100

# 2. Held-out comparison on dev
python -m agent.base_agent --split dev --max-turns 100
python -m agent.ai_agent_procedural_memory --split dev --max-turns 100

python -m pytest   # table formatter tests, no API calls
```

The memory agent's `--lessons` mode defaults to `learn` on train and `frozen` on dev:

| Mode | Lessons at start | Writes new lessons |
|---|---|---|
| `learn` | only the seeded percent rule | yes, to `processed/lessons.json` (or `lessons_dev.json` on dev) |
| `frozen` | loaded from `processed/lessons.json` | no, so dev measures whether train lessons help on new questions |

API calls: the base agent makes about 1 per turn; the memory agent about 2–4 when
learning, plus one embedding call per lesson loaded in frozen mode.

Each example prints RIGHT or WRONG, the plan, and the answer next to the expected one.
The memory agent also prints the lessons it used (key and similarity score) and whether a
lesson was saved or skipped. Results go to `processed/{agent}_{split}_results.json`
(e.g. `processed/memory_agent_dev_results.json`), with the same row layout for both agents
so they can be compared per example and by `turn_index`.

### Tracing with Langfuse

When the Langfuse keys are set, each turn becomes one trace:

- **Name and tags:** the trace is named after the agent and tagged with the agent, the
  split and (for memory) the lessons mode.
- **Contents:** each graph node and LLM call; the output holds the agent's and the
  expected program and answer.
- **Score:** a `correct` score (1/0), so you can filter to wrong answers.
- **Sessions:** each conversation in a run is one session, so its turns are shown
  together.

The embedding calls made by the store for lesson search are not traced.

## Known limitations

- In `learn` mode, lessons are learned and evaluated on the same turns; use the frozen dev
  run for a held-out number.
- `InMemoryStore` and the checkpointer live only for one run; `processed/lessons.json`
  is the saved copy (frozen mode reloads it, re-embedding each lesson).
- `MIN_LESSON_SCORE` (0.5) is a starting guess; tune it from the logged scores.
- If a turn raises an error, it isn't added to the history, so later turns in that
  conversation have a gap.
