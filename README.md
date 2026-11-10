# ConvFinQA calculation agent

A LangGraph agent that answers numerical questions about financial reports from the
[ConvFinQA](https://github.com/czyssrs/ConvFinQA) dataset. It learns from its mistakes:
wrong answers become short lessons that are retrieved for similar questions later.

## How it works

```
question + report text + table
        │
        ▼
plan_calculation ──► calculate ──► eval ──right──► END
 (LLM writes a       (Python runs     │
  plan of add /       each step)      └─wrong──► write_lesson ──► END
  subtract / ...)                                (LLM compares with the expected
                                                  program and stores a lesson)
```

- **Planner**: `gpt-4.1-mini` returns a structured plan such as
  `subtract(206588, 181001), divide(#0, 181001)`. `#0` refers to the first step's result.
- **Calculator**: plain Python does the arithmetic, so the LLM never computes numbers itself.
- **Eval**: compares the final answer with the dataset's `exe_ans` (with a small tolerance).
- **Memory**: lessons are stored in a LangGraph `InMemoryStore` with OpenAI embeddings on
  the `question` field. The planner retrieves the 3 most similar lessons for each new
  question. Lessons are saved to `processed/lessons.json` every 10 examples.

## Project layout

| Path | What it is |
|---|---|
| `agent/ai_agent_procedural_memory.py` | The agent with lesson memory: graph, prompts, and a run over the first 15 training turns |
| `agent/base_agent.py` | Baseline agent without memory (planner + calculator only), kept for comparison |
| `agent/table_format.py` | Cleans dataset tables and converts them to Markdown or row sentences |
| `processed/lessons.json` | Lessons saved from the latest run |
| `tests/` | pytest tests for the table formatter |
| `docs/decisions.md` | Design decisions |

## Setup

1. Python 3.10+, then install the dependencies:
   ```bash
   python -m pip install langgraph langchain-openai python-dotenv pydantic pytest
   ```
2. Download the ConvFinQA data into `data/` (needs `data/train_turn.json`).
   The data is not in git because `train_turn.json` is 159 MB.
3. Create a `.env` file with your key (it is gitignored):
   ```
   OPENAI_API_KEY=...
   ```

## Run

Run from the repository root, as a module (`-m`), so `from agent.table_format import ...`
and the `data/` and `processed/` paths resolve:

```bash
python -m agent.ai_agent_procedural_memory   # runs 15 examples (about 30-60 API calls)
python -m pytest                             # table formatter tests, no API calls
```

Each example prints RIGHT or WRONG, the plan, and the answer next to the expected one.

## Known limitations

- Lookup-only turns (the answer is a single table value) can't be written as two-argument operations.
- Follow-up turns like "what about in 2008?" only get the current question, not the earlier ones.
- `InMemoryStore` lives only for one run; `processed/lessons.json` is the saved copy.
