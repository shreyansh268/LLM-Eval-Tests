# LLM Eval Tests

A single repository for evaluating **local LLMs** with multiple testing frameworks — starting with **DeepEval + Ollama**, with a growing suite of framework integrations under one roof.

> **Stack:** Python 3.10 · DeepEval 3.8 · Ollama · LangChain-Ollama · MLflow · pytest

---

## Repository Structure

```
LLM-Eval-Tests/
├── tests-deepeval/          # DeepEval metric-based evaluation (primary)
│   ├── conftest.py          # Shared OllamaDeepEvalModel fixture (JSON mode + extraction)
│   ├── pytest.ini           # asyncio_mode=auto, test discovery config
│   ├── requirements.txt
│   ├── test_tinyllama.py    # AnswerRelevancy + Faithfulness metrics
│   ├── test_ollama.py       # Connectivity smoke tests
│   ├── test_agent.py        # Simulated agent pipeline evaluation
│   ├── test_langgraph_agent.py  # Tool-using agent evaluation
│   ├── test_deepeval.py     # SummarizationMetric (Ollama + optional phi-2)
│   └── test_smoke_phi2.py   # LM Studio phi-2 smoke test
│
└── tests-mlflow/            # MLflow experiment tracking + metric logging
    ├── test_mlflow_phi2.py
    └── test_mlflow_multi_metrics_phi2.py
```

---

## DeepEval Implementation

### Why DeepEval?

[DeepEval](https://github.com/confident-ai/deepeval) treats LLM quality like a test suite — each metric is an assertion with a threshold, a score, and a reason. It integrates natively with `pytest`, so evaluations run as standard test cases with pass/fail gates.

### Core Design: `conftest.py`

The central piece of this repo is the `OllamaDeepEvalModel` wrapper in `conftest.py`. It solves the biggest pain point when using small local models as DeepEval judges:

> **Problem:** DeepEval metrics ask the judge LLM to return strict JSON. Small models like TinyLlama or Llama 3.2 often wrap JSON in prose or markdown fences, causing `ValueError: Evaluation LLM outputted an invalid JSON`.

**Three-layer solution:**

```
DeepEval metric calls generate(prompt)
        │
        ├─ judge_llm  (ChatOllama, format="json")  ← Layer 1: Ollama forces JSON token stream
        │
        └─ _extract_json(raw_response)              ← Layer 2: Strips any remaining prose/fences
             ├── json.loads() directly                  (already clean JSON)
             ├── regex: ```json ... ```                 (markdown code fence)
             └── regex: first { } or [ ] block          (JSON embedded in text)
```

**Two LLM instances — separated by purpose:**

| Instance | Mode | Used for |
|---|---|---|
| `judge_llm` | `format="json"` | Called by DeepEval metrics to score/reason |
| `chat_llm` | plain text | Generating `actual_output` in test cases |

```python
# conftest.py (key excerpt)
class OllamaDeepEvalModel(DeepEvalBaseLLM):

    def generate(self, prompt: str) -> str:
        """Judge path — JSON mode + extraction."""
        raw = self.judge_llm.invoke([HumanMessage(content=prompt)]).content
        return self._extract_json(raw)

    def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        """Test subject path — plain natural language."""
        return self.chat_llm.invoke(messages).content

@pytest.fixture(scope="session")
def local_model() -> OllamaDeepEvalModel:
    return OllamaDeepEvalModel(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
```

The fixture is **session-scoped** — one connection, reused across all test files.

---

### Metrics Covered

| Metric | File | What it checks |
|---|---|---|
| `AnswerRelevancyMetric` | `test_tinyllama.py`, `test_agent.py`, `test_ollama.py`, `test_langgraph_agent.py` | Is the answer on-topic for the input? |
| `FaithfulnessMetric` | `test_tinyllama.py` | Does the answer stick to the provided context? |
| `SummarizationMetric` | `test_deepeval.py` | Does the summary capture key information without hallucination? |

All metrics use `threshold=0.5` — tuned conservatively for small local models. Raise thresholds as you upgrade models.

---

### Setup & Run

**Prerequisites:**
```bash
# 1. Install and start Ollama
# https://ollama.com/download

# 2. Pull a model (llama3.2 recommended, tinyllama also works)
ollama pull llama3.2
ollama serve
```

**Install dependencies:**
```bash
cd tests-deepeval
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Run all tests:**
```bash
# Via pytest directly (recommended during development)
pytest -v -s

# Via DeepEval CLI (enables Confident AI dashboard tracking)
deepeval test run .
```

**Run a specific file:**
```bash
pytest test_tinyllama.py -v -s
deepeval test run test_tinyllama.py
```

---

### Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain_community` | Wrong package | Use `langchain-ollama`, not `langchain-community` |
| `ValueError: invalid JSON` | Small model ignores JSON format | `format="json"` on `judge_llm` + `_extract_json()` fallback in `conftest.py` |
| `404` from Ollama | Wrong endpoint or model name | Use `http://localhost:11434`, run `ollama list` to verify model name |
| `collected N items / N deselected / 0 selected` | `pytest-asyncio` strict mode | `asyncio_mode = auto` in `pytest.ini` |
| `deepeval.ob` import error | Module doesn't exist in v3.8+ | Removed — `observe` decorator is not in DeepEval 3.8 public API |

---

### Model Recommendations

| Model | Size | JSON reliability | Good for |
|---|---|---|---|
| `tinyllama` | 1.1B | Low | Smoke tests, connectivity checks only |
| `llama3.2` | 3B | Medium | Development & evaluation |
| `phi3:mini` | 3.8B | Medium-High | Better scoring accuracy |
| `mistral` | 7B | High | Near-production quality gates |

> **Tip:** Use a stronger model (`llama3.2` or above) as the **judge** even if the model under test is `tinyllama`. The judge only needs to produce reliable JSON scoring — it doesn't need to answer the original question.

---

## Extending to Other Frameworks

This repo is structured so each framework lives in its own folder with independent dependencies. Adding a new framework = adding a new `tests-<framework>/` directory.

```
LLM-Eval-Tests/
├── tests-deepeval/     ✅ Live — pytest-native metric evaluation
├── tests-mlflow/       ✅ Live — experiment tracking + metric logging
├── tests-ragas/        🔜 Planned — RAG-specific metrics (faithfulness, context recall)
├── tests-arize/        🔜 Planned — Observability traces via Arize Phoenix
├── tests-garak/        🔜 Planned — Security red-teaming (prompt injection, jailbreak)
└── tests-langsmith/    🔜 Planned — LangSmith tracing + dataset evaluation
```

**Why one repo?**
- Shared Ollama infrastructure (one server, multiple test consumers)
- Compare the same model output across multiple frameworks side-by-side
- Single CI/CD pipeline triggers all framework suites on a PR
- Reuse golden datasets and test fixtures across frameworks

**Adding RAGAS (example):**
```bash
mkdir tests-ragas && cd tests-ragas
pip install ragas langchain-ollama
# ragas uses the same ChatOllama pattern — copy conftest.py, swap metrics
```

---

## CI/CD Quality Gate (GitHub Actions)

```yaml
# .github/workflows/eval-on-pr.yml
name: LLM Eval Quality Gate
on: [pull_request]

jobs:
  deepeval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install Ollama
        run: curl -fsSL https://ollama.com/install.sh | sh

      - name: Pull model + start server
        run: |
          ollama pull llama3.2 &
          ollama serve &
          sleep 10

      - name: Install dependencies
        run: |
          cd tests-deepeval
          pip install -r requirements.txt

      - name: Run DeepEval suite
        run: |
          cd tests-deepeval
          pytest -v --tb=short
```

---

## Tech Stack

| Tool | Role |
|---|---|
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM metric evaluation framework |
| [Ollama](https://ollama.com) | Local LLM inference server |
| [LangChain-Ollama](https://python.langchain.com/docs/integrations/chat/ollama/) | ChatOllama integration (correct API, no 404s) |
| [MLflow](https://mlflow.org) | Experiment tracking, metric logging |
| pytest + pytest-asyncio | Test runner with async support |

---

## Contributing

Each `tests-<framework>/` folder is self-contained:
- Its own `requirements.txt`
- Its own `conftest.py` with framework-specific fixtures
- Its own `pytest.ini` if needed

PRs adding new framework integrations are welcome. Follow the existing pattern: smoke test first, then metric tests, then agent/pipeline tests.

---

*Built for practitioners who want production-grade LLM evaluation without cloud dependencies.*
