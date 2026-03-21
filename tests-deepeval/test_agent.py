# test_agent.py
# Simulated agent pipeline (no crewai required).
# OllamaDeepEvalModel (with JSON judge mode) lives in conftest.py.

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


# ── Simulated agent (replace body with CrewAI once installed) ─────────────────

def local_agent(local_model, query: str) -> str:
    """
    Simulates a researcher agent via system+user prompt.
    Uses generate_text() — plain text mode, NOT JSON mode.
    """
    return local_model.generate_text(
        prompt=f"Research and summarize: {query}",
        system=(
            "You are a research analyst. Given a query, provide a concise, "
            "factual summary with key bullet points."
        ),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_agent_connection(local_model):
    """Verify the agent LLM is reachable before evaluation tests run."""
    reply = local_agent(local_model, "Say 'ok' in one word")
    assert reply.strip(), (
        "Empty response — check Ollama is running and tinyllama is pulled."
    )


def test_agent_answer_relevancy(local_model):
    """Agent response about IPL Bengaluru should score ≥ 0.5 on relevancy."""
    query = "IPL Bengaluru teams"
    output = local_agent(local_model, query)
    assert output.strip(), f"Agent returned empty output for: {query!r}"

    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=local_model,   # judge_llm uses JSON mode inside conftest wrapper
        include_reason=True,
    )
    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [metric])
