# test_ollama.py
# Smoke tests: verify Ollama is reachable and TinyLlama responds correctly.
# OllamaDeepEvalModel (with JSON judge mode) lives in conftest.py.

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


def test_ollama_connection(local_model):
    """Verify Ollama is running and tinyllama model is available."""
    reply = local_model.generate_text("Reply with one word: hello")
    assert reply.strip(), (
        "Empty response from Ollama. "
        "Run: ollama serve  and  ollama pull tinyllama"
    )


def test_ollama_answer_relevancy(local_model):
    """TinyLlama answer should score ≥ 0.5 on relevancy."""
    query = "Summarize RCB IPL 2026 status"
    # generate_text → plain text (NOT JSON mode) for the answer under test
    output = local_model.generate_text(query)
    assert output.strip(), f"Empty output for: {query!r}"

    # local_model as metric judge → uses JSON mode internally via conftest
    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=local_model,
        include_reason=True,
    )
    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [metric])
