# test_tinyllama.py
# Core DeepEval metric tests using TinyLlama via Ollama.
# The OllamaDeepEvalModel judge (with JSON mode) lives in conftest.py.

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric


def test_ollama_is_reachable(local_model):
    """Smoke test: fails fast if Ollama is down or tinyllama isn't pulled."""
    reply = local_model.generate_text("Reply with only the word: pong")
    assert reply.strip(), (
        "Empty response from Ollama. "
        "Run: ollama pull tinyllama  and  ollama serve"
    )


def test_answer_relevancy(local_model):
    """TinyLlama answer to a factual question should score ≥ 0.5 on relevancy."""
    question = "What is the capital of France?"
    answer = local_model.generate_text(question)

    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=local_model,       # judge uses JSON mode internally
        include_reason=True,
    )
    test_case = LLMTestCase(input=question, actual_output=answer)
    assert_test(test_case, [metric])


def test_faithfulness(local_model):
    """Answer grounded in a given context should score ≥ 0.5 on faithfulness."""
    context = ["Paris is the capital and largest city of France."]
    question = "What is the capital of France?"
    answer = local_model.generate_text(
        "Based on this context: 'Paris is the capital of France.' "
        "Answer in one sentence: What is the capital of France?"
    )

    metric = FaithfulnessMetric(
        threshold=0.5,
        model=local_model,
        include_reason=True,
    )
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
    )
    assert_test(test_case, [metric])
