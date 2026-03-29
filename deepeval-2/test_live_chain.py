import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

#actual langchain chain
from orion_league.chain import build_schedule.chain, get_retriever

@pytest.fixture(scope="session")
def chain():
    return build_schedule.chain()

@pytest.fixture(scope="session")
def retriever():
    return get_retriever()

@pytest.mark.chain
def test_live_chain(chain, retriever, faithfulness_metric, relevancy_metric):
    """End-to-end test of the live chain with real LLM calls."""
    user_input = "What matches are happening in Pune?"
    # Step 1 — run the real chain, capture actual output
    actual_output = chain.invoke(user_input)

    # Step 2 — capture what the retriever actually fetched
    retrieved_docs = retriever.invoke(user_input)
    retrieval_context = retriever.retrieve(user_input)

    tc = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )
    assert_test(tc, [faithfulness_metric, relevancy_metric])