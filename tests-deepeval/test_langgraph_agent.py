# test_langgraph_agent.py
# Simulates a tool-using agent (no langgraph required).
# OllamaDeepEvalModel (with JSON judge mode) lives in conftest.py.

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


# ── Simulated tool (replace with real LangGraph tool once installed) ──────────

def research_tool(query: str) -> str:
    """Returns canned sports context — simulates a tool call."""
    return "IPL Bengaluru: RCB (Royal Challengers Bengaluru) is the key team for 2026 season."


def agent_executor(local_model, query: str) -> str:
    """
    Simulates a tool-using agent:
      1. Call the tool to get context
      2. Ask the LLM to synthesise a final answer from that context
    Uses generate_text() — plain text mode, NOT JSON mode.
    """
    tool_output = research_tool(query)
    return local_model.generate_text(
        prompt=(
            f"Tool result: {tool_output}\n\n"
            f"User question: {query}\n\n"
            "Answer based on the tool result above:"
        ),
        system=(
            "You are a sports research assistant. Use the provided tool result "
            "to answer the user's question concisely."
        ),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_agent_executor_connection(local_model):
    """Verify LLM is reachable before agent tests run."""
    reply = local_model.generate_text("Say 'ok' in one word")
    assert reply.strip(), "Empty response — check Ollama is running and tinyllama is pulled."


def test_agent_rcb_relevancy(local_model):
    """Agent response about RCB IPL should score ≥ 0.5 on relevancy."""
    query = "RCB IPL status?"
    output = agent_executor(local_model, query)
    assert output.strip(), f"Agent returned empty output for: {query!r}"

    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=local_model,   # judge uses JSON mode via conftest wrapper
        include_reason=True,
    )
    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [metric])
