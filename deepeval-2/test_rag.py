import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
import json


# Load dataset at module level for parametrize
# (parametrize decorators run at collection time, before fixtures)
with open("datasets/orion_golden.json") as f:
    CASES = json.load(f)

# ── BASIC TEST ──────────────────────────────────────────────
# Simplest form: one input, one output, one assertion
# fixtures are injected by name — pytest matches parameter name
# to fixture name in conftest.py automatically

def u16_u16_schedule_faithfulness(
        faithfulness_metric,  # Injected fixture from conftest.py
        schedule_context,    # Injected fixture from conftest.py
        ):
    tc= LLMTestCase(
        input="List U16 matches in Pune?",
        actual_output=
            "U16 Pune: March 23 at MCA Ground 11AM, March 29 at Balewadi 10AM",
        retrieval_context=schedule_context,
    )
    assert_test(tc, [faithfulness_metric])

# ── PARAMETRIZE — run same test logic across multiple inputs ──
# @pytest.mark.parametrize generates one test per tuple
# name it descriptively — shows up in test report

@pytest.mark.parametrize("user_input, expected_theme", [
                         ("Show U16 matches in Pune", "U16 Pune"),
    ("What matches are in Nashik?", "Nashik"),
    ("List all senior fixtures", "Senior"),
])
def test_answer_relevancy_parametrized(
        user_input,
        expected_theme,
        relevancy_metric,  # Injected fixture
        schedule_context,  # Injected fixture
):
     # Simulate your LLM response (in real tests, call your actual LLM)
    simulated_output = f"Here are the matches related to {expected_theme}..."
    tc = LLMTestCase(
        input=user_input,
        actual_output=simulated_output,
        retrieval_context=schedule_context,
    )
    assert_test(tc, [relevancy_metric])

# ── SKIP + MARK — control which tests run when ──────────────
# Use marks to group tests by category

@pytest.mark.slow #custom mark — run with: pytest -m slow
def test_full_pipeline_end_to_end(faithfulness_metric, relevancy_metric, schedule_context):
    tc= LLMTestCase(
        input="What matches are happening in Pune?",
        actual_output="U16 Pune: March 23 at MCA Ground 11AM, March 29 at Balewadi 10AM",
        retrieval_context=schedule_context,
    )
    assert_test(tc, [faithfulness_metric, relevancy_metric])

@pytest.mark.skip(reason="LLM endpoint not available in CI")
def test_live_llm_call():
    pass  # placeholder for live integration test

@pytest.mark.parametrize("case", CASES, ids=[c["input"][:40] for c in CASES])
def test_golden_cases(case, faithfulness_metric, relevancy_metric):
    tc = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        retrieval_context=case.get("context", [])
    )
    assert_test(tc, [faithfulness_metric, relevancy_metric])