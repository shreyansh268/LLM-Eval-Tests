#Fixture file

# conftest.py
# pytest automatically loads this file before any test runs
# Think of it as your "test lab setup"

from flask import json
import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

# @pytest.fixture marks a function as reusable setup
# scope="session" means: create once, reuse across ALL tests in the run
# (not once per test — saves LLM client initialization time)

@pytest.fixture(scope="session")
def faithfulness_metric():
    """Reusable FaithfulnessMetric instance with our local Ollama model."""
    return FaithfulnessMetric(
        threshold=0.5,
        include_reason=True,
    )

@pytest.fixture(scope="session")
def relevancy_metric():
    return AnswerRelevancyMetric(threshold=0.75, verbose_mode=True)


@pytest.fixture(scope="session")
def schedule_context():
    # Your golden retrieval context — reused across tests
    return [
        "U16 Pune fixture: March 23, MCA Ground, 11:00 AM",
        "U16 Pune fixture: March 29, Balewadi Ground, 10:00 AM",
        "Senior Nashik fixture: March 24, Lam Ground, 8:00 AM",
    ]

@pytest.fixture(scope="session")
def golden_dataset():
        #load 50 test cases from a JSONL file
        with open("tests-deepeval/golden_dataset.jsonl") as f:
            CASES = json.load(f)
 # Each record: {"input": "...", "actual_output": "...", "context": [...]}