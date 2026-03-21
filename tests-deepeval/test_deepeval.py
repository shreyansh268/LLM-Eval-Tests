# test_deepeval.py
# Summarization metric tests.
# OllamaDeepEvalModel (with JSON judge mode) lives in conftest.py.

import os
import requests
import traceback

import pytest
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import SummarizationMetric


# ── Local LM Studio phi-2 helper (optional — skipped if server is offline) ───

PHI2_URL = os.getenv("PHI2_URL", "http://127.0.0.1:1234").rstrip("/")
GEN_ENDPOINT = PHI2_URL + "/v1/chat/completions"


def call_local_phi2(prompt: str, timeout: int = 20) -> str:
    """Call local LM Studio phi-2 endpoint (OpenAI-compatible API)."""
    payload = {
        "model": "phi-2:2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    r = requests.post(GEN_ENDPOINT, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and "choices" in j and j["choices"]:
        c0 = j["choices"][0]
        if "message" in c0:
            return c0["message"].get("content", "").strip()
        if "text" in c0:
            return c0.get("text", "").strip()
        if "output" in c0:
            return c0.get("output", "").strip()
    if isinstance(j, dict) and "output" in j:
        return str(j["output"]).strip()
    return str(j).strip()


def phi2_available() -> bool:
    """Returns True only when LM Studio phi-2 server is reachable."""
    try:
        requests.get(PHI2_URL, timeout=2)
        return True
    except Exception:
        return False


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_summarization_with_ollama(local_model):
    """
    SummarizationMetric: TinyLlama generates the summary; acts as judge too.
    generate_text() → plain-text summary (not JSON mode).
    local_model as metric judge → JSON mode via conftest wrapper.
    """
    original = "The quick brown fox jumps over the lazy dog."
    summary = local_model.generate_text(
        f"Summarize this in one sentence: {original}"
    )
    assert summary.strip(), "TinyLlama returned an empty summary."

    metric = SummarizationMetric(
        threshold=0.5,
        model=local_model,
        include_reason=True,
    )
    test_case = LLMTestCase(input=original, actual_output=summary)
    assert_test(test_case, [metric])


@pytest.mark.skipif(
    not phi2_available(),
    reason="LM Studio phi-2 server not running at 127.0.0.1:1234",
)
def test_summarization_with_phi2(local_model):
    """
    SummarizationMetric: phi-2 (LM Studio) generates the summary;
    TinyLlama judges it. Skipped automatically when LM Studio is offline.
    """
    original = "The quick brown fox jumps over the lazy dog."
    try:
        summary = call_local_phi2(f"Summarize in one sentence: {original}")
    except Exception:
        traceback.print_exc()
        pytest.fail("phi-2 call failed — see traceback above.")

    assert summary.strip(), "phi-2 returned an empty summary."

    metric = SummarizationMetric(
        threshold=0.5,
        model=local_model,
        include_reason=True,
    )
    test_case = LLMTestCase(input=original, actual_output=summary)
    assert_test(test_case, [metric])
