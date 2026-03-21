# deepeval_diag.py
import inspect
import os
import requests

print("=== Part 1: deepeval signature and metrics ===")
try:
    from deepeval import assert_test
    import deepeval.metrics as dm
    print("assert_test signature:", inspect.signature(assert_test))
    members = [m for m in dir(dm) if not m.startswith('_')]
    print("metrics module members:", members)
    BaseMetric = getattr(dm, "BaseMetric", None)
    print("BaseMetric type:", type(BaseMetric))
    if BaseMetric is not None:
        print("BaseMetric methods:", [m for m in dir(BaseMetric) if not m.startswith('_')])
except Exception as e:
    print("ERROR while inspecting deepeval:", repr(e))

print("\n=== Part 2: probe local LM Studio endpoint ===")
try:
    url = os.getenv("PHI2_URL", "http://127.0.0.1:1234").rstrip("/") + "/v1/chat/completions"
    print("Probing URL:", url)
    r = requests.post(url, json={"model":"phi-2","messages":[{"role":"user","content":"Hello"}]}, timeout=5)
    print("STATUS:", r.status_code)
    print("BODY (first 2000 chars):")
    print(r.text[:2000])
except Exception as e:
    print("PROBE ERROR:", repr(e))