# test_smoke_phi2.py
# open cmd and lms load phi-2 and start lms server start
# open lmstudio and check in dev mode that phi-2 is running at http://1234
import os, requests

ENDPOINT = os.getenv("PHI2_URL", "http://127.0.0.1:1234").rstrip("/") + "/v1/chat/completions"

def call_phi2(prompt):
    r = requests.post(ENDPOINT, json={"model":"phi-2:2","messages":[{"role":"user","content":prompt}], "temperature":0.0}, timeout=10)
    r.raise_for_status()
    j = r.json()
    if "choices" in j and j["choices"]:
        c0 = j["choices"][0]
        if "message" in c0:
            return c0["message"].get("content","").strip()
        if "text" in c0:
            return c0.get("text","").strip()
    return str(j)

def test_smoke_summary():
    out = call_phi2("Summarize in 10 words: large language models")
    assert out and len(out) > 0