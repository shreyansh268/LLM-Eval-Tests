# test_mlflow_phi2.py
import os, requests, mlflow

PHI2 = os.getenv("PHI2_URL","http://127.0.0.1:1234").rstrip("/") + "/v1/chat/completions"
mlflow.set_experiment("phi2-eval")

def call_phi2(prompt):
    r = requests.post(PHI2, json={"model":"phi-2:2","messages":[{"role":"user","content":prompt}], "temperature":0.0}, timeout=20)
    r.raise_for_status()
    j = r.json()
    c = j.get("choices",[{}])[0]
    return (c.get("message") or {}).get("content","") or c.get("text","") or str(j)

def exact_match(pred, ref):
    return 1.0 if pred.strip() == ref.strip() else 0.0

def run():
    prompt = "Summarize: The quick brown fox jumps over the lazy dog."
    ref = "A fox leaps over a lazy dog."
    pred = call_phi2(prompt)
    score = exact_match(pred, ref)
    with mlflow.start_run():
        mlflow.log_param("prompt", prompt)
        mlflow.log_metric("exact_match", score)
        mlflow.log_text(pred, "prediction.txt")
    print("pred:", pred, "score:", score)

if __name__ == '__main__':
    run()