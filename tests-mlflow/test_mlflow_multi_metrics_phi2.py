# mlflow_multi_metrics_phi2.py
import os, requests, json, math
import mlflow
from typing import List

PHI2_URL = os.getenv("PHI2_URL", "http://127.0.0.1:1234").rstrip("/")
GEN_ENDPOINT = PHI2_URL + "/v1/chat/completions"
MODEL_ID = os.getenv("PHI2_MODEL", "phi-2:2")

# Optional metric libs (install below)
try:
    import sacrebleu
except Exception:
    sacrebleu = None
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
try:
    from bert_score import score as bert_score_fn
except Exception:
    bert_score_fn = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None

def call_phi2(prompt: str, timeout: int = 20) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    r = requests.post(GEN_ENDPOINT, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and "choices" in j and j["choices"]:
        c0 = j["choices"][0]
        if isinstance(c0, dict) and "message" in c0 and isinstance(c0["message"], dict):
            return c0["message"].get("content", "").strip()
        if isinstance(c0, dict) and "text" in c0:
            return c0.get("text", "").strip()
    if isinstance(j, dict) and "output" in j and isinstance(j["output"], str):
        return j["output"].strip()
    return str(j).strip()

def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip() == ref.strip() else 0.0

def compute_bleu(preds: List[str], refs: List[str]) -> float:
    if not sacrebleu:
        raise RuntimeError("sacrebleu not installed")
    # sacrebleu expects list of references as list of lists
    return sacrebleu.corpus_bleu(preds, [refs]).score

def compute_rouge(pred: str, ref: str):
    if not rouge_scorer:
        raise RuntimeError("rouge_score not installed")
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    return scorer.score(ref, pred)  # note: rouge_scorer expects (target, prediction)

def compute_bertscore(preds: List[str], refs: List[str]):
    if not bert_score_fn:
        raise RuntimeError("bert_score not installed")
    P, R, F1 = bert_score_fn(preds, refs, lang="en", rescale_with_baseline=True)
    # return average F1 as float
    return float(F1.mean().item())

def compute_embedding_cosine(pred: str, ref: str, model_name="all-MiniLM-L6-v2"):
    if SentenceTransformer is None or cosine_similarity is None:
        raise RuntimeError("sentence-transformers or sklearn not installed")
    model = SentenceTransformer(model_name)
    emb = model.encode([pred, ref], convert_to_numpy=True)
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float(sim)

def main():
    prompt = "Summarize: The quick brown fox jumps over the lazy dog."
    reference = "A fox leaps over a lazy dog."

    print("Calling model:", MODEL_ID, "endpoint:", GEN_ENDPOINT)
    pred = call_phi2(prompt)
    print("Prediction:", pred)

    mlflow.set_experiment("phi2-multi-metrics")
    with mlflow.start_run():
        mlflow.log_param("model_id", MODEL_ID)
        mlflow.log_param("prompt", prompt)

        # Exact match
        em = exact_match(pred, reference)
        mlflow.log_metric("exact_match", em)
        print("ExactMatch:", em)

        # BLEU
        if sacrebleu:
            try:
                bleu = compute_bleu([pred], [reference])
                mlflow.log_metric("bleu", bleu)
                print("BLEU:", bleu)
            except Exception as e:
                print("BLEU error:", e)
        else:
            print("BLEU skipped (sacrebleu not installed)")

        # ROUGE
        if rouge_scorer:
            try:
                rouge_scores = compute_rouge(pred, reference)
                # log ROUGE-L F1 and ROUGE-1 F1
                mlflow.log_metric("rouge1_f1", rouge_scores["rouge1"].fmeasure)
                mlflow.log_metric("rouge2_f1", rouge_scores["rouge2"].fmeasure)
                mlflow.log_metric("rougeL_f1", rouge_scores["rougeL"].fmeasure)
                print("ROUGE:", rouge_scores)
            except Exception as e:
                print("ROUGE error:", e)
        else:
            print("ROUGE skipped (rouge_score not installed)")

        # BERTScore
        if bert_score_fn:
            try:
                bert_f1 = compute_bertscore([pred], [reference])
                mlflow.log_metric("bertscore_f1", bert_f1)
                print("BERTScore F1:", bert_f1)
            except Exception as e:
                print("BERTScore error:", e)
        else:
            print("BERTScore skipped (bert_score not installed)")

        # Embedding cosine
        if SentenceTransformer is not None:
            try:
                emb_sim = compute_embedding_cosine(pred, reference)
                mlflow.log_metric("embedding_cosine", emb_sim)
                print("Embedding cosine:", emb_sim)
            except Exception as e:
                print("Embedding error:", e)
        else:
            print("Embedding similarity skipped (sentence-transformers or sklearn not installed)")

        # Save prediction and reference as artifacts
        mlflow.log_text(pred, "prediction.txt")
        mlflow.log_text(reference, "reference.txt")

if __name__ == "__main__":
    main()