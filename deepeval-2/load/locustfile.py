import random
from locust import HttpUser, task, between, events
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import json
import threading

# Thread-safe results collector
results_lock = threading.Lock()
quality_results = []

QUESTIONS = [
    {"input": "What is the refund policy?", "context": ["Refunds accepted within 30 days."]},
    {"input": "How do I reset my password?", "context": ["Use the forgot password link on login page."]},
    {"input": "What payment methods are accepted?", "context": ["We accept Visa, Mastercard, and UPI."]},
]

# Sample rate — evaluate 1 in N responses (DeepEval calls LLM, keep costs sane)
EVAL_SAMPLE_RATE = 0.2

relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
faithfulness_metric = FaithfulnessMetric(threshold=0.7)


class RAGUser(HttpUser):
    wait_time = between(2, 5)

    @task
    def query_rag(self):
        question = random.choice(QUESTIONS)

        with self.client.post(
            "/api/query",
            json={"question": question["input"], "top_k": 5},
            catch_response=True
        ) as response:

            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            body = response.json()
            answer = body.get("answer", "")

            if not answer:
                response.failure("Empty answer field")
                return

            response.success()

            # Probabilistic DeepEval sampling (keep cost low under load)
            if random.random() < EVAL_SAMPLE_RATE:
                self._run_deepeval(question["input"], answer, question["context"])

    def _run_deepeval(self, input_text, actual_output, retrieval_context):
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        relevancy_metric.measure(test_case)
        faithfulness_metric.measure(test_case)

        with results_lock:
            quality_results.append({
                "input": input_text,
                "answer": actual_output,
                "relevancy_score": relevancy_metric.score,
                "relevancy_pass": relevancy_metric.is_successful(),
                "faithfulness_score": faithfulness_metric.score,
                "faithfulness_pass": faithfulness_metric.is_successful(),
            })


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print quality summary when load test ends."""
    if not quality_results:
        return

    total = len(quality_results)
    rel_pass = sum(1 for r in quality_results if r["relevancy_pass"])
    faith_pass = sum(1 for r in quality_results if r["faithfulness_pass"])
    avg_rel = sum(r["relevancy_score"] for r in quality_results) / total
    avg_faith = sum(r["faithfulness_score"] for r in quality_results) / total

    print(f"\n{'='*50}")
    print(f"QUALITY UNDER LOAD SUMMARY ({total} sampled responses)")
    print(f"  Relevancy:    {rel_pass}/{total} passed | avg score: {avg_rel:.2f}")
    print(f"  Faithfulness: {faith_pass}/{total} passed | avg score: {avg_faith:.2f}")
    print(f"{'='*50}")

    # Fail the load test if quality degrades
    if avg_rel < 0.6 or avg_faith < 0.6:
        environment.process_exit_code = 1