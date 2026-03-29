from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision
from datasets import Dataset

# load golden dataset
data = Dataset.from_dict({
    "question": [...],
    "answer": [...],
    "contexts": [...],
    "ground_truth": [...]
})

result = evaluate(data, metrics=[faithfulness, context_recall, context_precision])

# Quality gate — fail the CI step if below threshold
assert result["faithfulness"] >= 0.85,    f"Faithfulness {result['faithfulness']:.2f} below 0.85"
assert result["context_recall"] >= 0.80,  f"Context recall {result['context_recall']:.2f} below 0.80"
assert result["context_precision"] >= 0.70, f"Precision {result['context_precision']:.2f} below 0.70"

print("All RAG quality gates passed.", result)