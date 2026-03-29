import subprocess
import pytest
import json

@pytest.mark.load
def test_rag_load_performance():
    """Run Locust headless and assert on exit code + stats."""
    result = subprocess.run([
        "locust",
        "--headless",
        "--locustfile", "tests/load/locustfile.py",
        "--host", "http://localhost:8000",
        "--users", "20",
        "--spawn-rate", "5",
        "--run-time", "2m",
        "--csv", "tests/load/results",
        "--exit-code-on-error", "1"
    ], capture_output=True, text=True)
    
    print("Locust STDOUT:", result.stdout)
    print("Locust STDERR:", result.stderr)

#parse locus csv stats
import csv
with open("tests/load/results_stats.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Name"] == "POST /api/query":
            p90=float(row["90%"])
            error_pct=float(row["Failure Count"]) / float(row["Request Count"]) * 100

            assert p90 < 10000, f"90th percentile latency too high: {p90}ms"
            assert error_pct < 5, f"Failure percentage too high: {error_pct}%"
        
    assert result.returncode == 0, "Load test failed — check quality scores"