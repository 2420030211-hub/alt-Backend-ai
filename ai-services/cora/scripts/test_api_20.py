import json
import time
import urllib.request
import urllib.error
from pathlib import Path

URL = "http://127.0.0.1:8002"
complaints = [
    "I was charged an unauthorized fee on my credit card statement.",
    "My credit card payment was processed twice.",
    "I do not recognize a purchase on my credit card.",
    "The bank charged interest I did not agree to.",
    "My debit card was declined at a store.",
    "I cannot access my checking account online.",
    "There is a withdrawal I did not make.",
    "My savings account deposit has not appeared.",
    "The bank charged an unexpected maintenance fee.",
    "I transferred money but the recipient never received it.",
    "I think I was scammed through an online money transfer.",
    "My wire transfer is delayed.",
    "I was charged an incorrect personal loan fee.",
    "My loan payment was applied to the wrong account.",
    "The interest rate on my loan is incorrect.",
    "My mortgage escrow payment was calculated incorrectly.",
    "I am struggling to make my mortgage payments.",
    "My mortgage payment was not credited to my account.",
    "The bank reported incorrect information on my credit report.",
    "Customer service cannot explain a fee on my account.",
]
required = {
    "product", "sub_product", "issue", "sub_issue",
    "confidence", "routed_to", "hierarchy_valid", "model_version"
}
results = []

def get_json(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        URL + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.status, json.loads(response.read())

print("Checking health...")
try:
    status, health = get_json("/health")
    print("Health:", status, health)
    if status != 200 or not health.get("model_loaded"):
        raise RuntimeError("BERT model is not loaded.")
except Exception as e:
    print("API CHECK FAILED:", repr(e))
    print("Make sure the server is running in your first terminal on port 8002.")
    raise SystemExit(1)

print("\nTesting 20 complaints...")
latencies = []
for i, complaint in enumerate(complaints, 1):
    start = time.perf_counter()
    try:
        status, result = get_json("/classify", {"text": complaint})
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        confidence = result.get("confidence", {})
        valid_scores = all(
            isinstance(confidence.get(k), (int, float))
            and 0 <= confidence[k] <= 1
            for k in ("product", "sub_product", "issue", "sub_issue", "overall")
        )
        passed = (
            status == 200
            and required.issubset(result)
            and valid_scores
            and all(result.get(k) for k in ("product", "sub_product", "issue", "sub_issue"))
            and result.get("model_version") == "bert-base-uncased-hierarchical-v1"
        )
        record = {
            "test": i, "complaint": complaint, "passed": passed,
            "latency_seconds": round(elapsed, 3), "response": result,
        }
        print(
            f"{i:02}/20 {'PASS' if passed else 'FAIL'} "
            f"{elapsed:.2f}s | {result.get('product')} | {result.get('issue')}"
        )
    except Exception as e:
        record = {"test": i, "complaint": complaint, "passed": False, "error": repr(e)}
        print(f"{i:02}/20 FAIL: {e}")
    results.append(record)

out = Path("ai-services/cora/outputs/experiments/bert_base_gpu/api_test")
out.mkdir(parents=True, exist_ok=True)
passed_count = sum(r["passed"] for r in results)
summary = {
    "total": len(results),
    "passed": passed_count,
    "failed": len(results) - passed_count,
    "average_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
    "note": "API smoke tests are not accuracy measurements; labelled ground truth is required for accuracy and F1.",
    "results": results,
}
(out / "api_test_20_results.json").write_text(json.dumps(summary, indent=2))
(out / "api_test_20_summary.txt").write_text(
    f"Total: {len(results)}\nPassed: {passed_count}\n"
    f"Failed: {len(results)-passed_count}\n"
    f"Average latency: {summary['average_latency_seconds']} seconds\n"
    "Passing API tests does not mean predictions are correct.\n"
)
print("\n===== SUMMARY =====")
print(f"Passed: {passed_count}/20")
print(f"Failed: {len(results)-passed_count}/20")
print("Average latency:", summary["average_latency_seconds"], "seconds")
print("Reports saved to:", out)
