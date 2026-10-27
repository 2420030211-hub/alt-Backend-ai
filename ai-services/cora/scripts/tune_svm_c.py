import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

BASE = Path("ai-services/cora")
train = pd.read_csv(BASE / "data/cora_train.csv").fillna("")
val = pd.read_csv(BASE / "data/cora_validation.csv").fillna("")
baseline_dir = BASE / "outputs/baseline"
OUT = BASE / "outputs/svm_c_tuning"
OUT.mkdir(parents=True, exist_ok=True)

vectorizer = joblib.load(baseline_dir / "tfidf_vectorizer.joblib")
X_train = vectorizer.transform(train["text"].astype(str))
X_val = vectorizer.transform(val["text"].astype(str))

targets = ["Product", "Sub-product", "Issue", "Sub-issue"]
results = {}

for C in [0.1, 0.3, 3.0]:
    print(f"\n===== C={C} =====", flush=True)
    preds = {}
    results[str(C)] = {}

    for target in targets:
        y_train = train[target].astype(str)
        y_val = val[target].astype(str)

        model = LinearSVC(
            C=C, class_weight=None, random_state=42,
            max_iter=5000, dual="auto"
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        preds[target] = pred

        labels = sorted(set(y_train) | set(y_val))
        acc = accuracy_score(y_val, pred)
        mf1 = f1_score(
            y_val, pred, labels=labels,
            average="macro", zero_division=0
        )
        results[str(C)][target] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(mf1), 4)
        }
        print(
            f"{target:12s} accuracy={acc:.4f} macro-F1={mf1:.4f}",
            flush=True
        )

    exact = np.ones(len(val), dtype=bool)
    for target in targets:
        exact &= preds[target] == val[target].astype(str).to_numpy()

    results[str(C)]["exact_match_accuracy"] = round(float(exact.mean()), 4)
    print(f"Exact match: {exact.mean():.4f}")

baseline = json.loads((baseline_dir / "baseline_metrics.json").read_text())
results["baseline_C1.0"] = {
    target: {
        "accuracy": baseline["targets"][target]["accuracy"],
        "macro_f1": baseline["targets"][target]["macro_f1"]
    }
    for target in targets
}
results["baseline_C1.0"]["exact_match_accuracy"] = (
    baseline["overall_exact_match"]["exact_match_accuracy"]
)

path = OUT / "results.json"
path.write_text(json.dumps(results, indent=2))
print("\nSaved:", path)
