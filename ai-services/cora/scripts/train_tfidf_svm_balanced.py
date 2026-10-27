import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

BASE = Path("ai-services/cora")
TRAIN = BASE / "data/cora_train.csv"
VAL = BASE / "data/cora_validation.csv"
OUT = BASE / "outputs/baseline_balanced"
OLD = BASE / "outputs/baseline"
TARGETS = ["Product", "Sub-product", "Issue", "Sub-issue"]

train = pd.read_csv(TRAIN).fillna("")
val = pd.read_csv(VAL).fillna("")
OUT.mkdir(parents=True, exist_ok=True)

print("Fitting TF-IDF on training data only...", flush=True)
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
    min_df=2
)
X_train = vectorizer.fit_transform(train["text"].astype(str))
X_val = vectorizer.transform(val["text"].astype(str))
joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib")

predictions = {}
report = {"targets": {}}

for target in TARGETS:
    print(f"\nTraining balanced SVM: {target}", flush=True)
    y_train = train[target].astype(str)
    y_val = val[target].astype(str)

    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
        dual="auto"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    predictions[target] = pred

    suffix = target.replace("-", "_")
    joblib.dump(model, OUT / f"linear_svc_{suffix}.joblib")

    # Use the training label vocabulary plus validation labels so
    # unseen validation classes are not silently omitted from macro-F1.
    labels = sorted(set(y_train) | set(y_val))
    result = {
        "accuracy": round(accuracy_score(y_val, pred), 4),
        "macro_f1": round(f1_score(
            y_val, pred, labels=labels, average="macro", zero_division=0
        ), 4),
        "weighted_f1": round(f1_score(
            y_val, pred, average="weighted", zero_division=0
        ), 4)
    }
    report["targets"][target] = result
    pd.DataFrame(
        classification_report(
            y_val, pred, output_dict=True, zero_division=0
        )
    ).transpose().to_csv(OUT / f"report_{suffix}.csv")

    print(
        f"{target}: accuracy={result['accuracy']:.4f} "
        f"macro_f1={result['macro_f1']:.4f} "
        f"weighted_f1={result['weighted_f1']:.4f}",
        flush=True
    )

exact = np.ones(len(val), dtype=bool)
for target in TARGETS:
    exact &= predictions[target] == val[target].astype(str).to_numpy()

valid_paths = set(
    train[TARGETS].astype(str).itertuples(index=False, name=None)
)
consistent = sum(
    tuple(predictions[t][i] for t in TARGETS) in valid_paths
    for i in range(len(val))
)

report["overall"] = {
    "exact_matches": int(exact.sum()),
    "rows": len(val),
    "exact_match_accuracy": round(float(exact.mean()), 4),
    "hierarchy_consistency": round(consistent / len(val), 4)
}

(OUT / "balanced_metrics.json").write_text(json.dumps(report, indent=2))

print("\n===== COMPARISON: BASELINE vs BALANCED SVM =====")
old = json.loads((OLD / "baseline_metrics.json").read_text())
for target in TARGETS:
    a = old["targets"][target]
    b = report["targets"][target]
    print(
        f"{target:12s} "
        f"accuracy {a['accuracy']:.4f} -> {b['accuracy']:.4f} | "
        f"macro-F1 {a['macro_f1']:.4f} -> {b['macro_f1']:.4f}"
    )

print("\nBaseline exact match:",
      old["overall_exact_match"]["exact_match_accuracy"])
print("Balanced exact match:", report["overall"]["exact_match_accuracy"])
print("Balanced hierarchy consistency:",
      report["overall"]["hierarchy_consistency"])
print("Saved results:", OUT / "balanced_metrics.json")
