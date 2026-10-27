from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

BASE = Path("ai-services/cora")
DATA = BASE / "data"
OUT = BASE / "outputs/subissue_parent_aware_c3"
OUT.mkdir(parents=True, exist_ok=True)

PARENTS = ["Product", "Sub-product", "Issue"]
TARGET = "Sub-issue"

train = pd.read_csv(DATA / "cora_train.csv").fillna("")
val = pd.read_csv(DATA / "cora_validation.csv").fillna("")

def make_features(df):
    # Explicit parent context plus complaint text.
    parts = []
    for _, row in df.iterrows():
        parts.append(
            "PRODUCT: " + str(row["Product"])
            + " SUBPRODUCT: " + str(row["Sub-product"])
            + " ISSUE: " + str(row["Issue"])
            + " COMPLAINT: " + str(row["text"])
        )
    return parts

print("Preparing parent-aware text features...", flush=True)
x_train_text = make_features(train)
x_val_text = make_features(val)

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
    min_df=2,
)

X_train = vectorizer.fit_transform(x_train_text)
X_val = vectorizer.transform(x_val_text)

y_train = train[TARGET].astype(str)
y_val = val[TARGET].astype(str)

print("Training parent-aware Sub-issue classifier...", flush=True)
model = LinearSVC(C=3.0, random_state=42, max_iter=5000, dual="auto")
model.fit(X_train, y_train)

pred = model.predict(X_val)

# Compare on the exact same validation rows with the existing text-only SVM.
baseline_dir = BASE / "outputs/baseline"
baseline_vectorizer = joblib.load(baseline_dir / "tfidf_vectorizer.joblib")
baseline_model = joblib.load(
    baseline_dir / "linear_svc_Sub_issue.joblib"
)
baseline_pred = baseline_model.predict(
    baseline_vectorizer.transform(val["text"].astype(str))
)

def metrics(truth, predictions):
    return {
        "accuracy": round(float(accuracy_score(truth, predictions)), 4),
        "macro_f1": round(float(f1_score(
            truth, predictions,
            labels=sorted(set(y_train) | set(y_val)),
            average="macro",
            zero_division=0
        )), 4),
    }

results = {
    "text_only_svm_c1": metrics(y_val, baseline_pred),
    "parent_aware_svm_c1_oracle_parents": metrics(y_val, pred),
    "note": (
        "Parent-aware score uses true validation parent labels. "
        "It is an oracle-context experiment, not a production estimate."
    ),
}

joblib.dump(vectorizer, OUT / "parent_aware_vectorizer.joblib")
joblib.dump(model, OUT / "parent_aware_subissue_svm.joblib")
(OUT / "metrics.json").write_text(json.dumps(results, indent=2))

print("\n===== SUB-ISSUE EXPERIMENT =====")
print(json.dumps(results, indent=2))
print("Saved experimental model and metrics to:", OUT)
