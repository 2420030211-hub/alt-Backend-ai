from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

BASE = Path("ai-services/cora")
train = pd.read_csv(BASE / "data/cora_train.csv").fillna("")
val = pd.read_csv(BASE / "data/cora_validation.csv").fillna("")

vectorizer = joblib.load(BASE / "outputs/baseline/tfidf_vectorizer.joblib")
model = joblib.load(BASE / "outputs/baseline/linear_svc_Sub_issue.joblib")

X_val = vectorizer.transform(val["text"].astype(str))
y_true = val["Sub-issue"].astype(str)
y_pred = model.predict(X_val)

results = pd.DataFrame({
    "actual": y_true,
    "predicted": y_pred,
    "text": val["text"].astype(str)
})

print("\n=== COMMON SUB-ISSUE MISCLASSIFICATIONS ===")
wrong = results[results["actual"] != results["predicted"]]
pairs = (
    wrong.groupby(["actual", "predicted"])
    .size()
    .sort_values(ascending=False)
    .head(20)
)
print(pairs.to_string())

print("\n=== EXAMPLES FOR THE WORST-SUPPORTED CLASSES ===")
counts = train["Sub-issue"].value_counts()
worst_classes = [
    "Funds not handled or disbursed as instructed",
    "Problem accessing account",
    "Attempted to collect wrong amount",
    "Debt is not mine",
]

for label in worst_classes:
    print(f"\n--- {label} | training rows: {counts.get(label, 0)} ---")
    subset = results[results["actual"] == label]
    for _, row in subset.head(3).iterrows():
        print("ACTUAL:", row["actual"])
        print("PREDICTED:", row["predicted"])
        print("TEXT:", row["text"][:500])
        print()
