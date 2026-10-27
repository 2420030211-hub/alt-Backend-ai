from pathlib import Path
import joblib
import pandas as pd
from sklearn.svm import LinearSVC

BASE = Path("ai-services/cora")
DATA = BASE / "data"
BASELINE = BASE / "outputs/baseline"
OUT = BASE / "outputs/svm_c03"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "cora_train.csv")
vectorizer = joblib.load(BASELINE / "tfidf_vectorizer.joblib")
X_train = vectorizer.transform(train["text"].fillna("").astype(str))

for target in ["Issue", "Sub-issue"]:
    y = train[target].fillna("").astype(str)
    model = LinearSVC(C=0.3, random_state=42, max_iter=5000, dual="auto")
    model.fit(X_train, y)
    path = OUT / f"linear_svc_{target.replace('-', '_')}.joblib"
    joblib.dump(model, path)
    print(f"Saved {target}: {path}")

joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib")
print("Saved C=0.3 candidate models.")
