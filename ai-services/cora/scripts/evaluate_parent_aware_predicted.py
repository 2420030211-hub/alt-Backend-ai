import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

BASE = Path("ai-services/cora")
EXP = BASE / "outputs/experiments/bert_independent_v1"
BERT_BASE = BASE / "outputs/experiments/bert_weighted_30epochs"
SVM_DIR = BASE / "outputs/baseline"
PA_DIR = BASE / "outputs/subissue_parent_aware"
OUT = BASE / "outputs/subissue_parent_aware_predicted"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["Product", "Sub-product", "Issue", "Sub-issue"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256

train = pd.read_csv(BASE / "data/cora_train.csv").fillna("")
val = pd.read_csv(BASE / "data/cora_validation.csv").fillna("")
maps = json.loads((EXP / "label_mappings.json").read_text())

class IndependentBERT(nn.Module):
    def __init__(self, encoder, counts):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.3)
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in counts])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = self.shared(out.last_hidden_state[:, 0, :])
        return [head(features) for head in self.heads]

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = list(texts)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        e = self.tokenizer(
            str(self.texts[i]), max_length=MAX_LENGTH,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in e.items()
                if k in ("input_ids", "attention_mask")}

print("Loading BERT...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(str(BERT_BASE / "tokenizer"))
encoder = AutoModel.from_pretrained(str(BERT_BASE / "encoder"))
bert = IndependentBERT(encoder, [len(maps[t]) for t in TARGETS])
state = torch.load(EXP / "best_model.pt", map_location="cpu", weights_only=True)
bert.load_state_dict(state, strict=True)
bert.to(DEVICE).eval()

inverse = {
    t: {int(i): label for label, i in maps[t].items()}
    for t in TARGETS
}
pred = {t: [] for t in TARGETS}
loader = DataLoader(
    TextDataset(val["text"].astype(str), tokenizer),
    batch_size=32, shuffle=False, num_workers=2
)

print("Predicting Product and Sub-product...", flush=True)
with torch.inference_mode():
    for batch in loader:
        out = bert(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE)
        )
        for idx, target in enumerate(TARGETS[:2]):
            pred[target].extend(
                inverse[target][i]
                for i in out[idx].argmax(dim=1).cpu().tolist()
            )

del bert, encoder
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Predicting Issue with existing SVM...", flush=True)
base_vec = joblib.load(SVM_DIR / "tfidf_vectorizer.joblib")
X = base_vec.transform(val["text"].astype(str))
issue_model = joblib.load(SVM_DIR / "linear_svc_Issue.joblib")
pred["Issue"] = issue_model.predict(X).tolist()

print("Predicting Sub-issue with predicted parent labels...", flush=True)
pa_vec = joblib.load(PA_DIR / "parent_aware_vectorizer.joblib")
pa_model = joblib.load(PA_DIR / "parent_aware_subissue_svm.joblib")

features = [
    "PRODUCT: " + str(p)
    + " SUBPRODUCT: " + str(sp)
    + " ISSUE: " + str(issue)
    + " COMPLAINT: " + str(text)
    for p, sp, issue, text in zip(
        pred["Product"], pred["Sub-product"], pred["Issue"],
        val["text"].astype(str)
    )
]
pred["Sub-issue"] = pa_model.predict(pa_vec.transform(features)).tolist()

def metric(target, predictions):
    truth = val[target].astype(str).tolist()
    labels = sorted(set(maps[target].keys()) | set(truth))
    return {
        "accuracy": round(float(accuracy_score(truth, predictions)), 4),
        "macro_f1": round(float(f1_score(
            truth, predictions, labels=labels,
            average="macro", zero_division=0
        )), 4)
    }

report = {t: metric(t, pred[t]) for t in TARGETS}
exact = np.ones(len(val), dtype=bool)
for t in TARGETS:
    exact &= val[t].astype(str).to_numpy() == np.asarray(pred[t], dtype=str)

report["overall"] = {
    "exact_match_accuracy": round(float(exact.mean()), 4),
    "exact_matches": int(exact.sum()),
    "rows": len(val),
    "parent_context": "Predicted BERT Product/Sub-product and SVM Issue"
}
(OUT / "metrics.json").write_text(json.dumps(report, indent=2))

print("\n===== PREDICTED-PARENT RESULTS =====")
for t in TARGETS:
    print(f"{t:12s} accuracy={report[t]['accuracy']:.4f} "
          f"macro_f1={report[t]['macro_f1']:.4f}")
print("Exact-match accuracy:", report["overall"]["exact_match_accuracy"])
print("Saved:", OUT / "metrics.json")
