import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

BASE = Path("ai-services/cora")
EXP = BASE / "outputs/experiments/bert_independent_v1"
BERT_BASE = BASE / "outputs/experiments/bert_weighted_30epochs"
SVM_DIR = BASE / "outputs/svm_c03"
OUT = BASE / "outputs/hybrid_c03_evaluation"

TARGETS = ["Product", "Sub-product", "Issue", "Sub-issue"]
BERT_TARGETS = ["Product", "Sub-product"]
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(BASE / "data/cora_train.csv").fillna("")
val_df = pd.read_csv(BASE / "data/cora_validation.csv").fillna("")
maps = json.loads((EXP / "label_mappings.json").read_text())

class IndependentBERT(nn.Module):
    def __init__(self, encoder, counts):
        super().__init__()
        self.encoder = encoder
        hidden = self.encoder.config.hidden_size
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n) for n in counts
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        features = self.shared(out.last_hidden_state[:, 0, :])
        return [head(features) for head in self.heads]

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    encoder_dir = BERT_BASE / "encoder"
    tokenizer_dir = BERT_BASE / "tokenizer"
    checkpoint = EXP / "best_model.pt"

    for p in [encoder_dir, tokenizer_dir, checkpoint]:
        if not p.exists():
            raise FileNotFoundError(f"Required BERT asset missing: {p}")

    print("Device:", DEVICE)
    print("Loading independent BERT checkpoint...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    encoder = AutoModel.from_pretrained(str(encoder_dir))
    model = IndependentBERT(
        encoder, [len(maps[t]) for t in TARGETS]
    )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    loader = DataLoader(
        TextDataset(val_df["text"].astype(str).tolist(), tokenizer),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    inverse_maps = {
        t: {int(i): label for label, i in maps[t].items()}
        for t in TARGETS
    }
    predictions = {t: [] for t in TARGETS}

    print("Running BERT predictions...", flush=True)
    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            outputs = model(ids, mask)
            for i, target in enumerate(TARGETS):
                indices = outputs[i].argmax(dim=1).cpu().tolist()
                predictions[target].extend(
                    inverse_maps[target][j] for j in indices
                )

    del model, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Running saved SVM predictions...", flush=True)
    vectorizer = joblib.load(SVM_DIR / "tfidf_vectorizer.joblib")
    X_val = vectorizer.transform(val_df["text"].astype(str))

    for target in ["Issue", "Sub-issue"]:
        suffix = target.replace("-", "_")
        svm = joblib.load(SVM_DIR / f"linear_svc_{suffix}.joblib")
        predictions[target] = svm.predict(X_val).tolist()

    report = {}
    for target in TARGETS:
        truth = val_df[target].astype(str).tolist()
        pred = predictions[target]

        # Include labels unseen in training as zero-score classes.
        labels = sorted(set(maps[target].keys()) | set(truth))
        report[target] = {
            "accuracy": round(accuracy_score(truth, pred), 4),
            "macro_f1": round(
                f1_score(truth, pred, labels=labels,
                         average="macro", zero_division=0), 4
            ),
        }

    correct_all = np.ones(len(val_df), dtype=bool)
    for target in TARGETS:
        correct_all &= (
            val_df[target].astype(str).to_numpy()
            == np.asarray(predictions[target], dtype=str)
        )

    # A prediction is hierarchy-consistent if its full path occurred
    # at least once in the training data.
    valid_paths = set(
        map(tuple, train_df[TARGETS].astype(str).to_numpy().tolist())
    )
    predicted_paths = list(zip(*(predictions[t] for t in TARGETS)))
    consistent = sum(tuple(path) in valid_paths for path in predicted_paths)

    report["overall"] = {
        "exact_matches": int(correct_all.sum()),
        "rows": len(val_df),
        "exact_match_accuracy": round(float(correct_all.mean()), 4),
        "hierarchy_consistent_rows": int(consistent),
        "hierarchy_consistency": round(consistent / len(val_df), 4),
        "bert_targets": BERT_TARGETS,
        "svm_targets": ["Issue", "Sub-issue"],
    }

    output_file = OUT / "hybrid_metrics.json"
    output_file.write_text(json.dumps(report, indent=2))

    print("\n===== HYBRID RESULTS =====")
    for target in TARGETS:
        print(
            f"{target:12s} accuracy={report[target]['accuracy']:.4f} "
            f"macro_f1={report[target]['macro_f1']:.4f}"
        )
    print("Exact-match accuracy:",
          report["overall"]["exact_match_accuracy"])
    print("Hierarchy consistency:",
          report["overall"]["hierarchy_consistency"])
    print("Saved:", output_file)

if __name__ == "__main__":
    main()
