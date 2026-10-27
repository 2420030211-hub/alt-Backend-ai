import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

root = Path.cwd()
model_dir = root / "ai-services/cora/outputs/experiments/bert_15epochs_v1"
data_path = root / "ai-services/cora/data/cora_validation.csv"
report_dir = model_dir / "evaluation"
report_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(root / "ai-services/cora/scripts"))
from train_bert_gpu import HierarchicalBERT, TARGETS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(model_dir / "label_mappings.json", encoding="utf-8") as f:
    maps = json.load(f)

df = pd.read_csv(data_path)
df = df.dropna(subset=["text"]).reset_index(drop=True)

model = HierarchicalBERT(
    str(model_dir / "encoder"),
    [len(maps[t]) for t in TARGETS],
)
state = torch.load(
    model_dir / "best_model.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
inverse = {
    t: {int(i): label for label, i in maps[t].items()}
    for t in TARGETS
}

y_true = {t: [] for t in TARGETS}
y_pred = {t: [] for t in TARGETS}

print(f"Evaluating {len(df)} validation rows on {device}...")

with torch.inference_mode():
    for start in range(0, len(df), 16):
        batch = df.iloc[start:start + 16]
        texts = batch["text"].fillna("").astype(str).tolist()

        tokens = tokenizer(
            texts,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )

        for target, logits in zip(TARGETS, outputs):
            ids = logits.argmax(dim=1).cpu().tolist()
            for j, pred_id in enumerate(ids):
                actual = batch.iloc[j][target]
                if pd.isna(actual):
                    continue
                actual = str(actual)
                if actual not in maps[target]:
                    continue
                y_true[target].append(actual)
                y_pred[target].append(inverse[target].get(pred_id, "UNKNOWN"))

results = {}
print("\n===== VALIDATION RESULTS =====")

for target in TARGETS:
    a, p = y_true[target], y_pred[target]
    metrics = {
        "evaluated_rows": len(a),
        "accuracy": round(accuracy_score(a, p), 4),
        "macro_f1": round(f1_score(a, p, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(a, p, average="weighted", zero_division=0), 4),
    }
    results[target] = metrics
    print(
        f"{target}: accuracy={metrics['accuracy']:.4f}, "
        f"macro_F1={metrics['macro_f1']:.4f}, "
        f"weighted_F1={metrics['weighted_f1']:.4f}, "
        f"rows={metrics['evaluated_rows']}"
    )

report = {
    "validation_rows": len(df),
    "device": str(device),
    "mean_target_accuracy": round(
        sum(results[t]["accuracy"] for t in TARGETS) / len(TARGETS), 4
    ),
    "mean_target_macro_f1": round(
        sum(results[t]["macro_f1"] for t in TARGETS) / len(TARGETS), 4
    ),
    "targets": results,
    "note": "Validation labels absent from training are excluded for that target. Mean target metrics are averages across four targets, not exact-match accuracy.",
}

out = report_dir / "validation_metrics.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nMean target accuracy:", report["mean_target_accuracy"])
print("Mean target macro F1:", report["mean_target_macro_f1"])
print("Saved report:", out)
