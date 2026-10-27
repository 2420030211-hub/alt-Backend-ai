import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

root = Path.cwd()
model_dir = root / "ai-services/cora/outputs/experiments/bert_base_gpu"
data_path = root / "ai-services/cora/data/cora_validation.csv"
out_dir = model_dir / "evaluation"
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(root / "ai-services/cora/scripts"))
from train_bert_gpu import HierarchicalBERT, TARGETS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maps = json.loads((model_dir / "label_mappings.json").read_text())
df = pd.read_csv(data_path).dropna(subset=["text"]).reset_index(drop=True)

model = HierarchicalBERT(
    str(model_dir / "encoder"),
    [len(maps[t]) for t in TARGETS],
)
model.load_state_dict(torch.load(
    model_dir / "best_model.pt",
    map_location="cpu",
    weights_only=True,
))
model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
inverse = {
    t: {int(i): label for label, i in maps[t].items()}
    for t in TARGETS
}
truth = {t: [] for t in TARGETS}
preds = {t: [] for t in TARGETS}

with torch.inference_mode():
    for start in range(0, len(df), 16):
        batch = df.iloc[start:start + 16]
        encoded = tokenizer(
            batch["text"].fillna("").astype(str).tolist(),
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        for target, logits in zip(TARGETS, outputs):
            ids = logits.argmax(dim=1).cpu().tolist()
            for j, idx in enumerate(ids):
                actual = batch.iloc[j][target]
                if pd.isna(actual):
                    continue
                actual = str(actual)
                if actual not in maps[target]:
                    continue
                truth[target].append(actual)
                preds[target].append(inverse[target].get(idx, "UNKNOWN"))

reports = {}
for target in TARGETS:
    labels = sorted(set(truth[target]) | set(preds[target]))
    rep = classification_report(
        truth[target],
        preds[target],
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(truth[target], preds[target], labels=labels)

    per_class = []
    for label in labels:
        row = rep.get(label, {})
        per_class.append({
            "label": label,
            "precision": round(row.get("precision", 0), 4),
            "recall": round(row.get("recall", 0), 4),
            "f1": round(row.get("f1-score", 0), 4),
            "support": int(row.get("support", 0)),
        })

    per_class.sort(key=lambda x: x["f1"])
    reports[target] = {
        "evaluated_rows": len(truth[target]),
        "lowest_f1_classes": per_class[:15],
        "classification_report": rep,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n===== {target}: LOWEST F1 CLASSES =====")
    for item in per_class[:10]:
        print(
            f"{item['label']}: F1={item['f1']:.3f}, "
            f"P={item['precision']:.3f}, R={item['recall']:.3f}, "
            f"support={item['support']}"
        )

path = out_dir / "detailed_class_report.json"
path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
print("\nFull detailed report saved to:", path)
