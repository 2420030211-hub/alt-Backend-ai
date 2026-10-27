import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer

ROOT = Path("ai-services/cora")
EXPERIMENT = ROOT / "outputs/experiments/bert_targetweighted_v1"
BASE_CHECKPOINT = ROOT / "outputs/experiments/bert_weighted_30epochs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bert_gpu import TARGETS, ComplaintDataset, HierarchicalBERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv(ROOT / "data/cora_train.csv")
val_df = pd.read_csv(ROOT / "data/cora_validation.csv")

label_maps = json.loads((EXPERIMENT / "label_mappings.json").read_text())
if label_maps != json.loads((BASE_CHECKPOINT / "label_mappings.json").read_text()):
    raise RuntimeError("Label mappings differ; stopping to prevent incorrect evaluation.")

tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT / "tokenizer")
dataset = ComplaintDataset(val_df, tokenizer, label_maps, 256)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

model = HierarchicalBERT(
    str(BASE_CHECKPOINT / "encoder"),
    [len(label_maps[t]) for t in TARGETS],
).to(device)

state = torch.load(EXPERIMENT / "best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

all_true = [[] for _ in TARGETS]
all_pred = [[] for _ in TARGETS]

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids, attention_mask)
        for i, target in enumerate(TARGETS):
            pred = outputs[i].argmax(dim=1).cpu()
            actual = labels[:, i]
            valid = actual != -100
            all_true[i].extend(actual[valid].tolist())
            all_pred[i].extend(pred[valid].tolist())

out_dir = EXPERIMENT / "diagnostics"
out_dir.mkdir(parents=True, exist_ok=True)

for i, target in enumerate(TARGETS):
    names = [None] * len(label_maps[target])
    for label, idx in label_maps[target].items():
        names[idx] = label

    report = classification_report(
        all_true[i],
        all_pred[i],
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label in names:
        item = report[label]
        rows.append({
            "class": label,
            "support": int(item["support"]),
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1-score"],
        })

    class_df = pd.DataFrame(rows).sort_values(
        ["f1", "support"], ascending=[True, False]
    )
    class_df.to_csv(out_dir / f"{target.replace('/', '_')}_per_class.csv", index=False)

    cm = confusion_matrix(
        all_true[i], all_pred[i], labels=list(range(len(names)))
    )
    mistakes = []
    for actual_id in range(len(names)):
        for predicted_id in range(len(names)):
            if actual_id != predicted_id and cm[actual_id, predicted_id] > 0:
                mistakes.append({
                    "actual": names[actual_id],
                    "predicted": names[predicted_id],
                    "count": int(cm[actual_id, predicted_id]),
                })

    pd.DataFrame(mistakes, columns=["actual", "predicted", "count"]).sort_values(
        "count", ascending=False
    ).to_csv(out_dir / f"{target.replace('/', '_')}_confusions.csv", index=False)

    print(f"\n=== {target}: weakest classes with at least 5 validation examples ===")
    eligible = class_df[class_df["support"] >= 5]
    print(eligible.head(15).to_string(index=False))

    print(f"\n=== {target}: most frequent confusions ===")
    if mistakes:
        print(pd.DataFrame(mistakes).sort_values(
            "count", ascending=False
        ).head(10).to_string(index=False))
    else:
        print("No misclassifications.")

print(f"\nDiagnostic CSV files saved to: {out_dir}")
print("No training performed. Existing checkpoints were only loaded.")
