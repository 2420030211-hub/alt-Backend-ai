import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path.cwd()
MODEL_DIR = ROOT / "ai-services/cora/outputs/experiments/bert_weighted_30epochs"
DATA = ROOT / "ai-services/cora/data/cora_validation.csv"
OUT = MODEL_DIR / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "ai-services/cora/scripts"))
from train_bert_gpu import HierarchicalBERT, TARGETS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(DATA).dropna(subset=["text"]).reset_index(drop=True)
mappings = json.loads((MODEL_DIR / "label_mappings.json").read_text())

model = HierarchicalBERT(
    str(MODEL_DIR / "encoder"),
    [len(mappings[t]) for t in TARGETS],
)
state = torch.load(
    MODEL_DIR / "best_model.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state, strict=True)
model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
inverse = {
    t: {int(i): label for label, i in mappings[t].items()}
    for t in TARGETS
}
truth = {t: [] for t in TARGETS}
predictions = {t: [] for t in TARGETS}

print(f"Evaluating {len(df)} validation rows on {device}...")

with torch.inference_mode():
    for start in range(0, len(df), 16):
        batch = df.iloc[start:start + 16]
        tokens = tokenizer(
            batch["text"].fillna("").astype(str).tolist(),
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
            pred_ids = logits.argmax(dim=1).cpu().tolist()
            for j, pred_id in enumerate(pred_ids):
                actual = batch.iloc[j][target]
                if pd.isna(actual):
                    continue
                actual = str(actual)
                if actual not in mappings[target]:
                    continue
                truth[target].append(actual)
                predictions[target].append(
                    inverse[target].get(pred_id, "UNKNOWN")
                )

summary = {}
for target in TARGETS:
    labels = sorted(mappings[target].values())
    actual = truth[target]
    predicted = predictions[target]

    # Include every class from the saved training mappings.
    all_ids = sorted(inverse[target])
    all_names = [inverse[target][i] for i in all_ids]
    name_to_id = mappings[target]

    y_true = [name_to_id[x] for x in actual]
    y_pred = [name_to_id.get(x, -1) for x in predicted]

    macro = f1_score(
        y_true, y_pred,
        labels=all_ids,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(actual, predicted)

    report = classification_report(
        y_true, y_pred,
        labels=all_ids,
        target_names=all_names,
        output_dict=True,
        zero_division=0,
    )

    # Classes with the lowest F1 and their validation support.
    class_rows = []
    for name in all_names:
        row = report[name]
        class_rows.append({
            "class": name,
            "support": int(row["support"]),
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1": round(row["f1-score"], 4),
        })

    rare = sorted(
        [r for r in class_rows if r["support"] > 0],
        key=lambda r: (r["f1"], r["support"]),
    )[:15]

    # Most frequent wrong actual -> predicted pairs.
    errors = {}
    for a, p in zip(actual, predicted):
        if a != p:
            errors[(a, p)] = errors.get((a, p), 0) + 1
    common_errors = [
        {"actual": a, "predicted": p, "count": count}
        for (a, p), count in sorted(
            errors.items(), key=lambda item: item[1], reverse=True
        )[:15]
    ]

    summary[target] = {
        "rows": len(actual),
        "accuracy": round(accuracy, 4),
        "macro_f1_all_known_classes": round(macro, 4),
        "classes": len(all_ids),
        "worst_classes_with_validation_support": rare,
        "most_common_errors": common_errors,
    }

    print(f"\n===== {target} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 (all known classes): {macro:.4f}")
    print("Lowest-performing classes:")
    for row in rare[:10]:
        print(
            f"  {row['class']} | support={row['support']} "
            f"| F1={row['f1']:.3f} | recall={row['recall']:.3f}"
        )
    print("Most common mistakes:")
    for row in common_errors[:10]:
        print(
            f"  Actual: {row['actual']} -> Predicted: {row['predicted']} "
            f"({row['count']} rows)"
        )

out_file = OUT / "error_analysis.json"
out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nFull error report saved to:", out_file)
