import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path.cwd()
DATA = ROOT / "ai-services/cora/data/cora_validation.csv"
BASE = ROOT / "ai-services/cora/outputs/experiments"

sys.path.insert(0, str(ROOT / "ai-services/cora/scripts"))
from train_bert_gpu import HierarchicalBERT, TARGETS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(DATA).dropna(subset=["text"]).reset_index(drop=True)

def evaluate(name, folder):
    model_dir = BASE / folder
    with open(model_dir / "label_mappings.json", encoding="utf-8") as f:
        mappings = json.load(f)

    model = HierarchicalBERT(
        str(model_dir / "encoder"),
        [len(mappings[t]) for t in TARGETS],
    )
    state = torch.load(
        model_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    inverse = {
        t: {int(i): label for label, i in mappings[t].items()}
        for t in TARGETS
    }
    truth = {t: [] for t in TARGETS}
    predictions = {t: [] for t in TARGETS}

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
                ids = logits.argmax(dim=1).cpu().tolist()
                for j, pred_id in enumerate(ids):
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

    metrics = {}
    for target in TARGETS:
        metrics[target] = {
            "rows": len(truth[target]),
            "accuracy": round(
                accuracy_score(truth[target], predictions[target]), 4
            ),
            "macro_f1": round(
                f1_score(
                    truth[target], predictions[target],
                    average="macro", zero_division=0
                ), 4
            ),
        }

    metrics["mean_macro_f1"] = round(
        sum(metrics[t]["macro_f1"] for t in TARGETS) / len(TARGETS), 4
    )
    print(f"\n===== {name} =====")
    for target in TARGETS:
        m = metrics[target]
        print(
            f"{target}: accuracy={m['accuracy']:.4f}, "
            f"macro_F1={m['macro_f1']:.4f}, rows={m['rows']}"
        )
    print("Mean macro F1:", metrics["mean_macro_f1"])
    return metrics

results = {
    "original": evaluate("ORIGINAL MODEL", "bert_base_gpu"),
    "weighted_3epochs": evaluate("WEIGHTED MODEL (3-EPOCH)", "bert_weighted_v1"),
    "weighted_30epochs": evaluate("WEIGHTED MODEL (30-EPOCH RUN)", "bert_weighted_30epochs"),
}

out = BASE / "checkpoint_comparison.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nComparison saved to:", out)
