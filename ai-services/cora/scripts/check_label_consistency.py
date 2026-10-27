import json
from pathlib import Path
import pandas as pd

base = Path("ai-services/cora")
out = base / "outputs/experiments/bert_base_gpu"
mappings = json.loads((out / "label_mappings.json").read_text())

train = pd.read_csv(base / "data/cora_train.csv")
val = pd.read_csv(base / "data/cora_validation.csv")

targets = ["Product", "Sub-product", "Issue", "Sub-issue"]

for target in targets:
    print(f"\n{'=' * 60}\n{target}")
    mapping = mappings[target]
    mapped_labels = set(mapping.keys())

    print("Mapping labels:", len(mapped_labels))
    print("Train unique labels:", train[target].nunique())
    print("Validation unique labels:", val[target].nunique())

    train_labels = set(train[target].dropna().astype(str).unique())
    val_labels = set(val[target].dropna().astype(str).unique())

    print("Train labels missing from mapping:",
          sorted(train_labels - mapped_labels)[:20])
    print("Validation labels missing from mapping:",
          sorted(val_labels - mapped_labels)[:20])

    ids = list(mapping.values())
    print("IDs unique:", len(ids) == len(set(ids)))
    print("IDs contiguous:",
          sorted(ids) == list(range(len(ids))))

print("\nCheck complete.")
