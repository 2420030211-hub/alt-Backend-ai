import pandas as pd
from pathlib import Path

base = Path("ai-services/cora/data")
train = pd.read_csv(base / "cora_train.csv").fillna("")
val = pd.read_csv(base / "cora_validation.csv").fillna("")

pairs = [
    ("Product", "Sub-product"),
    ("Sub-product", "Issue"),
    ("Issue", "Sub-issue"),
]

for parent, child in pairs:
    allowed = train.groupby(child)[parent].agg(
        lambda s: set(s.astype(str))
    ).to_dict()

    known = val[val[child].astype(str).isin(allowed)]
    violations = 0

    for _, row in known.iterrows():
        parents = allowed[str(row[child])]
        if str(row[parent]) not in parents:
            violations += 1

    print(
        f"{parent} -> {child}: "
        f"{violations}/{len(known)} validation rows "
        f"({violations / max(len(known), 1):.2%}) "
        "have a parent-child combination unseen in training"
    )
