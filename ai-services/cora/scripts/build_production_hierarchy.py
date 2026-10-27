import json
from pathlib import Path
import pandas as pd

BASE = Path("ai-services/cora")
DATA = BASE / "data/cora_train.csv"
OUT = Path("ai-services/cora/outputs/production_hierarchy.json")

TARGETS = ["Product", "Sub-product", "Issue", "Sub-issue"]
df = pd.read_csv(DATA).fillna("")
df[TARGETS] = df[TARGETS].astype(str)

# Store all observed parent-child relationships and complete paths.
hierarchy = {
    "version": 1,
    "source": "cora_train.csv",
    "row_count": len(df),
    "labels": {},
    "product_to_subproducts": {},
    "product_subproduct_to_issues": {},
    "product_subproduct_issue_to_subissues": {},
    "valid_full_paths": [],
}

for target in TARGETS:
    hierarchy["labels"][target] = sorted(df[target].unique().tolist())

for product, sub in df[["Product", "Sub-product"]].drop_duplicates().itertuples(index=False, name=None):
    hierarchy["product_to_subproducts"].setdefault(product, []).append(sub)

for product, sub, issue in df[["Product", "Sub-product", "Issue"]].drop_duplicates().itertuples(index=False, name=None):
    key = json.dumps([product, sub], ensure_ascii=False)
    hierarchy["product_subproduct_to_issues"].setdefault(key, []).append(issue)

for product, sub, issue, subissue in df[TARGETS].drop_duplicates().itertuples(index=False, name=None):
    key = json.dumps([product, sub, issue], ensure_ascii=False)
    hierarchy["product_subproduct_issue_to_subissues"].setdefault(key, []).append(subissue)
    hierarchy["valid_full_paths"].append([product, sub, issue, subissue])

for mapping in [
    hierarchy["product_to_subproducts"],
    hierarchy["product_subproduct_to_issues"],
    hierarchy["product_subproduct_issue_to_subissues"],
]:
    for key in mapping:
        mapping[key] = sorted(set(mapping[key]))

hierarchy["valid_full_paths"].sort()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(hierarchy, indent=2, ensure_ascii=False))

# Reload and verify the artifact against every training row.
saved = json.loads(OUT.read_text())
valid = {tuple(path) for path in saved["valid_full_paths"]}
bad_rows = [
    i for i, row in enumerate(df[TARGETS].itertuples(index=False, name=None))
    if tuple(row) not in valid
]

print("Saved:", OUT)
print("Training rows:", len(df))
print("Unique full paths:", len(valid))
print("Invalid training paths:", len(bad_rows))
print("Labels:", {k: len(v) for k, v in saved["labels"].items()})

if bad_rows:
    raise SystemExit("Hierarchy verification failed.")
print("Hierarchy artifact verification passed.")
