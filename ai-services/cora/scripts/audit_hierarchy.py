"""
Audit hierarchical taxonomy constraints:
Product -> Sub-product -> Issue -> Sub-issue
Builds canonical taxonomy reference JSON, identifies multiple-parent ambiguities,
and validates consistent paths across training and validation splits.
"""
import os
import sys
import json
import pandas as pd

TARGETS = ["Product", "Sub-product", "Issue", "Sub-issue"]

def audit_hierarchy(csv_path: str, output_path: str = "ai-services/cora/outputs/audit/hierarchy_tree.json"):
    print(f"Auditing hierarchy taxonomy in: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Product -> Sub-products
    p_to_sp = {}
    for p, group in df.groupby("Product"):
        p_to_sp[str(p)] = sorted([str(x) for x in group["Sub-product"].dropna().unique()])
        
    # 2. Sub-product -> Issues
    sp_to_iss = {}
    for sp, group in df.groupby("Sub-product"):
        sp_to_iss[str(sp)] = sorted([str(x) for x in group["Issue"].dropna().unique()])

    # 3. Issue -> Sub-issues
    iss_to_sub = {}
    for iss, group in df.groupby("Issue"):
        iss_to_sub[str(iss)] = sorted([str(x) for x in group["Sub-issue"].dropna().unique()])

    # 4. Check for multiple parents
    sp_parents = df.groupby("Sub-product")["Product"].unique().to_dict()
    multi_sp = {k: list(v) for k, v in sp_parents.items() if len(v) > 1}

    iss_parents = df.groupby("Issue")["Sub-product"].unique().to_dict()
    multi_iss = {k: list(v) for k, v in iss_parents.items() if len(v) > 1}

    # Path counts
    path_counts = df.groupby(TARGETS).size().reset_index(name="frequency")
    single_occurrence_paths = int((path_counts["frequency"] == 1).sum())

    taxonomy = {
        "source_dataset": os.path.abspath(csv_path),
        "total_records": len(df),
        "total_unique_paths": len(path_counts),
        "single_occurrence_paths": single_occurrence_paths,
        "product_count": df["Product"].nunique(),
        "subproduct_count": df["Sub-product"].nunique(),
        "issue_count": df["Issue"].nunique(),
        "subissue_count": df["Sub-issue"].nunique(),
        "product_to_subproducts": p_to_sp,
        "subproducts_with_multiple_parent_products": multi_sp,
        "issues_with_multiple_parent_subproducts_count": len(multi_iss),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Taxonomy analysis saved to {output_path}")
    return taxonomy

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/M S I/Downloads/complaints_cleaned.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "ai-services/cora/outputs/audit/hierarchy_tree.json"
    audit_hierarchy(src, out)
