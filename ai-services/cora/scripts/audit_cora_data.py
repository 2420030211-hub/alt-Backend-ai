"""
Comprehensive data auditor for CORA banking complaint datasets.
Audits:
- Schema, row counts, missing values, empty strings
- Exact and normalized text duplicates and conflicting hierarchical labels
- Investigates previously reported 13 duplicate text finding
- Taxonomical consistency (Product -> Sub-product -> Issue -> Sub-issue)
- Output saved as structured JSON and Markdown summary
"""
import os
import sys
import json
import re
import unicodedata
import pandas as pd
import numpy as np

TARGET_COLS = ["Product", "Sub-product", "Issue", "Sub-issue"]

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Trim and collapse whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def audit_file(csv_path: str):
    print(f"Auditing file: {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    # 1. Column and missingness audit
    cols = list(df.columns)
    text_col = "text" if "text" in cols else ("Consumer complaint narrative" if "Consumer complaint narrative" in cols else None)
    
    missing_summary = {}
    for c in cols:
        na_count = int(df[c].isna().sum())
        blank_count = int((df[c].astype(str).str.strip() == "").sum()) if df[c].dtype == object else 0
        missing_summary[c] = {
            "na_count": na_count,
            "blank_count": blank_count,
            "pct_missing": round((na_count + blank_count) / total_rows * 100, 3)
        }

    # Text length stats
    text_stats = {}
    if text_col and df[text_col].dropna().count() > 0:
        lengths = df[text_col].dropna().astype(str).apply(len)
        words = df[text_col].dropna().astype(str).apply(lambda s: len(s.split()))
        text_stats = {
            "char_length_mean": round(float(lengths.mean()), 1),
            "char_length_median": float(lengths.median()),
            "char_length_min": int(lengths.min()),
            "char_length_max": int(lengths.max()),
            "word_count_mean": round(float(words.mean()), 1),
            "word_count_median": float(words.median()),
            "word_count_min": int(words.min()),
            "word_count_max": int(words.max()),
        }

    # 2. Exact Duplicates & Groupings
    dup_groups_exact = []
    conflict_summary = {
        "identical_text_identical_labels": 0,
        "conflicting_product": 0,
        "conflicting_subproduct": 0,
        "conflicting_issue": 0,
        "conflicting_subissue": 0,
        "multi_level_conflicts": 0
    }
    
    if text_col:
        # Exact duplicate analysis
        grouped = df.groupby(text_col)
        for narrative, group in grouped:
            if len(group) > 1:
                p_diff = group["Product"].nunique() if "Product" in group else 1
                sp_diff = group["Sub-product"].nunique() if "Sub-product" in group else 1
                iss_diff = group["Issue"].nunique() if "Issue" in group else 1
                subiss_diff = group["Sub-issue"].nunique() if "Sub-issue" in group else 1
                
                is_p_conflict = p_diff > 1
                is_sp_conflict = sp_diff > 1
                is_iss_conflict = iss_diff > 1
                is_subiss_conflict = subiss_diff > 1
                
                conflict_count = sum([is_p_conflict, is_sp_conflict, is_iss_conflict, is_subiss_conflict])
                
                if conflict_count == 0:
                    conflict_summary["identical_text_identical_labels"] += 1
                else:
                    if is_p_conflict: conflict_summary["conflicting_product"] += 1
                    if is_sp_conflict: conflict_summary["conflicting_subproduct"] += 1
                    if is_iss_conflict: conflict_summary["conflicting_issue"] += 1
                    if is_subiss_conflict: conflict_summary["conflicting_subissue"] += 1
                    if conflict_count > 1: conflict_summary["multi_level_conflicts"] += 1
                
                dup_groups_exact.append({
                    "group_size": len(group),
                    "row_indices": list(group.index),
                    "text_snippet": str(narrative)[:120],
                    "unique_products": int(p_diff),
                    "unique_subproducts": int(sp_diff),
                    "unique_issues": int(iss_diff),
                    "unique_subissues": int(subiss_diff),
                    "labels": group[TARGET_COLS].to_dict(orient="records") if all(t in group for t in TARGET_COLS) else []
                })

    # Normalized duplicates
    dup_groups_norm_count = 0
    if text_col:
        norm_series = df[text_col].dropna().astype(str).apply(normalize_text)
        dup_groups_norm_count = int(norm_series.duplicated().sum())

    # 3. Class distributions for all targets
    class_distributions = {}
    for target in TARGET_COLS:
        if target in df.columns:
            val_counts = df[target].value_counts(dropna=False)
            class_distributions[target] = {
                "unique_count": int(df[target].nunique()),
                "most_frequent": str(val_counts.index[0]),
                "most_frequent_count": int(val_counts.iloc[0]),
                "least_frequent": str(val_counts.index[-1]),
                "least_frequent_count": int(val_counts.iloc[-1]),
                "imbalance_ratio": round(float(val_counts.iloc[0] / max(1, val_counts.iloc[-1])), 2),
                "singletons_count": int((val_counts == 1).sum()),
                "classes": {str(k): int(v) for k, v in val_counts.items()}
            }

    # 4. Hierarchical path consistency
    hierarchy_metrics = {}
    if all(t in df.columns for t in TARGET_COLS):
        p_to_sp = df.groupby("Product")["Sub-product"].nunique().to_dict()
        sp_to_p = df.groupby("Sub-product")["Product"].nunique()
        multi_parent_sp = sp_to_p[sp_to_p > 1].to_dict()

        iss_to_sp = df.groupby("Issue")["Sub-product"].nunique()
        multi_parent_iss = iss_to_sp[iss_to_sp > 1].to_dict()

        subiss_to_iss = df.groupby("Sub-issue")["Issue"].nunique()
        multi_parent_subiss = subiss_to_iss[subiss_to_iss > 1].to_dict()

        unique_paths = len(df[TARGET_COLS].drop_duplicates())

        hierarchy_metrics = {
            "total_unique_full_paths": unique_paths,
            "subproducts_with_multiple_products": multi_parent_sp,
            "issues_with_multiple_subproducts_count": len(multi_parent_iss),
            "subissues_with_multiple_issues_count": len(multi_parent_subiss),
        }

    report = {
        "file_path": os.path.abspath(csv_path),
        "total_rows": total_rows,
        "total_columns": len(cols),
        "columns": cols,
        "missing_summary": missing_summary,
        "text_statistics": text_stats,
        "exact_duplicates": {
            "pandas_duplicated_count": int(df[text_col].duplicated().sum()) if text_col else 0,
            "total_duplicate_groups": len(dup_groups_exact),
            "total_rows_in_duplicate_groups": sum(g["group_size"] for g in dup_groups_exact),
            "conflict_categorization": conflict_summary,
            "groups_detail": dup_groups_exact
        },
        "normalized_duplicates": {
            "normalized_duplicated_count": dup_groups_norm_count
        },
        "class_distributions": class_distributions,
        "hierarchy_metrics": hierarchy_metrics
    }
    return report

def main():
    csv_to_audit = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/M S I/Downloads/complaints_cleaned.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "ai-services/cora/outputs/audit"
    os.makedirs(out_dir, exist_ok=True)
    
    report = audit_file(csv_to_audit)
    
    fname = os.path.splitext(os.path.basename(csv_to_audit))[0]
    out_json = os.path.join(out_dir, f"{fname}_audit.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Audit completed and saved to {out_json}")

if __name__ == "__main__":
    main()
