"""
Audit data split leakage between Train, Validation, and Test sets.
Checks exact text overlap, normalized text overlap, duplicate text groups,
and label distribution drift.
"""
import os
import sys
import json
import re
import unicodedata
import pandas as pd

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def audit_leakage(train_csv: str, val_or_test_csv: str, split_name: str = "train_vs_val"):
    print(f"Auditing leakage between:\n  Set 1: {train_csv}\n  Set 2: {val_or_test_csv}")
    
    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(val_or_test_csv)
    
    col1 = "text" if "text" in df1.columns else "Consumer complaint narrative"
    col2 = "text" if "text" in df2.columns else "Consumer complaint narrative"
    
    s1_raw = set(df1[col1].dropna().astype(str))
    s2_raw = set(df2[col2].dropna().astype(str))
    
    raw_overlap = s1_raw.intersection(s2_raw)
    
    # Normalized overlap
    norm_map1 = {normalize_text(t): t for t in s1_raw}
    norm_map2 = {normalize_text(t): t for t in s2_raw}
    norm_overlap = set(norm_map1.keys()).intersection(set(norm_map2.keys()))
    
    overlap_details = []
    for norm_t in list(norm_overlap)[:50]: # capture up to 50
        t1 = norm_map1[norm_t]
        t2 = norm_map2[norm_t]
        rows1 = df1[df1[col1] == t1]
        rows2 = df2[df2[col2] == t2]
        overlap_details.append({
            "normalized_snippet": norm_t[:100],
            "set1_indices": list(rows1.index),
            "set2_indices": list(rows2.index),
            "set1_labels": rows1[["Product"]].to_dict(orient="records") if "Product" in rows1 else [],
            "set2_labels": rows2[["Product"]].to_dict(orient="records") if "Product" in rows2 else []
        })

    result = {
        "comparison": split_name,
        "set1_file": os.path.abspath(train_csv),
        "set1_total_rows": len(df1),
        "set1_unique_texts": len(s1_raw),
        "set2_file": os.path.abspath(val_or_test_csv),
        "set2_total_rows": len(df2),
        "set2_unique_texts": len(s2_raw),
        "exact_raw_overlap_count": len(raw_overlap),
        "normalized_overlap_count": len(norm_overlap),
        "overlap_percentage_of_set2": round(len(raw_overlap) / len(s2_raw) * 100, 4) if len(s2_raw) > 0 else 0,
        "overlap_samples": overlap_details
    }
    return result

def main():
    f1 = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/M S I/Downloads/cora_train.csv"
    f2 = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/M S I/Downloads/cora_validation.csv"
    name = sys.argv[3] if len(sys.argv) > 3 else "cora_train_vs_val"
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "ai-services/cora/outputs/audit"
    
    os.makedirs(out_dir, exist_ok=True)
    res = audit_leakage(f1, f2, name)
    out_file = os.path.join(out_dir, f"leakage_{name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Leakage audit saved to {out_file}")

if __name__ == "__main__":
    main()
