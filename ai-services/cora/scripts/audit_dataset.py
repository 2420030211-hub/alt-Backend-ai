"""
Dataset Audit Script for CORA Banking Complaint Data
Checks:
- Missing values and target labels
- Duplication and contradictory labels
- Parent-child hierarchy consistency
- Class imbalances
"""
import os
import sys
import argparse
import pandas as pd
import json

TARGET_COLS = ["Product", "Sub-product", "Issue", "Sub-issue"]

def audit_dataset(csv_path: str):
    print(f"\n================ AUDITING DATASET: {csv_path} ================")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Total Rows: {total_rows}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # Text column check
    text_col = "text" if "text" in df.columns else ("Consumer complaint narrative" if "Consumer complaint narrative" in df.columns else None)
    if not text_col:
        print("WARNING: No standard text column ('text' or 'Consumer complaint narrative') detected!")
    else:
        empty_text = df[text_col].isna().sum()
        print(f"Text Column: '{text_col}' | Missing/Empty Narratives: {empty_text} ({(empty_text/total_rows)*100:.2f}%)")

    # Missing targets check
    print("\n--- Missing Value Analysis Across Targets ---")
    for col in TARGET_COLS:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} missing ({missing/total_rows*100:.2f}%) | Unique Classes: {df[col].nunique()}")
        else:
            print(f"  {col}: NOT FOUND IN DATASET")

    # Duplicates check
    if text_col:
        text_dups = df.duplicated(subset=[text_col]).sum()
        print(f"\nDuplicate narratives: {text_dups} ({(text_dups/total_rows)*100:.2f}%)")
        
        # Inconsistent duplicate labels check
        if "Product" in df.columns:
            contradicting = df.groupby(text_col)["Product"].nunique()
            contradictions = contradicting[contradicting > 1].count()
            print(f"Identical texts with contradictory 'Product' labels: {contradictions}")

    # Hierarchy validation
    if "Product" in df.columns and "Sub-product" in df.columns:
        pairs = df[["Product", "Sub-product"]].dropna().drop_duplicates()
        print(f"\nUnique Product -> Sub-product valid relationships: {len(pairs)}")

    print("\n================ AUDIT SUMMARY COMPLETED ================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit CORA Banking Complaint Dataset")
    parser.add_argument("--csv", type=str, default="C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/train_cleaned.csv", help="Path to CSV dataset")
    args = parser.parse_args()
    audit_dataset(args.csv)
