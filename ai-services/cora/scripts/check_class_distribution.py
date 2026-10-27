import sys
import os
import pandas as pd
from collections import Counter

def check_distribution(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n===== CLASS DISTRIBUTION FOR {csv_path} =====")
    targets = ["Product", "Sub-product", "Issue", "Sub-issue"]

    for col in targets:
        if col in df.columns:
            print(f"\n--- Distribution for {col} (Top 10) ---")
            counts = df[col].value_counts()
            for label, count in counts.head(10).items():
                print(f"  {label:<45}: {count:>5} ({count/len(df)*100:>5.2f}%)")
            print(f"Total Unique Classes for {col}: {len(counts)}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/train_cleaned.csv"
    check_distribution(csv_file)
