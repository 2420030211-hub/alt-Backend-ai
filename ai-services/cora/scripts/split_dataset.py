import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def make_split(csv_path: str, output_dir: str):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")

    # Remove rows where text or primary Product is null
    text_col = "text" if "text" in df.columns else "Consumer complaint narrative"
    df = df.dropna(subset=[text_col, "Product"])

    # Split train (70%), val (15%), test (15% held out)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["Product"] if df["Product"].nunique() > 1 else None)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "cora_train.csv")
    val_path = os.path.join(output_dir, "cora_validation.csv")
    test_path = os.path.join(output_dir, "cora_test_heldout.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Split completed successfully:")
    print(f"  Train ({len(train_df)} rows) -> {train_path}")
    print(f"  Validation ({len(val_df)} rows) -> {val_path}")
    print(f"  Held-out Test ({len(test_df)} rows) -> {test_path}")

if __name__ == "__main__":
    src_csv = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/train_cleaned.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/M S I/Downloads/banking-complaint-backend/datasets/cora"
    make_split(src_csv, out)
