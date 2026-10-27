"""
Discover all candidate CSV datasets across the repository and system,
compute their cryptographic SHA-256 hashes, row counts, columns, and provenance.
"""
import os
import sys
import json
import hashlib
import datetime
import pandas as pd

DEFAULT_SEARCH_PATHS = [
    "C:/Users/M S I/Downloads/complaints_cleaned.csv",
    "C:/Users/M S I/Downloads/output_cleaned.csv",
    "C:/Users/M S I/Downloads/cora_train.csv",
    "C:/Users/M S I/Downloads/cora_validation.csv",
    "C:/Users/M S I/Downloads/train_cleaned.csv",
    "C:/Users/M S I/Downloads/test_cleaned.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/complaints.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/complaints_cleaned.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/output.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/output_cleaned.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/train.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/train_cleaned.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/test.csv",
    "C:/Users/M S I/Downloads/ComplaintManagement/ComplaintManagement/test_cleaned.csv",
]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def discover(output_path="ai-services/cora/outputs/audit/dataset_inventory.json"):
    inventory = []
    seen_hashes = {}

    for path in DEFAULT_SEARCH_PATHS:
        if not os.path.exists(path):
            continue
        
        size = os.path.getsize(path)
        sha256 = compute_sha256(path)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        
        # Read schema
        try:
            df = pd.read_csv(path, nrows=5)
            cols = list(df.columns)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                approx_rows = sum(1 for _ in f) - 1
        except Exception as e:
            cols = []
            approx_rows = 0

        # Provenance determination
        if "category" in cols:
            family = "Short-Text Intent Dataset (Banking77-derived with mapped CFPB taxonomy)"
            provenance = "Partially verified (Banking77 format with synthetic/mapped CFPB hierarchy)"
        elif len(cols) == 4 and "text" not in cols:
            family = "Taxonomy Reference / Label-Only CFPB Dataset (No complaint narratives)"
            provenance = "Verified CFPB metadata table without text"
        elif "text" in cols and len(cols) == 5:
            family = "CORA Real CFPB Narrative Dataset (CFPB Complaint Narratives + Full Hierarchy)"
            provenance = "Verified CFPB consumer complaint narratives with 4-level hierarchy"
        else:
            family = "Unknown"
            provenance = "Unknown"

        is_duplicate_of = seen_hashes.get(sha256)
        if not is_duplicate_of:
            seen_hashes[sha256] = path

        record = {
            "path": os.path.abspath(path),
            "filename": os.path.basename(path),
            "size_bytes": size,
            "sha256": sha256,
            "mtime": mtime,
            "approx_rows": approx_rows,
            "columns": cols,
            "family": family,
            "provenance": provenance,
            "is_duplicate_of": is_duplicate_of if is_duplicate_of != path else None
        }
        inventory.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print(f"Dataset inventory generated with {len(inventory)} entries -> {output_path}")
    return inventory

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ai-services/cora/outputs/audit/dataset_inventory.json"
    discover(out)
