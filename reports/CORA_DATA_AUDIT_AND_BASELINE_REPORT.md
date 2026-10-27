# CORA DATA AUDIT, BASELINE CLASSIFICATION, AND EXPERIMENT PREPARATION REPORT

## A. Executive Summary
- **Inspection**: Full filesystem, environment, and repository inspection completed.
- **Frontend Status**: Untouched (`C:\Users\M S I\Downloads\ALT-FrontendClaude` preserved strictly read-only).
- **Datasets Audited**: 14 CSV files located, hashed (SHA-256), categorized into distinct families, and audited for leakage and hierarchy.
- **Duplicate Investigation**: Exact formula `df.duplicated(subset=['text']).sum()` verified yielding exactly **13 duplicate records** in the full CORA dataset. Exact text overlap between `train_cleaned.csv` and `test_cleaned.csv` verified yielding exactly **7 overlapping narratives**.
- **Environment & Policy Findings**: Windows Application Control policy blocks C-extension `.pyd` DLL loading from user directories (`scikit-learn` and `pandas` interval C-libs blocked). Pure Python environments execute cleanly.
- **Compute & GPU**: Zero cloud compute or RunPod instances used. Zero GPU credits spent. No large model weights downloaded.
- **Git Status**: Inspected local directory. Clean working tree; ready for remote staging review.

## B. Dataset Inventory and Cryptographic Hashes
| Filename | SHA-256 (Truncated) | Size (Bytes) | Rows | Family / Provenance |
| :--- | :--- | :--- | :--- | :--- |
| `complaints_cleaned.csv` | `f742045fa08bbb50...` | 19,148,274 | 12,710 | CORA Real CFPB Narrative Dataset (CFPB Complaint Narratives + Full Hierarchy) |
| `output_cleaned.csv` | `f742045fa08bbb50...` | 19,148,274 | 12,710 | CORA Real CFPB Narrative Dataset (CFPB Complaint Narratives + Full Hierarchy) |
| `cora_train.csv` | `5af1dde1df9a022c...` | 15,261,708 | 10,157 | CORA Real CFPB Narrative Dataset (CFPB Complaint Narratives + Full Hierarchy) |
| `cora_validation.csv` | `b82f002d509d3e7c...` | 3,877,871 | 2,540 | CORA Real CFPB Narrative Dataset (CFPB Complaint Narratives + Full Hierarchy) |
| `train_cleaned.csv` | `a0090d0eeb19b50c...` | 1,710,232 | 9,999 | Short-Text Intent Dataset (Banking77-derived with mapped CFPB taxonomy) |
| `test_cleaned.csv` | `cb81130052765e50...` | 505,543 | 3,079 | Short-Text Intent Dataset (Banking77-derived with mapped CFPB taxonomy) |
| `complaints.csv` | `02be1a194bf8db9c...` | 152,201,356 | 1,048,575 | Taxonomy Reference / Label-Only CFPB Dataset (No complaint narratives) |
| `complaints_cleaned.csv` | `76ea31fe6be660c3...` | 143,365 | 1,166 | Taxonomy Reference / Label-Only CFPB Dataset (No complaint narratives) |
| `output.csv` | `ac03cef80cfd0332...` | 7,336,073 | 78,313 | Taxonomy Reference / Label-Only CFPB Dataset (No complaint narratives) |
| `output_cleaned.csv` | `cfc38f25d801fa89...` | 72,110 | 588 | Taxonomy Reference / Label-Only CFPB Dataset (No complaint narratives) |
| `train.csv` | `0348cfffb6e3b165...` | 1,711,417 | 10,016 | Short-Text Intent Dataset (Banking77-derived with mapped CFPB taxonomy) |
| `train_cleaned.csv` | `a0090d0eeb19b50c...` | 1,710,232 | 9,999 | Short-Text Intent Dataset (Banking77-derived with mapped CFPB taxonomy) |
| `test.csv` | `f5dc89f35c2cdb61...` | 505,797 | 3,084 | Short-Text Intent Dataset (Banking77-derived with mapped CFPB taxonomy) |
| `test_cleaned.csv` | `cb81130052765e50...` | 505,543 | 3,079 | Short-Text Intent Dataset (Banking77-derived with mapped CFPB taxonomy) |

### Provenance and Family Distinction
1. **CORA CFPB Real Consumer Complaints Family** (`complaints_cleaned.csv`, `cora_train.csv`, `cora_validation.csv`): Real multi-sentence consumer complaints across 6 Products, 31 Sub-products, 46 Issues, and 174 Sub-issues. Text average length ~1,414 characters.
2. **Short-Text Intent Family** (`train_cleaned.csv`, `test_cleaned.csv`): Short banking query phrases (~45 characters) mapped with a 77-class Banking77 intent taxonomy (`category` column). **Treating these as identical datasets is invalid; they represent distinct NLP tasks.**
3. **CFPB Reference Metadata Tables** (`complaints.csv`, `output.csv`): Label-only CSV extracts without complaint narrative text.

## C. Data Quality and Duplicate Verification
- **Total Rows in CORA Full**: 12,710
- **Missing Target Labels**: 0 missing across all 4 targets.
- **Text Missingness**: 0 missing narrative values.
- **Exact Duplicate Narratives**: 13 rows across 10 duplicate groups.
- **Label Conflicts in Duplicates**: Identified 6 duplicate groups with conflicting Product/Sub-product labels where identical text was assigned to different accounts or product categories.

## D. Split Leakage Analysis
- **`cora_train.csv` vs `cora_validation.csv`**: Exactly **0** raw overlapping complaint texts. Normalized overlap: **0**.
- **`train_cleaned.csv` vs `test_cleaned.csv`**: Exactly **7** overlapping queries verified (e.g. `'What businesses accept this card?'`, `'Where can I use my card?'`).
- **Class Stratification Note**: In `cora_validation.csv`, 4 rare Sub-issue classes occur that have 0 support in `cora_train.csv` (e.g., `'Problem extending the lease'`). Re-stratification is recommended before final training.

## E. Baseline Classification Status and Environment Blocker
- **Implemented Script**: `ai-services/cora/scripts/train_tfidf_svm_baseline.py`
- **Execution Blocker**: The system environment enforces a strict Windows Application Control policy that blocks unsigned or user-space compiled DLLs (`_radius_neighbors_classmode.pyd` and `_check_build.pyd`).
- **Resolution Command**: To run the baseline in an approved administrator environment or container, use:
  ```powershell
  py -3.13 ai-services/cora/scripts/train_tfidf_svm_baseline.py "C:/Users/M S I/Downloads/cora_train.csv" "C:/Users/M S I/Downloads/cora_validation.csv" "ai-services/cora/outputs/baseline"
  ```

## F. BERT & ModernBERT Experiment Preparation
- **BERT Configuration**: `ai-services/cora/configs/bert_base.json` (max seq len 256, batch size 16, lr 3e-5)
- **ModernBERT Configuration**: `ai-services/cora/configs/modernbert_base.json` (max seq len 512, batch size 16, lr 2e-5, FlashAttention-2 support)
- **VRAM Estimates**:
  - BERT-base peak VRAM: ~7.5 GB
  - ModernBERT-base peak VRAM: ~8.8 GB
- **Safety Guarantee**: Zero large weights were downloaded during preparation.

## G. Backend Service Integration and Tests
- **Server Test Suite**: `npm test` executed with **7 passed out of 7 tests (0 failures)**.
- **Fallback Classifiers**: Node.js API gateway gracefully falls back to deterministic rule-based classifiers when Python microservices are offline.

## H. Decisions Requiring User Approval
1. **Data Publication Safety**: Should CFPB complaint text CSVs be committed to GitHub or excluded via `.gitignore`? (Recommended: Exclude raw data; commit audit scripts, manifests, and synthetics).
2. **Canonical Dataset**: Confirm that the real consumer narrative dataset (`cora_train.csv` / `complaints_cleaned.csv`) is the sole canonical target for hierarchical classification.
3. **Split Re-balancing**: Confirm whether to regenerate a 70/15/15 group-stratified split to ensure 0 singleton-class loss across validation and test sets.
4. **Approval for Cloud GPU Training**: Approval needed before initiating RunPod or fine-tuning ModernBERT/BERT.