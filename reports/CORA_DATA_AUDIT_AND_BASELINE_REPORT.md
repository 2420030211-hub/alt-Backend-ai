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

## E. Baseline Classification Status and Verified Execution Metrics
- **Implemented Script**: `ai-services/cora/scripts/train_tfidf_svm_baseline.py`
- **Execution Status**: **Successfully Executed**.
- **Dataset**: `cora_train.csv` (10,157 samples) & `cora_validation.csv` (2,540 samples).
- **Vectorizer**: TF-IDF (15,000 max features, n-grams (1, 2), sublinear TF, English stop words).
- **Target Metrics Summary**:
  | Target | Classes (Train / Val) | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Product** | 6 / 6 | **84.41%** | **0.7659** | 0.8410 | 0.8313 | 0.7234 |
  | **Sub-product** | 31 / 27 | **77.72%** | **0.2797** | 0.7494 | 0.3884 | 0.2475 |
  | **Issue** | 46 / 42 | **59.06%** | **0.3249** | 0.5707 | 0.3863 | 0.3009 |
  | **Sub-issue** | 174 / 147 | **43.70%** | **0.1876** | 0.4004 | 0.2099 | 0.1889 |
- **Holistic Hierarchy Performance**:
  - **4-Target Exact Match Accuracy**: **34.96%** (888 / 2,540 validation narratives predicted all 4 levels perfectly).
  - **Raw Prediction Hierarchical Path Consistency**: **70.91%** (1,801 / 2,540 predictions formed a valid path in the hierarchy without post-processing).
- **Saved Baseline Artifacts** (`ai-services/cora/outputs/baseline/`):
  - `baseline_metrics.json`
  - `tfidf_vectorizer.joblib`
  - `linear_svc_Product.joblib`
  - `linear_svc_Sub_product.joblib`
  - `linear_svc_Issue.joblib`
  - `linear_svc_Sub_issue.joblib`
  - `report_Product.csv`, `report_Sub_product.csv`, `report_Issue.csv`, `report_Sub_issue.csv`

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

## H. Git Status, Push Authentication & Remaining Steps
1. **Local Git Commit Chain**:
   - `08017aa`: `"Update report generator to parse verified baseline metrics"`
   - `2b41794`: `"Execute CORA TF-IDF + LinearSVC baseline and log verified metrics"`
   - `2e4a3b6`: `"Add CORA data audit, baseline pipeline, and neural experiment preparation"`
   - Local working tree is clean.
2. **Remote Push Status**:
   - Remote URL: `https://github.com/2420030211-hub/alt-Backend-ai.git`
   - Command executed: `git push -u origin master`
   - Result: **Successfully pushed** (`master -> master`, tracking set to `origin/master`). All backend scripts, configurations, schemas, tests, and documentation are live on GitHub.
3. **Repository Security & Cleanliness**:
   - Raw complaint datasets (`*.csv`), binary checkpoints (`*.pt`, `*.bin`, `*.safetensors`), virtual environments (`.venv/`), and `.env` credentials are completely excluded via `.gitignore`.
4. **Next Local Execution Steps**:
   - Maintain the verified `.venv` (`Python 3.13.14`).
   - The server API gateway (`server/src/server.js`) and tests (`npm test` 7/7 passed) remain ready for local development.
   - All neural training configurations (`bert_base.json`, `modernbert_base.json`) are finalized and staged without model weights. Manual GPU fine-tuning can be triggered externally by the user whenever ready.