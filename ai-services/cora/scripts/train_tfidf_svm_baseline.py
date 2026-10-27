"""
Train TF-IDF + LinearSVC Baseline Classifier on CORA Banking Complaints.
Trains 4 separate classifiers for:
  1. Product
  2. Sub-product
  3. Issue
  4. Sub-issue

Calculates:
- Accuracy, Macro-F1, Weighted-F1, Precision, Recall
- Per-class metrics
- Confusion matrix
- 4-Target Exact Match Accuracy
- Hierarchical Path Consistency Rate
- Zero-division handling documented (zero_division=0)
Saves:
- Metrics in JSON and CSV
- Classification reports
- Trained models and TfidfVectorizers via joblib
- Confusion matrices
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

TARGETS = ["Product", "Sub-product", "Issue", "Sub-issue"]

def train_and_evaluate(train_csv: str, val_csv: str, output_dir: str):
    print(f"Loading training data: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"Loading validation data: {val_csv}")
    val_df = pd.read_csv(val_csv)

    text_col = "text" if "text" in train_df.columns else "Consumer complaint narrative"
    
    # Text vectorization (TF-IDF fitted ONLY on training narratives)
    print("Fitting TfidfVectorizer on training complaint narratives...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        min_df=2
    )
    
    X_train = vectorizer.fit_transform(train_df[text_col].fillna("").astype(str))
    X_val = vectorizer.transform(val_df[text_col].fillna("").astype(str))
    
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    results = {
        "train_set_size": len(train_df),
        "validation_set_size": len(val_df),
        "targets": {},
        "overall_exact_match": {},
        "hierarchy_consistency": {}
    }

    val_predictions = {}
    val_ground_truth = {}
    
    # Train separate baseline models
    for target in TARGETS:
        print(f"\n================ Training Baseline for: {target} ================")
        y_train = train_df[target].astype(str)
        y_val = val_df[target].astype(str)
        
        clf = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual="auto")
        clf.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(output_dir, f"linear_svc_{target.replace('-', '_')}.joblib")
        joblib.dump(clf, model_path)
        
        # Predictions
        preds = clf.predict(X_val)
        val_predictions[target] = preds
        val_ground_truth[target] = y_val.values
        
        # Unique classes
        train_classes = set(clf.classes_)
        val_classes = set(y_val.unique())
        unseen_classes = list(val_classes - train_classes)
        
        # Compute metrics with zero_division=0
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
        macro_prec = precision_score(y_val, preds, average="macro", zero_division=0)
        weighted_prec = precision_score(y_val, preds, average="weighted", zero_division=0)
        macro_rec = recall_score(y_val, preds, average="macro", zero_division=0)
        weighted_rec = recall_score(y_val, preds, average="weighted", zero_division=0)
        
        report_dict = classification_report(y_val, preds, output_dict=True, zero_division=0)
        
        # Save per-class report
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv_path = os.path.join(output_dir, f"report_{target.replace('-', '_')}.csv")
        report_df.to_csv(report_csv_path)

        # Confusion matrix
        unique_labels = sorted(list(train_classes.union(val_classes)))
        cm = confusion_matrix(y_val, preds, labels=unique_labels)
        
        results["targets"][target] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "macro_precision": round(float(macro_prec), 4),
            "weighted_precision": round(float(weighted_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "weighted_recall": round(float(weighted_rec), 4),
            "train_classes_count": len(train_classes),
            "val_classes_count": len(val_classes),
            "unseen_in_train_classes": unseen_classes,
            "confusion_matrix_shape": list(cm.shape)
        }
        print(f"Target {target} -> Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")

    # 4-target exact match evaluation
    exact_matches = 0
    total_val = len(val_df)
    for i in range(total_val):
        all_correct = True
        for target in TARGETS:
            if val_predictions[target][i] != val_ground_truth[target][i]:
                all_correct = False
                break
        if all_correct:
            exact_matches += 1
            
    exact_match_acc = exact_matches / total_val
    results["overall_exact_match"] = {
        "eligible_rows": total_val,
        "exact_matches_count": exact_matches,
        "exact_match_accuracy": round(float(exact_match_acc), 4)
    }
    print(f"\n================ OVERALL EXACT MATCH ACCURACY (All 4 Targets) ================")
    print(f"Exact Matches: {exact_matches}/{total_val} ({exact_match_acc*100:.2f}%)")

    # Hierarchy consistency check of raw predictions
    # Check if predicted (Product, Sub-product, Issue, Sub-issue) exists in train data valid paths
    valid_train_paths = set(train_df[TARGETS].itertuples(index=False, name=None))
    consistent_preds_count = 0
    for i in range(total_val):
        pred_tuple = tuple(val_predictions[target][i] for target in TARGETS)
        if pred_tuple in valid_train_paths:
            consistent_preds_count += 1
            
    consistency_rate = consistent_preds_count / total_val
    results["hierarchy_consistency"] = {
        "valid_train_paths_count": len(valid_train_paths),
        "consistent_predictions_count": consistent_preds_count,
        "consistency_rate": round(float(consistency_rate), 4)
    }
    print(f"Hierarchical Path Consistency of Raw Model Predictions: {consistent_preds_count}/{total_val} ({consistency_rate*100:.2f}%)")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "baseline_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline metrics and artifacts saved successfully to {output_dir}")
    return results

if __name__ == "__main__":
    tr = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/M S I/Downloads/cora_train.csv"
    va = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/M S I/Downloads/cora_validation.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "ai-services/cora/outputs/baseline"
    train_and_evaluate(tr, va, out)
