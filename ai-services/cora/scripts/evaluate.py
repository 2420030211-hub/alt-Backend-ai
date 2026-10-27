"""
Fair Comparison Benchmark:
1. TF-IDF + Linear SVM (Baseline)
2. BERT (google-bert/bert-base-uncased)
3. ModernBERT (answerdotai/ModernBERT-base)
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

def evaluate_baseline(train_csv: str, val_csv: str, target: str = "Product"):
    print(f"\n--- Training & Evaluating Baseline (TF-IDF + Linear SVM) on Target: {target} ---")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    text_col = "text" if "text" in train_df.columns else "Consumer complaint narrative"

    train_df = train_df.dropna(subset=[text_col, target])
    val_df = val_df.dropna(subset=[text_col, target])

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df[text_col])
    y_train = train_df[target]

    X_val = vectorizer.transform(val_df[text_col])
    y_val = val_df[target]

    clf = LinearSVC(random_state=42, max_iter=2000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro")
    weighted_f1 = f1_score(y_val, preds, average="weighted")

    print(f"Results for TF-IDF + Linear SVM ({target}):")
    print(f"  Accuracy:    {acc * 100:.2f}%")
    print(f"  Macro F1:    {macro_f1 * 100:.2f}%")
    print(f"  Weighted F1: {weighted_f1 * 100:.2f}%")

    return {"model": "TF-IDF + Linear SVM", "target": target, "accuracy": acc, "macro_f1": macro_f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="C:/Users/M S I/Downloads/banking-complaint-backend/datasets/cora/cora_train.csv")
    parser.add_argument("--val", default="C:/Users/M S I/Downloads/banking-complaint-backend/datasets/cora/cora_validation.csv")
    parser.add_argument("--target", default="Product")
    args = parser.parse_args()

    if os.path.exists(args.train) and os.path.exists(args.val):
        evaluate_baseline(args.train, args.val, args.target)
    else:
        print(f"Dataset split files not found at {args.train}. Run split_dataset.py first.")
