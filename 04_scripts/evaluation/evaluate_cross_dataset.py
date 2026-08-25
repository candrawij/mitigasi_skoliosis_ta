"""
EXP-05: Cross-Dataset Generalization Evaluation.

Evaluates models trained on one dataset against test sets of other datasets
using the unified Binary Taxonomy (Good vs Bad Posture):
  - Project Design: upright (Good) vs others (Bad)
  - Postureexercise: thang (Good) vs others (Bad)
  - SPD: good_posture (Good) vs others (Bad)
  - IKORN: Good (Good) vs Bad (Bad)
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "07_results" / "experiments" / "EXP-05-CROSS-DATASET"


def load_postureexercise_binary():
    """Load Postureexercise keypoint features and map labels to Binary (0: Bad, 1: Good)."""
    proc_dir = PROJECT_ROOT / "02_data" / "processed" / "postureexercise"
    if not (proc_dir / "X_train.npy").exists():
        return None
    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    # Map class_id: 4 is 'thang' (Good = 1), 0..3 are lean/tilt (Bad = 0)
    y_train_bin = (y_train == 4).astype(np.int64)
    y_test_bin = (y_test == 4).astype(np.int64)

    return X_train, y_train_bin, X_test, y_test_bin


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== EXP-05: Cross-Dataset Evaluation ===")
    
    # We will record binary cross evaluations
    records = []
    
    # Example: Postureexercise Binary MLP model
    pe_data = load_postureexercise_binary()
    if pe_data is not None:
        X_train, y_train, X_test, y_test = pe_data
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        clf.fit(X_train_s, y_train)
        
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        
        records.append({
            "Train_Dataset": "Postureexercise (Binary)",
            "Test_Dataset": "Postureexercise (Test)",
            "Model": "MLP Binary",
            "Accuracy": round(acc, 4),
            "F1_Binary": round(f1, 4)
        })
        print(f"Postureexercise Self-Test (Binary): Acc={acc:.4f}, F1={f1:.4f}")
        
    df_res = pd.DataFrame(records)
    df_res.to_csv(RESULTS_DIR / "cross_dataset_results.csv", index=False)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
