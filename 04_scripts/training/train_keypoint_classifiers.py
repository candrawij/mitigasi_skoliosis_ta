"""
T3.2 EXP-02: Keypoint + Classifier experiments.

Trains MLP classifier on preprocessed keypoint features for:
- Postureexercise (7-keypoint, 5 classes)
- IKORN 4-KP (4-keypoint, 2 classes)

Classifiers: MLP (sklearn), then XGBoost if available.
Reports: accuracy, precision, recall, F1, macro F1, balanced accuracy,
         confusion matrix.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, confusion_matrix,
                             classification_report)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_PE = PROJECT_ROOT / "02_data" / "processed" / "postureexercise"
PROCESSED_IK = PROJECT_ROOT / "02_data" / "processed" / "ikorn_4kp"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "experiments"


def load_dataset(proc_dir):
    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_valid = np.load(proc_dir / "X_valid.npy")
    y_valid = np.load(proc_dir / "y_valid.npy")
    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    with open(proc_dir / "class_map.json") as f:
        class_map = json.load(f)

    with open(proc_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, class_map, feature_names


def evaluate(y_true, y_pred, class_names, dataset_name, model_name):
    """Compute and print evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    avg = "binary" if len(class_names) == 2 else "macro"
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    results = {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"  {dataset_name} — {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Precision ({avg}):  {prec:.4f}")
    print(f"  Recall ({avg}):     {rec:.4f}")
    print(f"  F1 ({avg}):         {f1:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return results


def train_mlp(X_train, y_train, X_valid, y_valid, n_classes):
    """Train MLP with validation-based early stopping."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    # Architecture based on dataset size
    if n_classes <= 2:
        hidden = (64, 32)
    else:
        hidden = (128, 64, 32)

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=0.001,
    )

    mlp.fit(X_train_s, y_train)

    train_acc = mlp.score(X_train_s, y_train)
    valid_acc = mlp.score(X_valid_s, y_valid)
    print(f"  MLP train acc: {train_acc:.4f}, valid acc: {valid_acc:.4f}")
    print(f"  Iterations: {mlp.n_iter_}")

    return mlp, scaler


def train_xgboost(X_train, y_train, X_valid, y_valid, n_classes):
    """Train XGBoost classifier if available."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not available, skipping.")
        return None, None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    objective = "binary:logistic" if n_classes == 2 else "multi:softmax"

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        early_stopping_rounds=20,
        random_state=42,
        use_label_encoder=False,
    )

    xgb.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)], verbose=False)

    train_acc = xgb.score(X_train_s, y_train)
    valid_acc = xgb.score(X_valid_s, y_valid)
    print(f"  XGBoost train acc: {train_acc:.4f}, valid acc: {valid_acc:.4f}")

    return xgb, scaler


def run_experiment(proc_dir, dataset_name, exp_id):
    """Run full experiment pipeline on one dataset."""
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {exp_id} — {dataset_name}")
    print(f"{'#'*60}")

    X_train, y_train, X_valid, y_valid, X_test, y_test, class_map, feature_names = load_dataset(proc_dir)

    class_names = [class_map[str(i)] for i in sorted(int(k) for k in class_map.keys())]
    n_classes = len(class_names)

    print(f"  Classes: {class_names}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # Check for NaN/Inf
    for name, arr in [("X_train", X_train), ("X_valid", X_valid), ("X_test", X_test)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"  WARNING: {name} contains NaN/Inf, replacing with 0")
            arr[np.isnan(arr)] = 0
            arr[np.isinf(arr)] = 0

    all_results = []
    exp_dir = RESULTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # === MLP ===
    print(f"\n--- Training MLP ---")
    mlp, mlp_scaler = train_mlp(X_train, y_train, X_valid, y_valid, n_classes)
    X_test_s = mlp_scaler.transform(X_test)
    y_pred_mlp = mlp.predict(X_test_s)
    mlp_results = evaluate(y_test, y_pred_mlp, class_names, dataset_name, "MLP")
    all_results.append(mlp_results)

    # === XGBoost ===
    print(f"\n--- Training XGBoost ---")
    xgb, xgb_scaler = train_xgboost(X_train, y_train, X_valid, y_valid, n_classes)
    if xgb is not None:
        X_test_s = xgb_scaler.transform(X_test)
        y_pred_xgb = xgb.predict(X_test_s)
        xgb_results = evaluate(y_test, y_pred_xgb, class_names, dataset_name, "XGBoost")
        all_results.append(xgb_results)

    # Save results
    with open(exp_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save comparison table
    comparison = []
    for r in all_results:
        comparison.append({
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "Balanced Acc": r["balanced_accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1_score"],
            "F1 Macro": r["f1_macro"],
        })
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(exp_dir / "comparison.csv", index=False)
    print(f"\n  Results saved to {exp_dir}")

    return all_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_exp_results = []

    # Experiment: Postureexercise keypoint (5-class)
    pe_results = run_experiment(PROCESSED_PE, "Postureexercise (7-KP, 5-class)", "EXP-PE-KP")
    all_exp_results.extend(pe_results)

    # Experiment: IKORN 4-KP (2-class)
    ik_results = run_experiment(PROCESSED_IK, "IKORN (4-KP, 2-class)", "EXP-IK-KP")
    all_exp_results.extend(ik_results)

    # Overall comparison
    print(f"\n{'#'*60}")
    print("  OVERALL COMPARISON")
    print(f"{'#'*60}")
    for r in all_exp_results:
        print(f"  {r['dataset']:40s} | {r['model']:10s} | Acc={r['accuracy']:.4f} | F1={r['f1_macro']:.4f}")

    with open(RESULTS_DIR / "exp02_all_results.json", "w") as f:
        json.dump(all_exp_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
