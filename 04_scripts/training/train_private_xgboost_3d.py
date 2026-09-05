"""
train_private_xgboost_3d.py — Train and Evaluate Stereo 3D XGBoost (6 Classes)
Using Subject-Aware Stratified Grouped 5-Fold Cross-Validation on Intersection Dataset.

Inputs:
  - 02_data/private_processed/features/private_features_3d_intersection.csv
  - 03_metadata/private_final_split/private_stratified_group_5fold.csv

Outputs in 07_results/experiments/private_final/3d/:
  - fold_metrics.csv
  - oof_predictions.csv
  - best_params_per_fold.json
  - classification_report.txt
  - confusion_matrix.png
  - summary_metrics.json
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
from private_feature_common import CLASS_TO_ID, ID_TO_CLASS, MAIN_CLASSES, NUM_CLASSES, FEATURE_NAMES_3D

FEATURES_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
SPLIT_DIR = PROJECT_ROOT / "03_metadata" / "private_final_split"
OUT_DIR = PROJECT_ROOT / "07_results" / "experiments" / "private_final" / "3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_3d_xgboost():
    print("=" * 80)
    print("  STEP 11: TRAIN XGBOOST STEREO 3D (SUBJECT-AWARE 5-FOLD CV)")
    print("=" * 80)

    feat_file = FEATURES_DIR / "private_features_3d_intersection.csv"
    split_file = SPLIT_DIR / "private_stratified_group_5fold.csv"

    if not feat_file.exists() or not split_file.exists():
        raise FileNotFoundError("Prerequisite files missing. Ensure build_private_intersection.py and create_private_subject_folds.py ran!")

    df_feat = pd.read_csv(feat_file)
    df_split = pd.read_csv(split_file)

    # Merge on capture_id
    df = pd.merge(df_split[["capture_id", "fold_id"]], df_feat, on="capture_id", how="inner")
    print(f"Loaded dataset: {len(df)} captures across {df['subject_id'].nunique()} subjects")
    assert len(df) == 403, f"Expected 403 intersection captures, found {len(df)}"

    # Feature matrix X and target y
    X_features = FEATURE_NAMES_3D
    print(f"Number of 3D features: {len(X_features)}")

    # Check device
    device_type = "cuda"
    try:
        test_clf = xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method="hist", device="cuda")
        test_clf.fit(np.zeros((10, 5)), np.zeros(10))
        print("Using GPU acceleration: device='cuda', tree_method='hist'")
    except Exception:
        device_type = "cpu"
        print("Using CPU acceleration: device='cpu', tree_method='hist'")

    # Parameter distributions for inner tuning (identical protocol as 2D)
    param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3],
        "gamma": [0.0, 0.1, 0.5],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [0.1, 1.0, 2.0]
    }

    fold_metrics = []
    oof_records = []
    best_params_per_fold = {}

    for fold in range(5):
        print(f"\n>>> Running Outer Fold {fold}/5 ...")
        train_mask = df["fold_id"] != fold
        test_mask = df["fold_id"] == fold

        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        train_subs = set(df_train["subject_id"])
        test_subs = set(df_test["subject_id"])
        assert len(train_subs.intersection(test_subs)) == 0, f"Leakage detected in Fold {fold}!"

        print(f"  Train: {len(df_train)} captures ({len(train_subs)} subs) | "
              f"Test: {len(df_test)} captures ({len(test_subs)} subs: {sorted(list(test_subs))})")

        X_train = df_train[X_features].copy()
        y_train = df_train["class_id"].values
        groups_train = df_train["subject_id"].values

        X_test = df_test[X_features].copy()
        y_test = df_test["class_id"].values

        # Inner CV for hyperparameter tuning (Subject-Aware 3-Fold)
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        base_estimator = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            tree_method="hist",
            device=device_type,
            random_state=42,
            n_jobs=4
        )

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_dist,
            n_iter=30,
            scoring="f1_macro",
            cv=inner_cv,
            random_state=42,
            n_jobs=1,
            verbose=0
        )

        search.fit(X_train, y_train, groups=groups_train)
        best_model = search.best_estimator_
        best_params_per_fold[f"fold_{fold}"] = search.best_params_
        print(f"  Best Inner F1-Macro: {search.best_score_:.4f}")
        print(f"  Best Params: depth={search.best_params_['max_depth']}, lr={search.best_params_['learning_rate']}, "
              f"n_est={search.best_params_['n_estimators']}, subsample={search.best_params_['subsample']}")

        # Predict on Outer Test Fold (Unseen Subjects)
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        p_mac = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_mac = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"  >>> Fold {fold} Test Metrics: Acc = {acc*100:.2f}%, Macro F1 = {f1_mac:.4f}, "
              f"Macro P = {p_mac:.4f}, Macro R = {r_mac:.4f}")

        fold_metrics.append({
            "fold_id": fold,
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "test_subjects": ",".join(sorted(list(test_subs))),
            "accuracy": round(acc, 4),
            "precision_macro": round(p_mac, 4),
            "recall_macro": round(r_mac, 4),
            "f1_macro": round(f1_mac, 4)
        })

        # Save OOF predictions
        for i, (_, row_t) in enumerate(df_test.iterrows()):
            rec = {
                "capture_id": row_t["capture_id"],
                "subject_id": row_t["subject_id"],
                "fold_id": fold,
                "y_true": int(y_test[i]),
                "y_pred": int(y_pred[i]),
                "label_true": ID_TO_CLASS[int(y_test[i])],
                "label_pred": ID_TO_CLASS[int(y_pred[i])],
                "is_correct": bool(y_test[i] == y_pred[i])
            }
            for c_id in range(NUM_CLASSES):
                rec[f"prob_{ID_TO_CLASS[c_id]}"] = float(y_prob[i, c_id])
            oof_records.append(rec)

    df_fold_metrics = pd.DataFrame(fold_metrics)
    df_oof = pd.DataFrame(oof_records)

    # Sort OOF by capture_id
    df_oof = df_oof.sort_values("capture_id").reset_index(drop=True)

    # Overall OOF metrics
    oof_acc = accuracy_score(df_oof["y_true"], df_oof["y_pred"])
    oof_p_mac = precision_score(df_oof["y_true"], df_oof["y_pred"], average="macro", zero_division=0)
    oof_r_mac = recall_score(df_oof["y_true"], df_oof["y_pred"], average="macro", zero_division=0)
    oof_f1_mac = f1_score(df_oof["y_true"], df_oof["y_pred"], average="macro", zero_division=0)

    mean_acc = df_fold_metrics["accuracy"].mean()
    std_acc = df_fold_metrics["accuracy"].std()
    mean_f1 = df_fold_metrics["f1_macro"].mean()
    std_f1 = df_fold_metrics["f1_macro"].std()
    mean_p = df_fold_metrics["precision_macro"].mean()
    std_p = df_fold_metrics["precision_macro"].std()
    mean_r = df_fold_metrics["recall_macro"].mean()
    std_r = df_fold_metrics["recall_macro"].std()

    print("\n" + "=" * 80)
    print("  XGBOOST STEREO 3D — FINAL 5-FOLD OOF RESULTS")
    print("=" * 80)
    print(f"Overall OOF Accuracy:   {oof_acc*100:.2f}%")
    print(f"Overall OOF Macro F1:   {oof_f1_mac:.4f}")
    print(f"Overall OOF Macro P:    {oof_p_mac:.4f}")
    print(f"Overall OOF Macro R:    {oof_r_mac:.4f}")
    print(f"5-Fold Mean Accuracy:   {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"5-Fold Mean Macro F1:   {mean_f1:.4f} ± {std_f1:.4f}")

    # Per-Class Classification Report
    cls_report = classification_report(
        df_oof["y_true"], df_oof["y_pred"],
        target_names=MAIN_CLASSES, digits=4
    )
    print("\nDetailed Per-Class Classification Report:\n" + cls_report)

    # Confusion Matrix
    cm = confusion_matrix(df_oof["y_true"], df_oof["y_pred"], labels=list(range(NUM_CLASSES)))

    # Save Confusion Matrix Plot
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=MAIN_CLASSES, yticklabels=MAIN_CLASSES,
        cbar=True
    )
    plt.title(f"Confusion Matrix: XGBoost Stereo 3D (OOF N={len(df_oof)}, Macro F1={oof_f1_mac:.4f})", fontsize=12)
    plt.ylabel("Ground Truth Posture", fontsize=11)
    plt.xlabel("Predicted Posture", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_plot_path = OUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # Save summary JSON
    summary_data = {
        "model": "XGBoost",
        "representation": "stereo_3d",
        "n_samples": len(df_oof),
        "n_features": len(X_features),
        "features": X_features,
        "classes": MAIN_CLASSES,
        "overall_oof": {
            "accuracy": round(oof_acc, 4),
            "macro_precision": round(oof_p_mac, 4),
            "macro_recall": round(oof_r_mac, 4),
            "macro_f1": round(oof_f1_mac, 4)
        },
        "kfold_stats": {
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "mean_macro_precision": round(mean_p, 4),
            "std_macro_precision": round(std_p, 4),
            "mean_macro_recall": round(mean_r, 4),
            "std_macro_recall": round(std_r, 4)
        },
        "per_class_f1": {
            cls_name: round(f1_score(df_oof["y_true"] == c_id, df_oof["y_pred"] == c_id, zero_division=0), 4)
            for c_id, cls_name in enumerate(MAIN_CLASSES)
        }
    }

    # Save all output files
    out_metrics_csv = OUT_DIR / "fold_metrics.csv"
    out_oof_csv = OUT_DIR / "oof_predictions.csv"
    out_params_json = OUT_DIR / "best_params_per_fold.json"
    out_report_txt = OUT_DIR / "classification_report.txt"
    out_summary_json = OUT_DIR / "summary_metrics.json"

    df_fold_metrics.to_csv(out_metrics_csv, index=False)
    df_oof.to_csv(out_oof_csv, index=False)
    with open(out_params_json, "w") as fp:
        json.dump(best_params_per_fold, fp, indent=2)
    with open(out_report_txt, "w") as fp:
        fp.write(cls_report)
    with open(out_summary_json, "w") as fp:
        json.dump(summary_data, fp, indent=2)

    print(f"[SAVED] Fold Metrics:          {out_metrics_csv}")
    print(f"[SAVED] OOF Predictions:       {out_oof_csv}")
    print(f"[SAVED] Best Params:           {out_params_json}")
    print(f"[SAVED] Classification Report: {out_report_txt}")
    print(f"[SAVED] Confusion Matrix Plot: {cm_plot_path}")
    print(f"[SAVED] Summary Metrics:       {out_summary_json}")

    return summary_data


if __name__ == "__main__":
    train_3d_xgboost()
