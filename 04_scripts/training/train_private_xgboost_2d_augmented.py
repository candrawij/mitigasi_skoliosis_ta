"""
train_private_xgboost_2d_augmented.py — Train and Evaluate 2D XGBoost with Data Augmentation
Subject-Aware Stratified Grouped 5-Fold Cross-Validation on Intersection Dataset.

Ensures strict zero-leakage: Augmentation is applied ONLY to the training fold in each split.
Test folds remain 100% untouched real subject captures.

Outputs in 07_results/experiments/private_augmented/2d/:
  - fold_metrics.csv
  - oof_predictions.csv
  - classification_report.txt
  - confusion_matrix.png
  - summary_metrics.json
  - comparison_vs_baseline.json
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
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "training"))

from private_feature_common import CLASS_TO_ID, ID_TO_CLASS, MAIN_CLASSES, NUM_CLASSES, FEATURE_NAMES_2D
from augment_slouching_dataset import augment_features, calculate_sample_weights

FEATURES_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
SPLIT_DIR = PROJECT_ROOT / "03_metadata" / "private_final_split"
OUT_DIR = PROJECT_ROOT / "07_results" / "experiments" / "private_augmented" / "2d"
BASELINE_DIR = PROJECT_ROOT / "07_results" / "experiments" / "private_final" / "2d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_augmented_2d_xgboost(
    n_augment_samples: int = 120,
    noise_level: float = 0.015,
    slouch_class_weight: float = 1.5,
    lean_fwd_weight: float = 1.3
):
    print("=" * 80)
    print("  AUGMENTED 2D MULTI-VIEW XGBOOST TRAINING (SUBJECT-AWARE 5-FOLD CV)")
    print(f"  Target Augmentation Classes: slouching (ID {CLASS_TO_ID['slouching']}), "
          f"leaning_forward (ID {CLASS_TO_ID['leaning_forward']})")
    print(f"  Synthetic samples per class: {n_augment_samples} | Noise: {noise_level} | "
          f"Weights: Slouch={slouch_class_weight}, LeanFwd={lean_fwd_weight}")
    print("=" * 80)

    feat_file = FEATURES_DIR / "private_features_2d_intersection.csv"
    split_file = SPLIT_DIR / "private_stratified_group_5fold.csv"

    if not feat_file.exists() or not split_file.exists():
        raise FileNotFoundError("Prerequisite files missing. Check features and split files.")

    df_feat = pd.read_csv(feat_file)
    df_split = pd.read_csv(split_file)

    df = pd.merge(df_split[["capture_id", "fold_id"]], df_feat, on="capture_id", how="inner")
    print(f"Loaded dataset: {len(df)} captures across {df['subject_id'].nunique()} subjects")

    X_features = FEATURE_NAMES_2D
    print(f"Number of 2D features: {len(X_features)}")

    # Check device
    device_type = "cpu"
    print("Using CPU acceleration: device='cpu', tree_method='hist'")

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

    class_weights_dict = {
        CLASS_TO_ID["slouching"]: slouch_class_weight,
        CLASS_TO_ID["leaning_forward"]: lean_fwd_weight
    }

    fold_metrics = []
    oof_records = []
    best_params_per_fold = {}

    target_classes = [CLASS_TO_ID["slouching"], CLASS_TO_ID["leaning_forward"]]

    for fold in range(5):
        print(f"\n>>> Running Outer Fold {fold}/5 ...")
        train_mask = df["fold_id"] != fold
        test_mask = df["fold_id"] == fold

        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        train_subs = set(df_train["subject_id"])
        test_subs = set(df_test["subject_id"])
        assert len(train_subs.intersection(test_subs)) == 0, f"Leakage detected in Fold {fold}!"

        X_train_raw = df_train[X_features].values
        y_train_raw = df_train["class_id"].values
        groups_train_raw = df_train["subject_id"].values

        X_test = df_test[X_features].values
        y_test = df_test["class_id"].values

        # Apply augmentation strictly on training split
        X_train_aug, y_train_aug = augment_features(
            X_train=X_train_raw,
            y_train=y_train_raw,
            target_class_ids=target_classes,
            n_samples_per_class=n_augment_samples,
            noise_level=noise_level,
            interpolation_ratio=0.5,
            random_state=42 + fold
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test)

        # Compute sample weights
        sample_weights_train = calculate_sample_weights(y_train_aug, class_weights_dict)

        print(f"  Train: {len(X_train_raw)} original -> {len(X_train_aug)} augmented | Test: {len(df_test)}")

        # Inner CV for hyperparameter tuning using groups on original data
        # For inner CV, we can use 3-fold StratifiedKFold on augmented data
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        # For inner tuning, use the original group-aware data to pick stable hyperparameters
        X_train_orig_scaled = scaler.transform(X_train_raw)
        sample_weights_orig = calculate_sample_weights(y_train_raw, class_weights_dict)

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
            n_iter=25,
            scoring="f1_macro",
            cv=inner_cv,
            random_state=42,
            n_jobs=1,
            verbose=0
        )

        search.fit(X_train_orig_scaled, y_train_raw, sample_weight=sample_weights_orig, groups=groups_train_raw)
        best_params = search.best_params_
        best_params_per_fold[f"fold_{fold}"] = best_params

        # Fit model on the full augmented training set
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            tree_method="hist",
            device=device_type,
            random_state=42,
            n_jobs=4,
            **best_params
        )
        model.fit(X_train_scaled, y_train_aug, sample_weight=sample_weights_train)

        # Predict on Outer Test Fold (Unseen Subjects)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        p_mac = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_mac = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Calculate slouching-specific metrics
        slouch_id = CLASS_TO_ID["slouching"]
        slouch_mask = (y_test == slouch_id)
        slouch_recall = recall_score(y_test == slouch_id, y_pred == slouch_id, zero_division=0)
        slouch_prec = precision_score(y_test == slouch_id, y_pred == slouch_id, zero_division=0)
        slouch_f1 = f1_score(y_test == slouch_id, y_pred == slouch_id, zero_division=0)

        print(f"  >>> Fold {fold} Test: Acc={acc*100:.2f}%, Macro F1={f1_mac:.4f} | "
              f"Slouching: Rec={slouch_recall*100:.1f}%, Prec={slouch_prec*100:.1f}%, F1={slouch_f1:.4f}")

        fold_metrics.append({
            "fold_id": fold,
            "train_samples_orig": len(X_train_raw),
            "train_samples_aug": len(X_train_aug),
            "test_samples": len(df_test),
            "test_subjects": ",".join(sorted(list(test_subs))),
            "accuracy": round(acc, 4),
            "precision_macro": round(p_mac, 4),
            "recall_macro": round(r_mac, 4),
            "f1_macro": round(f1_mac, 4),
            "slouch_recall": round(slouch_recall, 4),
            "slouch_precision": round(slouch_prec, 4),
            "slouch_f1": round(slouch_f1, 4)
        })

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

    # Global Out-of-Fold Evaluation
    y_true_all = df_oof["y_true"].values
    y_pred_all = df_oof["y_pred"].values

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_p_mac = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    overall_r_mac = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    overall_f1_mac = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    clf_report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=MAIN_CLASSES,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 80)
    print("  OVERALL OUT-OF-FOLD (OOF) EVALUATION RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy:       {overall_acc*100:.2f}% (Mean across folds: {df_fold_metrics['accuracy'].mean()*100:.2f} ± {df_fold_metrics['accuracy'].std()*100:.2f}%)")
    print(f"Overall Macro Precision:{overall_p_mac:.4f} (Mean across folds: {df_fold_metrics['precision_macro'].mean():.4f})")
    print(f"Overall Macro Recall:   {overall_r_mac:.4f} (Mean across folds: {df_fold_metrics['recall_macro'].mean():.4f})")
    print(f"Overall Macro F1-Score: {overall_f1_mac:.4f} (Mean across folds: {df_fold_metrics['f1_macro'].mean():.4f} ± {df_fold_metrics['f1_macro'].std():.4f})")
    print("\nClassification Report:\n")
    print(clf_report)

    # Save artifacts
    df_fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    df_oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    with open(OUT_DIR / "best_params_per_fold.json", "w") as f:
        json.dump(best_params_per_fold, f, indent=2)
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(clf_report)

    # Confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=MAIN_CLASSES, yticklabels=MAIN_CLASSES, ax=axes[0])
    axes[0].set_title("Augmented 2D XGBoost — Raw Counts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=MAIN_CLASSES, yticklabels=MAIN_CLASSES, ax=axes[1])
    axes[1].set_title("Augmented 2D XGBoost — Normalized Recall", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # Compare against baseline if available
    summary = {
        "overall_accuracy": round(overall_acc, 4),
        "overall_macro_precision": round(overall_p_mac, 4),
        "overall_macro_recall": round(overall_r_mac, 4),
        "overall_macro_f1": round(overall_f1_mac, 4),
        "augmentation_config": {
            "n_augment_samples": n_augment_samples,
            "noise_level": noise_level,
            "slouch_class_weight": slouch_class_weight,
            "lean_fwd_weight": lean_fwd_weight
        }
    }
    with open(OUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    baseline_summary_file = BASELINE_DIR / "summary_metrics.json"
    if baseline_summary_file.exists():
        with open(baseline_summary_file) as f:
            base_summary = json.load(f)
        comparison = {
            "baseline": base_summary,
            "augmented": summary,
            "f1_delta": round(overall_f1_mac - base_summary.get("overall_macro_f1", 0), 4),
            "accuracy_delta": round(overall_acc - base_summary.get("overall_accuracy", 0), 4)
        }
        with open(OUT_DIR / "comparison_vs_baseline.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print("\n" + "=" * 80)
        print("  COMPARISON AGAINST BASELINE (2D UNTOUCHED)")
        print("=" * 80)
        print(f"Accuracy:  Baseline {base_summary.get('overall_accuracy', 0)*100:.2f}% -> Augmented {overall_acc*100:.2f}% (Delta: {comparison['accuracy_delta']*100:+.2f}%)")
        print(f"Macro F1:  Baseline {base_summary.get('overall_macro_f1', 0):.4f} -> Augmented {overall_f1_mac:.4f} (Delta: {comparison['f1_delta']:+.4f})")

    return summary


if __name__ == "__main__":
    train_augmented_2d_xgboost()
