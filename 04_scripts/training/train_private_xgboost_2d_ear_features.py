"""
train_private_xgboost_2d_ear_features.py — Evaluate 42-Feature 2D XGBoost with Ear Sagittal Features
Subject-Aware Stratified Grouped 5-Fold Cross-Validation.

Compares against 36-feature baseline and 42-feature augmented model.

Outputs in 07_results/experiments/private_ear_v2/2d/:
  - fold_metrics.csv
  - oof_predictions.csv
  - classification_report.txt
  - confusion_matrix.png
  - summary_metrics.json
  - comparison_vs_baseline.json
"""

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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))

from private_feature_common import CLASS_TO_ID, ID_TO_CLASS, MAIN_CLASSES, NUM_CLASSES, FEATURE_NAMES_2D

FEATURES_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
SPLIT_DIR = PROJECT_ROOT / "03_metadata" / "private_final_split"
OUT_DIR = PROJECT_ROOT / "07_results" / "experiments" / "private_ear_v2" / "2d"
BASELINE_RESULTS = PROJECT_ROOT / "07_results" / "experiments" / "private_final" / "2d" / "summary_metrics.json"
AUGMENTED_RESULTS = PROJECT_ROOT / "07_results" / "experiments" / "private_augmented" / "2d" / "summary_metrics.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_ear_feature_2d_xgboost():
    print("=" * 80)
    print("  EAR FEATURE 2D XGBOOST — SUBJECT-AWARE 5-FOLD CV (42 FEATURES)")
    print("  New features: cam02_ear_shoulder_horizontal_norm, cam02_ear_shoulder_vertical_norm,")
    print("                cam02_ear_neck_angle_deg")
    print("=" * 80)

    feat_file = FEATURES_DIR / "private_features_2d_v2.csv"
    split_file = SPLIT_DIR / "private_stratified_group_5fold.csv"

    if not feat_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_file}\nRun extract_private_2d_features.py first!")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    df_feat = pd.read_csv(feat_file)
    df_split = pd.read_csv(split_file)

    # Merge on capture_id (intersection dataset)
    df = pd.merge(df_split[["capture_id", "fold_id"]], df_feat, on="capture_id", how="inner")
    print(f"Loaded dataset: {len(df)} captures across {df['subject_id'].nunique()} subjects")

    X_features = FEATURE_NAMES_2D  # 42 features
    print(f"Number of 2D features: {len(X_features)} (includes 3 ear-based sagittal features)")

    ear_features = [f for f in X_features if 'ear' in f]
    print(f"Ear features: {ear_features}")

    # Check NaN coverage of ear features in dataset
    df_usable = df[df['status_2d'] == 'USABLE'] if 'status_2d' in df.columns else df
    for ef in [f for f in ear_features if f.startswith('cam02')]:
        nan_pct = df_usable[ef].isna().mean() * 100
        print(f"  {ef}: {nan_pct:.1f}% NaN")

    device_type = "cpu"
    print("\nUsing CPU acceleration: device='cpu', tree_method='hist'")

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

        X_train = df_train[X_features].values
        y_train = df_train["class_id"].values
        groups_train = df_train["subject_id"].values

        X_test = df_test[X_features].values
        y_test = df_test["class_id"].values

        # Impute NaN (cam01 ear features are always NaN — impute with 0 median for consistency)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imp)

        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

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
            n_iter=25,
            scoring="f1_macro",
            cv=inner_cv,
            random_state=42,
            n_jobs=1,
            verbose=0
        )

        search.fit(X_train_scaled, y_train, groups=groups_train)
        best_params = search.best_params_
        best_params_per_fold[f"fold_{fold}"] = best_params
        print(f"  Best Inner F1-Macro: {search.best_score_:.4f}")

        # Fit best model
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
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        p_mac = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_mac = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)

        slouch_id = CLASS_TO_ID["slouching"]
        slouch_recall = recall_score(y_test == slouch_id, y_pred == slouch_id, zero_division=0)
        slouch_prec = precision_score(y_test == slouch_id, y_pred == slouch_id, zero_division=0)
        slouch_f1 = f1_score(y_test == slouch_id, y_pred == slouch_id, zero_division=0)

        lean_id = CLASS_TO_ID["leaning_forward"]
        lean_recall = recall_score(y_test == lean_id, y_pred == lean_id, zero_division=0)

        print(f"  >>> Fold {fold}: Acc={acc*100:.2f}%, Macro F1={f1_mac:.4f} | "
              f"Slouching: Rec={slouch_recall*100:.1f}%, F1={slouch_f1:.4f} | "
              f"LeanFwd Recall={lean_recall*100:.1f}%")

        fold_metrics.append({
            "fold_id": fold,
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "test_subjects": ",".join(sorted(list(test_subs))),
            "accuracy": round(acc, 4),
            "precision_macro": round(p_mac, 4),
            "recall_macro": round(r_mac, 4),
            "f1_macro": round(f1_mac, 4),
            "slouch_recall": round(slouch_recall, 4),
            "slouch_f1": round(slouch_f1, 4),
            "lean_fwd_recall": round(lean_recall, 4)
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

    y_true_all = df_oof["y_true"].values
    y_pred_all = df_oof["y_pred"].values

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_p_mac = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    overall_r_mac = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    overall_f1_mac = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    clf_report = classification_report(
        y_true_all, y_pred_all,
        target_names=MAIN_CLASSES, digits=4, zero_division=0
    )

    print("\n" + "=" * 80)
    print("  OVERALL OUT-OF-FOLD (OOF) EVALUATION — EAR FEATURE MODEL (42 features)")
    print("=" * 80)
    print(f"Overall Accuracy:        {overall_acc*100:.2f}%")
    print(f"Overall Macro F1-Score:  {overall_f1_mac:.4f}")
    print(f"\nClassification Report:\n")
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=MAIN_CLASSES, yticklabels=MAIN_CLASSES, ax=axes[0])
    axes[0].set_title("Ear Feature 2D XGBoost — Raw Counts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=MAIN_CLASSES, yticklabels=MAIN_CLASSES, ax=axes[1])
    axes[1].set_title("Ear Feature 2D XGBoost — Normalized Recall", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    summary = {
        "overall_accuracy": round(overall_acc, 4),
        "overall_macro_precision": round(overall_p_mac, 4),
        "overall_macro_recall": round(overall_r_mac, 4),
        "overall_macro_f1": round(overall_f1_mac, 4),
        "n_features": len(X_features),
        "ear_features_added": [f for f in X_features if 'ear' in f and f.startswith('cam02')]
    }
    with open(OUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Comparison against baselines
    print("\n" + "=" * 80)
    print("  COMPARISON AGAINST BASELINES")
    print("=" * 80)
    comparison = {"ear_v2": summary}
    for label, path in [("baseline_36feat", BASELINE_RESULTS), ("augmented_36feat", AUGMENTED_RESULTS)]:
        if path.exists():
            with open(path) as f:
                base = json.load(f)
            comparison[label] = base
            base_f1 = base.get("overall_macro_f1", base.get("overall_oof", {}).get("macro_f1", 0))
            delta = overall_f1_mac - base_f1
            print(f"  vs {label:<25}: Macro F1 {base_f1:.4f} → {overall_f1_mac:.4f} (Delta: {delta:+.4f})")

    with open(OUT_DIR / "comparison_vs_baseline.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return summary


if __name__ == "__main__":
    train_ear_feature_2d_xgboost()
