"""
train_private_xgboost_3d_canonical.py
======================================
FIX-05: Retrain XGBoost on canonical 3D features.

Uses:
  - private_features_3d_canonical.csv (22 features, canonical frame)
  - private_2d3d_intersection_5fold.csv (SAME fold file as all other experiments)
  - SAME 403 captures, SAME class order, SAME XGBoost protocol
  
Output:
  07_results/experiments/private_final/3d_canonical/
    fold_metrics.csv
    oof_predictions.csv
    classification_report.csv
    classification_report.txt
    confusion_matrix.png
"""

import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT / "04_scripts" / "preprocessing"))

FEAT_CSV    = ROOT / "02_data/private_processed/features/private_features_3d_canonical.csv"
FOLDS_CSV   = ROOT / "03_metadata/private_final_split/private_2d3d_intersection_5fold.csv"
OUT_DIR     = ROOT / "07_results/experiments/private_final/3d_canonical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = sorted(["upright","leaning_forward","leaning_backward","leaning_left","leaning_right","slouching"])
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

# Canonical feature schema (22 features — defined in extract_private_3d_features_canonical.py)
FEATURE_NAMES_3D_CANONICAL = [
    "nose_x", "nose_y", "nose_z",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
    "left_hip_x", "left_hip_y", "left_hip_z",
    "right_hip_x", "right_hip_y", "right_hip_z",
    "shoulder_roll_deg", "hip_roll_deg",
    "torso_3d_inclination_deg",
    "head_torso_angle_3d_deg",
    "head_depth_offset_norm", "head_lateral_offset_norm",
    "shoulder_depth_asymmetry_norm",
]

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 6,
    "tree_method": "hist",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("  FIX-05: TRAIN XGBOOST ON CANONICAL 3D FEATURES (22 FEATURES)")
    print("=" * 80)

    feat_df = pd.read_csv(FEAT_CSV)
    folds   = pd.read_csv(FOLDS_CSV)

    print(f"Feature CSV: {len(feat_df)} rows")
    print(f"Feature columns being used: {FEATURE_NAMES_3D_CANONICAL}")

    # Restrict to intersection
    intersection_caps = set(folds["capture_id"])
    feat_isect = feat_df[
        feat_df["capture_id"].isin(intersection_caps) & (feat_df["status_3d"] == "USABLE")
    ].copy()
    print(f"Intersection usable: {len(feat_isect)}")
    assert len(feat_isect) == 403, f"Expected 403, got {len(feat_isect)}"

    n_folds = folds["fold"].nunique()
    all_y_true, all_y_pred, all_caps, all_folds = [], [], [], []
    fold_accs, fold_f1s = [], []
    all_probs = []

    for fold_idx in range(n_folds):
        train_caps = set(folds[(folds["fold"] == fold_idx) & (folds["split"] == "train")]["capture_id"])
        test_caps  = set(folds[(folds["fold"] == fold_idx) & (folds["split"] == "test")]["capture_id"])

        train_df = feat_isect[feat_isect["capture_id"].isin(train_caps)].copy()
        test_df  = feat_isect[feat_isect["capture_id"].isin(test_caps)].copy()

        X_train = train_df[FEATURE_NAMES_3D_CANONICAL].values
        y_train = np.array([CLASS_TO_ID[lbl] for lbl in train_df["label"]])
        X_test  = test_df[FEATURE_NAMES_3D_CANONICAL].values
        y_test  = np.array([CLASS_TO_ID[lbl] for lbl in test_df["label"]])

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)

        print(f"  Fold {fold_idx} (train={len(train_df)}, test={len(test_df)}): "
              f"acc={acc:.4f}  macro_f1={f1:.4f}")

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_caps.extend(test_df["capture_id"].tolist())
        all_folds.extend([fold_idx] * len(test_df))
        all_probs.append(y_proba)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_probs_arr = np.vstack(all_probs)

    # OOF metrics
    pooled_acc  = accuracy_score(all_y_true, all_y_pred)
    pooled_prec = precision_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    pooled_rec  = recall_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    pooled_f1   = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    mean_fold_f1 = np.mean(fold_f1s)
    std_fold_f1  = np.std(fold_f1s)

    print(f"\n{'='*60}")
    print(f"OOF RESULTS (Canonical 3D, 22 features)")
    print(f"{'='*60}")
    print(f"  Pooled Accuracy    : {pooled_acc:.4f}")
    print(f"  Pooled Macro Prec  : {pooled_prec:.4f}")
    print(f"  Pooled Macro Rec   : {pooled_rec:.4f}")
    print(f"  Pooled Macro F1    : {pooled_f1:.4f}")
    print(f"  Fold Mean Macro F1 : {mean_fold_f1:.4f} ± {std_fold_f1:.4f}")

    # Save fold metrics
    fold_rows = [
        {
            "fold_id": i,
            "train_samples": len(folds[(folds["fold"] == i) & (folds["split"] == "train")]),
            "test_samples":  len(folds[(folds["fold"] == i) & (folds["split"] == "test")]),
            "accuracy": fold_accs[i],
            "f1_macro": fold_f1s[i],
        }
        for i in range(n_folds)
    ]
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_metrics.csv", index=False)

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "capture_id":  all_caps,
        "fold":        all_folds,
        "true_class_id": all_y_true,
        "pred_class_id": all_y_pred,
        "true_label":  [ID_TO_CLASS[y] for y in all_y_true],
        "pred_label":  [ID_TO_CLASS[y] for y in all_y_pred],
        "correct":     (all_y_true == all_y_pred).astype(int),
    })
    for i, cls in enumerate(CLASS_NAMES):
        oof_df[f"prob_{cls}"] = all_probs_arr[:, i]
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    # Classification report
    cr_text = classification_report(
        all_y_true, all_y_pred,
        target_names=CLASS_NAMES, zero_division=0
    )
    print(f"\n{cr_text}")
    with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cr_text)

    cr_dict = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES,
                                    zero_division=0, output_dict=True)
    cr_rows = []
    for cls in CLASS_NAMES:
        cr_rows.append({
            "class": cls,
            "precision": cr_dict[cls]["precision"],
            "recall":    cr_dict[cls]["recall"],
            "f1-score":  cr_dict[cls]["f1-score"],
            "support":   cr_dict[cls]["support"],
        })
    pd.DataFrame(cr_rows).to_csv(OUT_DIR / "classification_report.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(6)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Canonical 3D OOF Confusion Matrix (F1={pooled_f1:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Summary
    summary = {
        "model": "3D Canonical",
        "n_features": len(FEATURE_NAMES_3D_CANONICAL),
        "n_samples": len(feat_isect),
        "pooled_accuracy": float(pooled_acc),
        "pooled_macro_precision": float(pooled_prec),
        "pooled_macro_recall": float(pooled_rec),
        "pooled_macro_f1": float(pooled_f1),
        "fold_mean_macro_f1": float(mean_fold_f1),
        "fold_std_macro_f1": float(std_fold_f1),
        "fold_f1s": [float(f) for f in fold_f1s],
    }
    with open(OUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {OUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
