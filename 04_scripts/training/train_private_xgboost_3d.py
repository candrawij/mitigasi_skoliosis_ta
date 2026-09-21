"""
3D-10+11 -- Train XGBoost 3D and Evaluate (Subject-Aware 5-Fold)
=================================================================
Outputs:
    07_results/experiments/private_final/3d/fold_metrics.csv
    07_results/experiments/private_final/3d/oof_predictions.csv
    07_results/experiments/private_final/3d/best_params_per_fold.json
    07_results/experiments/private_final/3d/classification_report.txt
    07_results/experiments/private_final/3d/confusion_matrix.png
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

ROOT = Path(r"d:\.Candra\Project\TA")
FEAT3D_CSV  = ROOT / "02_data/private_processed/features/private_features_3d.csv"
FOLDS_CSV   = ROOT / "03_metadata/private_final_split/private_2d3d_intersection_5fold.csv"
OUT_DIR     = ROOT / "07_results/experiments/private_final/3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES_3D = [
    "nose_x","nose_y","nose_z",
    "left_shoulder_x","left_shoulder_y","left_shoulder_z",
    "right_shoulder_x","right_shoulder_y","right_shoulder_z",
    "left_hip_x","left_hip_y","left_hip_z",
    "right_hip_x","right_hip_y","right_hip_z",
    "shoulder_roll_deg","hip_roll_deg",
    "torso_lateral_lean_deg","torso_sagittal_lean_deg","torso_3d_inclination_deg",
    "head_torso_angle_3d_deg","head_depth_offset_norm","head_lateral_offset_norm",
    "shoulder_depth_asymmetry_norm","hip_depth_asymmetry_norm"
]

CLASS_NAMES = sorted(["upright","leaning_forward","leaning_backward","leaning_left","leaning_right","slouching"])
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

# XGBoost default params (no tuning — lock before seeing test)
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
    "n_jobs": -1,
    "verbosity": 0,
    "eval_metric": "mlogloss"
}

def main():
    print("=" * 70)
    print("3D-10+11 -- XGBoost 3D Training & Evaluation (Subject-Aware 5-Fold)")
    print("=" * 70)

    warnings.filterwarnings("ignore")

    feat3d = pd.read_csv(FEAT3D_CSV)
    folds  = pd.read_csv(FOLDS_CSV)

    # Only usable samples
    feat3d_usable = feat3d[feat3d["status_3d"] == "USABLE"].copy()
    print(f"\nLoaded: {len(feat3d_usable)} usable 3D samples")
    print(f"Fold file: {len(folds)} rows")

    n_folds = folds["fold"].nunique()
    print(f"Number of folds: {n_folds}")

    oof_rows = []
    fold_metrics = []
    best_params_per_fold = {}

    for fold_idx in range(n_folds):
        fold_data = folds[folds["fold"] == fold_idx]
        train_caps = set(fold_data[fold_data["split"]=="train"]["capture_id"])
        test_caps  = set(fold_data[fold_data["split"]=="test"]["capture_id"])

        df_train = feat3d_usable[feat3d_usable["capture_id"].isin(train_caps)]
        df_test  = feat3d_usable[feat3d_usable["capture_id"].isin(test_caps)]

        X_train = df_train[FEATURE_NAMES_3D].values.astype(np.float64)
        y_train = df_train["class_id"].values
        X_test  = df_test[FEATURE_NAMES_3D].values.astype(np.float64)
        y_test  = df_test["class_id"].values

        # Impute NaN (nose features) with median
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        # Train XGBoost
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"\nFold {fold_idx}: train={len(df_train)} test={len(df_test)}")
        print(f"  Accuracy={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

        fold_metrics.append({
            "fold": fold_idx,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1
        })

        best_params_per_fold[f"fold_{fold_idx}"] = XGB_PARAMS.copy()

        for i, cap_id in enumerate(df_test["capture_id"].values):
            oof_rows.append({
                "capture_id": cap_id,
                "fold": fold_idx,
                "true_class_id": int(y_test[i]),
                "pred_class_id": int(y_pred[i]),
                "true_label": CLASS_NAMES[y_test[i]],
                "pred_label": CLASS_NAMES[y_pred[i]],
                "correct": int(y_test[i] == y_pred[i]),
                **{f"prob_{CLASS_NAMES[j]}": float(y_prob[i,j]) for j in range(6)}
            })

    # Aggregate metrics
    df_fold_metrics = pd.DataFrame(fold_metrics)
    df_oof = pd.DataFrame(oof_rows)

    print("\n" + "=" * 70)
    print("FOLD METRICS SUMMARY")
    print("=" * 70)
    print(df_fold_metrics.to_string(index=False))
    print(f"\nMacro F1: {df_fold_metrics['macro_f1'].mean():.4f} +/- {df_fold_metrics['macro_f1'].std():.4f}")
    print(f"Accuracy:  {df_fold_metrics['accuracy'].mean():.4f} +/- {df_fold_metrics['accuracy'].std():.4f}")

    # OOF overall classification report
    y_true_all = df_oof["true_class_id"].values
    y_pred_all = df_oof["pred_class_id"].values

    cr = classification_report(y_true_all, y_pred_all, target_names=CLASS_NAMES, zero_division=0)
    print("\n" + "=" * 70)
    print("OOF CLASSIFICATION REPORT (All Folds Combined)")
    print("=" * 70)
    print(cr)

    # Confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    tick_marks = np.arange(6)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=9)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — XGBoost 3D (OOF, Subject-Aware 5-Fold)")
    plt.tight_layout()
    cm_path = OUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()

    # Save all outputs
    df_fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    df_oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    with open(OUT_DIR / "best_params_per_fold.json", "w") as f:
        json.dump(best_params_per_fold, f, indent=2)
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(cr)

    print(f"\n[DONE] Results saved to {OUT_DIR}")
    print(f"  fold_metrics.csv")
    print(f"  oof_predictions.csv")
    print(f"  confusion_matrix.png")
    print(f"  classification_report.txt")
    print(f"  best_params_per_fold.json")

if __name__ == "__main__":
    main()
