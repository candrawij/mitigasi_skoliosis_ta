"""
3D-12 -- Compare 2D vs 3D (Fair Comparison on Intersection)
============================================================
Trains 2D model on the intersection subset using the same 5-fold split,
then compares 2D vs 3D metrics.

Output:
    07_results/experiments/private_final/comparison_2d_vs_3d.csv
    07_results/experiments/private_final/comparison_2d_vs_3d_per_class.csv
    07_results/experiments/private_final/comparison_2d_vs_3d.txt
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(r"d:\.Candra\Project\TA")
FEAT2D_CSV  = ROOT / "02_data/private_processed/features/private_features_2d_v2.csv"
FEAT3D_CSV  = ROOT / "02_data/private_processed/features/private_features_3d.csv"
FOLDS_CSV   = ROOT / "03_metadata/private_final_split/private_2d3d_intersection_5fold.csv"
OOF3D_CSV   = ROOT / "07_results/experiments/private_final/3d/oof_predictions.csv"
OUT_DIR     = ROOT / "07_results/experiments/private_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = sorted(["upright","leaning_forward","leaning_backward","leaning_left","leaning_right","slouching"])
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

FEATURE_NAMES_2D = [
    "cam01_nose_x","cam01_nose_y","cam01_left_shoulder_x","cam01_left_shoulder_y",
    "cam01_right_shoulder_x","cam01_right_shoulder_y","cam01_left_hip_x","cam01_left_hip_y",
    "cam01_right_hip_x","cam01_right_hip_y","cam01_shoulder_slope_deg","cam01_hip_slope_deg",
    "cam01_torso_inclination_deg","cam01_head_torso_angle_deg","cam01_head_to_shoulder_norm",
    "cam01_torso_length_norm","cam01_head_horizontal_offset_norm","cam01_torso_horizontal_offset_norm",
    "cam01_ear_shoulder_horizontal_norm","cam01_ear_shoulder_vertical_norm","cam01_ear_neck_angle_deg",
    "cam02_nose_x","cam02_nose_y","cam02_left_shoulder_x","cam02_left_shoulder_y",
    "cam02_right_shoulder_x","cam02_right_shoulder_y","cam02_left_hip_x","cam02_left_hip_y",
    "cam02_right_hip_x","cam02_right_hip_y","cam02_shoulder_slope_deg","cam02_hip_slope_deg",
    "cam02_torso_inclination_deg","cam02_head_torso_angle_deg","cam02_head_to_shoulder_norm",
    "cam02_torso_length_norm","cam02_head_horizontal_offset_norm","cam02_torso_horizontal_offset_norm",
    "cam02_ear_shoulder_horizontal_norm","cam02_ear_shoulder_vertical_norm","cam02_ear_neck_angle_deg"
]

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
    "verbosity": 0
}

def train_fold_2d(feat2d, folds, fold_idx):
    fold_data = folds[folds["fold"] == fold_idx]
    train_caps = set(fold_data[fold_data["split"]=="train"]["capture_id"])
    test_caps  = set(fold_data[fold_data["split"]=="test"]["capture_id"])

    df_train = feat2d[feat2d["capture_id"].isin(train_caps)]
    df_test  = feat2d[feat2d["capture_id"].isin(test_caps)]

    X_train = df_train[FEATURE_NAMES_2D].values.astype(np.float64)
    y_train = df_train["class_id"].values
    X_test  = df_test[FEATURE_NAMES_2D].values.astype(np.float64)
    y_test  = df_test["class_id"].values

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)

    return y_test, y_pred, list(df_test["capture_id"])

def main():
    print("=" * 70)
    print("3D-12 -- Compare 2D vs 3D (Fair Intersection Comparison)")
    print("=" * 70)

    warnings.filterwarnings("ignore")

    feat2d = pd.read_csv(FEAT2D_CSV)
    feat3d = pd.read_csv(FEAT3D_CSV)
    folds  = pd.read_csv(FOLDS_CSV)
    oof3d  = pd.read_csv(OOF3D_CSV)

    # Filter 2D to intersection
    intersection_caps = set(folds["capture_id"])
    feat2d_isect = feat2d[feat2d["capture_id"].isin(intersection_caps)].copy()
    feat3d_isect = feat3d[(feat3d["capture_id"].isin(intersection_caps)) &
                          (feat3d["status_3d"]=="USABLE")].copy()

    print(f"\n2D in intersection: {len(feat2d_isect)}")
    print(f"3D in intersection: {len(feat3d_isect)}")

    n_folds = folds["fold"].nunique()

    # ---- Train 2D on same folds ----
    print("\nTraining 2D on intersection (same 5-fold)...")
    all_y_true_2d = []
    all_y_pred_2d = []
    all_caps_2d   = []

    for fold_idx in range(n_folds):
        y_test, y_pred, caps = train_fold_2d(feat2d_isect, folds, fold_idx)
        all_y_true_2d.extend(y_test)
        all_y_pred_2d.extend(y_pred)
        all_caps_2d.extend(caps)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"  Fold {fold_idx}: acc={acc:.4f} F1={f1:.4f}")

    y_true_2d = np.array(all_y_true_2d)
    y_pred_2d = np.array(all_y_pred_2d)

    # ---- Aggregate 3D OOF ----
    y_true_3d = oof3d["true_class_id"].values
    y_pred_3d = oof3d["pred_class_id"].values

    # ---- Overall Comparison ----
    def compute_metrics(y_true, y_pred, name):
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {"name": name, "accuracy": acc, "macro_precision": prec, "macro_recall": rec, "macro_f1": f1}

    m2d = compute_metrics(y_true_2d, y_pred_2d, "2D Multi-View")
    m3d = compute_metrics(y_true_3d, y_pred_3d, "3D Stereo")

    print("\n" + "=" * 70)
    print("OVERALL COMPARISON (Intersection, Subject-Aware 5-Fold)")
    print("=" * 70)
    header = f"{'Metric':<25} {'2D Multi-View':>15} {'3D Stereo':>15}"
    print(header)
    print("-" * 55)
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        print(f"{key:<25} {m2d[key]:>15.4f} {m3d[key]:>15.4f}")

    df_overall = pd.DataFrame([m2d, m3d])

    # ---- Per-class comparison ----
    f1_2d = f1_score(y_true_2d, y_pred_2d, average=None, zero_division=0, labels=list(range(6)))
    f1_3d = f1_score(y_true_3d, y_pred_3d, average=None, zero_division=0, labels=list(range(6)))
    prec_2d = precision_score(y_true_2d, y_pred_2d, average=None, zero_division=0, labels=list(range(6)))
    prec_3d = precision_score(y_true_3d, y_pred_3d, average=None, zero_division=0, labels=list(range(6)))
    rec_2d  = recall_score(y_true_2d, y_pred_2d, average=None, zero_division=0, labels=list(range(6)))
    rec_3d  = recall_score(y_true_3d, y_pred_3d, average=None, zero_division=0, labels=list(range(6)))

    print("\n" + "=" * 70)
    print("PER-CLASS F1 COMPARISON")
    print("=" * 70)
    print(f"{'Class':<22} {'2D F1':>8} {'3D F1':>8} {'Delta':>8}")
    print("-" * 50)
    per_class_rows = []
    for i, cls in enumerate(CLASS_NAMES):
        delta = f1_3d[i] - f1_2d[i]
        print(f"{cls:<22} {f1_2d[i]:>8.4f} {f1_3d[i]:>8.4f} {delta:>+8.4f}")
        per_class_rows.append({
            "class": cls,
            "f1_2d": f1_2d[i], "f1_3d": f1_3d[i], "f1_delta_3d_minus_2d": delta,
            "precision_2d": prec_2d[i], "precision_3d": prec_3d[i],
            "recall_2d": rec_2d[i], "recall_3d": rec_3d[i]
        })

    df_per_class = pd.DataFrame(per_class_rows)

    # ---- Additional info ----
    print("\n" + "=" * 70)
    print("ADDITIONAL INFO")
    print("=" * 70)
    print(f"  2D usable (full dataset)   : 704")
    print(f"  3D usable (intersection)   : {len(feat3d_isect)}")
    print(f"  2D on intersection         : {len(feat2d_isect)}")
    print(f"  Coverage (3D/2D total)     : {len(feat3d_isect)/704*100:.1f}%")

    # Save outputs
    report_2d = classification_report(y_true_2d, y_pred_2d, target_names=CLASS_NAMES, zero_division=0)
    report_3d = classification_report(y_true_3d, y_pred_3d, target_names=CLASS_NAMES, zero_division=0)

    out_txt = OUT_DIR / "comparison_2d_vs_3d.txt"
    with open(out_txt, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("2D vs 3D COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<25} {'2D Multi-View':>15} {'3D Stereo':>15}\n")
        f.write("-" * 55 + "\n")
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            f.write(f"{key:<25} {m2d[key]:>15.4f} {m3d[key]:>15.4f}\n")
        f.write("\n\nPER-CLASS F1:\n")
        f.write(df_per_class.to_string(index=False))
        f.write("\n\n2D CLASSIFICATION REPORT (on intersection):\n" + report_2d)
        f.write("\n3D CLASSIFICATION REPORT (OOF):\n" + report_3d)

    df_overall.to_csv(OUT_DIR / "comparison_2d_vs_3d.csv", index=False)
    df_per_class.to_csv(OUT_DIR / "comparison_2d_vs_3d_per_class.csv", index=False)
    print(f"\n[DONE] Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
