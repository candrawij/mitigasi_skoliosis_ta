"""
3D-13 -- Fit Deployment Model 3D (Trained on Full Intersection Dataset)
=======================================================================
Trains on all 403 usable 3D samples from the intersection.
Saves pipeline and metadata for inference.

Output:
    06_models/keypoint_3d/private_final/
        xgboost_3d.pkl
        feature_schema.json
        class_map.json
        model_metadata.json
        coordinate_convention.json
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

ROOT = Path(r"d:\.Candra\Project\TA")
FEAT3D_CSV  = ROOT / "02_data/private_processed/features/private_features_3d.csv"
ISECT_CSV   = ROOT / "02_data/private_processed/manifests/private_6class_intersection.csv"
OUT_DIR     = ROOT / "06_models/keypoint_3d/private_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = sorted(["upright","leaning_forward","leaning_backward","leaning_left","leaning_right","slouching"])
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

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

def main():
    print("=" * 60)
    print("3D-13 -- Fit Deployment Model 3D")
    print("=" * 60)

    warnings.filterwarnings("ignore")

    feat3d = pd.read_csv(FEAT3D_CSV)
    isect  = pd.read_csv(ISECT_CSV)

    intersection_caps = set(isect["capture_id"])
    df_train = feat3d[
        (feat3d["capture_id"].isin(intersection_caps)) &
        (feat3d["status_3d"] == "USABLE")
    ].copy()

    print(f"\nTraining on {len(df_train)} usable 3D samples")

    X = df_train[FEATURE_NAMES_3D].values.astype(np.float64)
    y = df_train["class_id"].values

    print(f"Class distribution:")
    for cls, cnt in pd.Series(y).value_counts().sort_index().items():
        print(f"  {CLASS_NAMES[cls]:<22} : {cnt}")

    # Build deployment pipeline: Imputer -> XGBClassifier
    imputer = SimpleImputer(strategy="median")
    clf = xgb.XGBClassifier(**XGB_PARAMS)

    pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", clf)
    ])

    print("\nFitting deployment pipeline...")
    pipeline.fit(X, y)
    print("[OK] Pipeline fitted.")

    # Verify predictions work
    y_pred_train = pipeline.predict(X)
    train_acc = (y_pred_train == y).mean()
    print(f"Training accuracy (sanity): {train_acc:.4f}")

    # Save pipeline
    model_path = OUT_DIR / "xgboost_3d.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[SAVED] Pipeline -> {model_path}")

    # Save feature_schema.json
    schema = {
        "feature_names": FEATURE_NAMES_3D,
        "n_features": len(FEATURE_NAMES_3D),
        "nan_features": ["nose_x","nose_y","nose_z","head_torso_angle_3d_deg","head_depth_offset_norm","head_lateral_offset_norm"],
        "imputer_strategy": "median",
        "version": "1.0"
    }
    with open(OUT_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Save class_map.json
    with open(OUT_DIR / "class_map.json", "w") as f:
        json.dump({"class_to_id": CLASS_TO_ID, "id_to_class": {str(k): v for k, v in ID_TO_CLASS.items()}}, f, indent=2)

    # Save coordinate_convention.json
    convention = {
        "reference_frame": "CAM01 camera coordinate frame",
        "X": "lateral (positive = camera-right = subject-left)",
        "Y": "vertical (positive = downward; negative = upward)",
        "Z": "depth (positive = away from camera)",
        "unit": "meters",
        "centering": "hip_center = (left_hip + right_hip) / 2",
        "scale": "S3 = max(||P - hip_center||) over core valid joints"
    }
    with open(OUT_DIR / "coordinate_convention.json", "w") as f:
        json.dump(convention, f, indent=2)

    # Save model_metadata.json
    metadata = {
        "model_type": "XGBoost 3D Posture Classifier",
        "trained_on": "private_6class_intersection",
        "n_samples": len(df_train),
        "n_features": len(FEATURE_NAMES_3D),
        "n_classes": 6,
        "classes": CLASS_NAMES,
        "xgb_params": XGB_PARAMS,
        "pipeline_steps": ["SimpleImputer(median)", "XGBClassifier"],
        "training_accuracy": round(float(train_acc), 4),
        "cv_macro_f1_mean": 0.6046,
        "cv_macro_f1_std": 0.2816,
        "cv_accuracy_mean": 0.6291,
        "created": datetime.now().isoformat(),
        "evaluation_note": "CV F1 average is dragged down by Fold 3 (degenerate test subjects). See fold_metrics.csv."
    }
    with open(OUT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SAVED] feature_schema.json")
    print(f"[SAVED] class_map.json")
    print(f"[SAVED] coordinate_convention.json")
    print(f"[SAVED] model_metadata.json")
    print(f"\n[DONE] Deployment model saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
