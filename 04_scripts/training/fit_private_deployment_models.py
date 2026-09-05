"""
fit_private_deployment_models.py — Fit and Serialize Deployment XGBoost Models (2D & 3D)
Saves production-ready model artifacts, scalers, schemas, and metadata for both representations.

Artifacts saved to:
  - 06_models/keypoint_2d/private_final/
      - xgboost_2d_deployment.pkl
      - model.pkl
      - pipeline.pkl
      - scaler.pkl
      - feature_schema.json
      - class_map.json
      - model_metadata.json
  - 06_models/keypoint_3d/private_final/
      - xgboost_3d_deployment.pkl
      - model.pkl
      - pipeline.pkl
      - scaler.pkl
      - feature_schema.json
      - class_map.json
      - model_metadata.json
      - coordinate_convention.json
"""

import os
import sys
import json
import pickle
import shutil
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
from private_feature_common import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    MAIN_CLASSES,
    NUM_CLASSES,
    FEATURE_NAMES_2D,
    FEATURE_NAMES_3D,
    COORDINATE_CONVENTION_3D
)

DATA_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
MODELS_ROOT = PROJECT_ROOT / "06_models"
DIR_2D = MODELS_ROOT / "keypoint_2d" / "private_final"
DIR_3D = MODELS_ROOT / "keypoint_3d" / "private_final"

DIR_2D.mkdir(parents=True, exist_ok=True)
DIR_3D.mkdir(parents=True, exist_ok=True)


def get_xgb_device():
    try:
        test_clf = xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method="hist", device="cuda")
        test_clf.fit(np.zeros((10, 5)), np.zeros(10))
        return "cuda"
    except Exception:
        return "cpu"


def fit_and_save_deployment_model(
    mode: str,
    feature_file: Path,
    feature_names: list,
    out_dir: Path,
    device_type: str
):
    print("\n" + "=" * 80)
    print(f"  FITTING DEPLOYMENT MODEL: {mode.upper()}")
    print("=" * 80)

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    df = pd.read_csv(feature_file)
    # Filter only USABLE samples (rejected/invalid inputs are handled by Reject Gate)
    if "status_2d" in df.columns:
        df = df[df["status_2d"] == "USABLE"].copy().reset_index(drop=True)
    elif "status_3d" in df.columns:
        df = df[df["status_3d"] == "USABLE"].copy().reset_index(drop=True)

    print(f"Loaded dataset: {len(df)} USABLE samples across {df['subject_id'].nunique()} subjects")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    X = df[feature_names].values
    y = df["class_id"].values
    groups = df["subject_id"].values

    # Check for NaNs and handle imputations if necessary
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"Note: Found {nan_count} NaN values in feature matrix (XGBoost handles missing values natively).")

    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hyperparameter tuning using StratifiedGroupKFold on subject_id
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

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    base_xgb = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist",
        device=device_type,
        random_state=42,
        n_jobs=4
    )

    print("Running hyperparameter search on full dataset (subject-aware CV)...")
    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=25,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=0
    )
    search.fit(X_scaled, y, groups=groups)

    best_params = search.best_params_
    best_cv_f1 = search.best_score_
    print(f"Best CV Macro F1: {best_cv_f1:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    # Fit final deployment model on ALL samples using optimal hyperparameters
    final_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist",
        device=device_type,
        random_state=42,
        n_jobs=4,
        **best_params
    )
    final_model.fit(X_scaled, y)

    # Evaluate on training set
    y_pred_train = final_model.predict(X_scaled)
    train_acc = accuracy_score(y, y_pred_train)
    train_f1 = f1_score(y, y_pred_train, average="macro", zero_division=0)
    print(f"Training Set Accuracy: {train_acc*100:.2f}% | Macro F1: {train_f1:.4f}")

    # Build Pipeline object for convenience
    pipeline = Pipeline([
        ("scaler", scaler),
        ("xgb", final_model)
    ])

    # 1. Save Models
    model_name = f"xgboost_{mode.lower()}_deployment.pkl"
    model_path = out_dir / model_name
    model_alias_path = out_dir / "model.pkl"
    pipeline_path = out_dir / "pipeline.pkl"
    scaler_path = out_dir / "scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(model_alias_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[SAVED] Deployment Model:  {model_path}")
    print(f"[SAVED] Model Alias:       {model_alias_path}")
    print(f"[SAVED] Pipeline:          {pipeline_path}")
    print(f"[SAVED] Scaler:            {scaler_path}")

    # 2. Save Feature Schema
    schema_data = {
        "representation": mode,
        "n_features": len(feature_names),
        "features": [
            {"index": i, "name": fn, "dtype": "float64"}
            for i, fn in enumerate(feature_names)
        ]
    }
    schema_path = out_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=2)
    print(f"[SAVED] Feature Schema:    {schema_path}")

    # 3. Save Class Map
    class_map_data = {
        "class_to_id": CLASS_TO_ID,
        "id_to_class": {str(k): v for k, v in ID_TO_CLASS.items()},
        "classes": MAIN_CLASSES,
        "num_classes": NUM_CLASSES
    }
    class_map_path = out_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map_data, f, indent=2)
    print(f"[SAVED] Class Map:         {class_map_path}")

    # 4. Save Coordinate Convention (for 3D)
    if mode.lower() == "3d":
        coord_conv_path = out_dir / "coordinate_convention.json"
        with open(coord_conv_path, "w", encoding="utf-8") as f:
            json.dump(COORDINATE_CONVENTION_3D, f, indent=2)
        print(f"[SAVED] Coord Convention:  {coord_conv_path}")

    # 5. Save Model Metadata
    metadata = {
        "model": "XGBoost",
        "representation": "2D_multi_view" if mode.lower() == "2d" else "stereo_3d",
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "classes": MAIN_CLASSES,
        "pose_estimator": "yolov8n-pose.pt",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_dataset": str(feature_file.relative_to(PROJECT_ROOT)),
        "n_samples": len(df),
        "n_subjects": int(df["subject_id"].nunique()),
        "subjects": sorted(df["subject_id"].unique().tolist()),
        "trained_timestamp": datetime.datetime.now().isoformat(),
        "tree_method": "hist",
        "device": device_type,
        "best_cv_macro_f1": round(best_cv_f1, 4),
        "training_accuracy": round(train_acc, 4),
        "training_macro_f1": round(train_f1, 4),
        "hyperparameters": best_params
    }
    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Model Metadata:    {meta_path}")

    return metadata


def main():
    print("=" * 80)
    print("  STEP 16: FIT AND SERIALIZE PRIVATE DEPLOYMENT MODELS")
    print("=" * 80)

    device_type = get_xgb_device()
    print(f"XGBoost acceleration device: {device_type.upper()}")

    # 1. 2D Deployment Model
    feat_2d_file = DATA_DIR / "private_features_2d.csv"
    meta_2d = fit_and_save_deployment_model(
        mode="2d",
        feature_file=feat_2d_file,
        feature_names=FEATURE_NAMES_2D,
        out_dir=DIR_2D,
        device_type=device_type
    )

    # 2. 3D Deployment Model
    feat_3d_file = DATA_DIR / "private_features_3d.csv"
    meta_3d = fit_and_save_deployment_model(
        mode="3d",
        feature_file=feat_3d_file,
        feature_names=FEATURE_NAMES_3D,
        out_dir=DIR_3D,
        device_type=device_type
    )

    print("\n" + "=" * 80)
    print("  ALL DEPLOYMENT MODELS AND ARTIFACTS SERIALIZED SUCCESSFULLY!")
    print("=" * 80)
    print(f"2D Deployment Directory: {DIR_2D}")
    print(f"3D Deployment Directory: {DIR_3D}")


if __name__ == "__main__":
    main()
