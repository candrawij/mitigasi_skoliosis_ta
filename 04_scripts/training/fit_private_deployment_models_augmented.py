"""
fit_private_deployment_models_augmented.py — Fit and Serialize Augmented Deployment Models
Saves production-ready model artifacts trained with augmented slouching and leaning_forward data.

Artifacts saved to:
  - 06_models/keypoint_2d/private_augmented/
      - xgboost_2d_deployment.pkl
      - model.pkl
      - pipeline.pkl
      - scaler.pkl
      - feature_schema.json
      - class_map.json
      - model_metadata.json
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "training"))

from private_feature_common import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    MAIN_CLASSES,
    NUM_CLASSES,
    FEATURE_NAMES_2D
)
from augment_slouching_dataset import augment_features, calculate_sample_weights

DATA_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
MODELS_ROOT = PROJECT_ROOT / "06_models"
DIR_2D_AUG = MODELS_ROOT / "keypoint_2d" / "private_augmented"
DIR_2D_AUG.mkdir(parents=True, exist_ok=True)


def get_xgb_device():
    return "cpu"


def fit_augmented_deployment_model(
    feature_file: Path,
    feature_names: list,
    out_dir: Path,
    device_type: str,
    n_augment_samples: int = 150,
    noise_level: float = 0.015,
    slouch_class_weight: float = 1.5,
    lean_fwd_weight: float = 1.3
):
    print("\n" + "=" * 80)
    print("  FITTING AUGMENTED DEPLOYMENT MODEL: 2D MULTI-VIEW")
    print(f"  Augmentation: +{n_augment_samples} samples for slouching & leaning_forward")
    print(f"  Sample Weights: Slouching={slouch_class_weight}, LeaningForward={lean_fwd_weight}")
    print("=" * 80)

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    df = pd.read_csv(feature_file)
    if "status_2d" in df.columns:
        df = df[df["status_2d"] == "USABLE"].copy().reset_index(drop=True)

    print(f"Original dataset: {len(df)} USABLE samples across {df['subject_id'].nunique()} subjects")
    print(f"Original class counts: {df['label'].value_counts().to_dict()}")

    X_raw = df[feature_names].values
    y_raw = df["class_id"].values
    groups = df["subject_id"].values

    target_classes = [CLASS_TO_ID["slouching"], CLASS_TO_ID["leaning_forward"]]

    # Augment features
    X_aug, y_aug = augment_features(
        X_train=X_raw,
        y_train=y_raw,
        target_class_ids=target_classes,
        n_samples_per_class=n_augment_samples,
        noise_level=noise_level,
        interpolation_ratio=0.5,
        random_state=42
    )

    class_weights_dict = {
        CLASS_TO_ID["slouching"]: slouch_class_weight,
        CLASS_TO_ID["leaning_forward"]: lean_fwd_weight
    }
    sample_weights_aug = calculate_sample_weights(y_aug, class_weights_dict)
    sample_weights_orig = calculate_sample_weights(y_raw, class_weights_dict)

    print(f"Augmented dataset total size: {len(X_aug)} samples (+{len(X_aug)-len(X_raw)} synthetic samples)")

    # Impute missing values (e.g. cam01 ear features) and Scale on augmented data
    imputer = SimpleImputer(strategy="median")
    X_imp_aug = imputer.fit_transform(X_aug)

    scaler = StandardScaler()
    X_scaled_aug = scaler.fit_transform(X_imp_aug)

    X_imp_orig = imputer.transform(X_raw)
    X_scaled_orig = scaler.transform(X_imp_orig)

    # Hyperparameter tuning using StratifiedGroupKFold on original subjects
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

    print("Running hyperparameter search on subject-aware CV...")
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
    search.fit(X_scaled_orig, y_raw, sample_weight=sample_weights_orig, groups=groups)

    best_params = search.best_params_
    best_cv_f1 = search.best_score_
    print(f"Best CV Macro F1: {best_cv_f1:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    # Fit final deployment model on ALL augmented samples
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
    final_model.fit(X_scaled_aug, y_aug, sample_weight=sample_weights_aug)

    # Evaluate on original real samples
    y_pred_real = final_model.predict(X_scaled_orig)
    real_acc = accuracy_score(y_raw, y_pred_real)
    real_f1 = f1_score(y_raw, y_pred_real, average="macro", zero_division=0)
    print(f"Performance on Original Dataset: Accuracy = {real_acc*100:.2f}% | Macro F1 = {real_f1:.4f}")
    print("\nDetailed Report on Original Data:")
    print(classification_report(y_raw, y_pred_real, target_names=MAIN_CLASSES, digits=4, zero_division=0))

    # Build Pipeline object: [imputer -> scaler -> xgb]
    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("xgb", final_model)
    ])

    # 1. Save Models
    model_name = "xgboost_2d_deployment.pkl"
    model_path = out_dir / model_name
    model_alias_path = out_dir / "model.pkl"
    pipeline_path = out_dir / "pipeline.pkl"
    scaler_path = out_dir / "scaler.pkl"
    imputer_path = out_dir / "imputer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(model_alias_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(imputer_path, "wb") as f:
        pickle.dump(imputer, f)

    print(f"[SAVED] Deployment Model:  {model_path}")
    print(f"[SAVED] Model Alias:       {model_alias_path}")
    print(f"[SAVED] Pipeline:          {pipeline_path}")
    print(f"[SAVED] Scaler:            {scaler_path}")
    print(f"[SAVED] Imputer:           {imputer_path}")

    # 2. Save Feature Schema
    schema_data = {
        "representation": "2d_augmented",
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

    # 4. Save Model Metadata
    metadata = {
        "model": "XGBoost",
        "representation": "2D_multi_view_augmented",
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "classes": MAIN_CLASSES,
        "pose_estimator": "yolov8n-pose.pt",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_dataset": str(feature_file.relative_to(PROJECT_ROOT)),
        "n_samples_orig": len(df),
        "n_samples_augmented": len(X_aug),
        "augmentation_config": {
            "n_augment_samples": n_augment_samples,
            "noise_level": noise_level,
            "slouch_class_weight": slouch_class_weight,
            "lean_fwd_weight": lean_fwd_weight
        },
        "n_subjects": int(df["subject_id"].nunique()),
        "subjects": sorted(df["subject_id"].unique().tolist()),
        "trained_timestamp": datetime.datetime.now().isoformat(),
        "tree_method": "hist",
        "device": device_type,
        "best_cv_macro_f1": round(best_cv_f1, 4),
        "original_data_accuracy": round(real_acc, 4),
        "original_data_macro_f1": round(real_f1, 4),
        "hyperparameters": best_params
    }
    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Model Metadata:    {meta_path}")

    return metadata


def main():
    device_type = get_xgb_device()
    print(f"XGBoost acceleration device: {device_type.upper()}")

    feat_2d_file = DATA_DIR / "private_features_2d_v2.csv"
    fit_augmented_deployment_model(
        feature_file=feat_2d_file,
        feature_names=FEATURE_NAMES_2D,
        out_dir=DIR_2D_AUG,
        device_type=device_type
    )


if __name__ == "__main__":
    main()
