"""
EXP-03: YOLO Pose + Classifier Pipeline.

Architecture:
  Image -> YOLOv8 Pose (pretrained) -> 17 COCO Body Keypoints -> Feature Engineering -> Classifier (MLP / XGBoost) -> Posture Class

Datasets supported:
  1. project_design (5 classes)
  2. sitting_posture_detection (4 classes)
"""
import os
import json
import math
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
RESULTS_DIR = PROJECT_ROOT / "07_results" / "experiments"

# COCO 17 Keypoints mapping
COCO_KP = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def extract_pose_keypoints(image_paths, model_name="yolov8n-pose.pt"):
    """Run YOLO Pose model on a list of images and extract 17 keypoints + confidences."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for YOLO Pose. Install via: pip install ultralytics")

    print(f"Loading YOLO Pose model ({model_name})...")
    model = YOLO(model_name)

    all_kps = []
    print(f"Extracting pose from {len(image_paths)} images...")
    
    # Run batch inference with progress
    results = model(image_paths, verbose=False)

    for i, r in enumerate(results):
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            # Take the person detection with highest confidence (index 0 if sorted)
            kpts_tensor = r.keypoints.data[0].cpu().numpy()  # shape: (17, 3) -> x, y, conf
            # Get image dimensions for normalization
            orig_shape = r.orig_shape  # (height, width)
            h, w = orig_shape[0], orig_shape[1]

            kpt_dict = {}
            for kp_idx, name in enumerate(COCO_KP):
                kx, ky, conf = kpts_tensor[kp_idx]
                kpt_dict[f"{name}_x"] = float(kx / w) if w > 0 else 0.0
                kpt_dict[f"{name}_y"] = float(ky / h) if h > 0 else 0.0
                kpt_dict[f"{name}_conf"] = float(conf)
            kpt_dict["pose_detected"] = True
        else:
            # No person / pose detected: fill with zeros
            kpt_dict = {f"{name}_x": 0.0 for name in COCO_KP}
            kpt_dict.update({f"{name}_y": 0.0 for name in COCO_KP})
            kpt_dict.update({f"{name}_conf": 0.0 for name in COCO_KP})
            kpt_dict["pose_detected"] = False

        all_kps.append(kpt_dict)

    return pd.DataFrame(all_kps)


def compute_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def compute_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def engineer_yolo_pose_features(df_kps):
    """Extract geometric and biomechanical features from 17 YOLO pose keypoints."""
    feature_rows = []

    for _, row in df_kps.iterrows():
        f = {}
        if not row.get("pose_detected", True):
            # If no pose detected, zero features
            f = {f"feat_{i}": 0.0 for i in range(25)}
            feature_rows.append(f)
            continue

        pts = {name: (row[f"{name}_x"], row[f"{name}_y"]) for name in COCO_KP}

        # Shoulder midpoint & width & inclination
        ls, rs = pts["left_shoulder"], pts["right_shoulder"]
        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        f["shoulder_width"] = compute_distance(ls, rs)
        f["shoulder_angle"] = compute_angle(ls, rs)
        f["shoulder_dy"] = rs[1] - ls[1]

        # Hip midpoint & width & inclination
        lh, rh = pts["left_hip"], pts["right_hip"]
        hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        f["hip_width"] = compute_distance(lh, rh)
        f["hip_angle"] = compute_angle(lh, rh)

        # Head / Nose position relative to shoulder center
        nose = pts["nose"]
        f["nose_to_shoulder_dx"] = nose[0] - shoulder_mid[0]
        f["nose_to_shoulder_dy"] = nose[1] - shoulder_mid[1]
        f["head_tilt_angle"] = compute_angle(shoulder_mid, nose)

        # Torso spine axis (hip center to shoulder center)
        f["torso_length"] = compute_distance(hip_mid, shoulder_mid)
        f["torso_angle"] = compute_angle(hip_mid, shoulder_mid)
        f["torso_lean_dx"] = shoulder_mid[0] - hip_mid[0]

        # Ear symmetry (head lateral tilt)
        lear, rear = pts["left_ear"], pts["right_ear"]
        f["ear_angle"] = compute_angle(lear, rear)
        f["ear_dy"] = rear[1] - lear[1]

        # Eye inclination
        leye, reye = pts["left_eye"], pts["right_eye"]
        f["eye_angle"] = compute_angle(leye, reye)

        # Raw normalized coordinates for key upper body points
        for name in ["nose", "left_shoulder", "right_shoulder", "left_ear", "right_ear", "left_hip", "right_hip"]:
            f[f"raw_{name}_x"] = pts[name][0]
            f[f"raw_{name}_y"] = pts[name][1]

        feature_rows.append(f)

    return pd.DataFrame(feature_rows)


def load_dataset_paths(dataset_name):
    """Load image paths and labels according to final_split manifest."""
    if dataset_name == "project_design":
        split_csv = PROJECT_ROOT / "03_metadata" / "final_split" / "project_design.csv"
        image_dir = PROJECT_ROOT / "02_data" / "interim" / "project_design_curated"
        classes = ["leaning_backward", "leaning_forward", "leaning_left", "leaning_right", "upright"]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        df_split = pd.read_csv(split_csv)
        data_by_split = {}

        for s in ["train", "valid", "test"]:
            sub = df_split[df_split["split"] == s]
            paths, labels = [], []
            for _, r in sub.iterrows():
                cname = r["class_name"]
                if cname not in class_to_idx:
                    continue
                # Look for file
                for ext in [".jpg", ".jpeg", ".png"]:
                    p = image_dir / cname / (str(r["image_id"]) + ext)
                    if p.exists():
                        paths.append(str(p))
                        labels.append(class_to_idx[cname])
                        break
            data_by_split[s] = (paths, np.array(labels), classes)
        return data_by_split

    elif dataset_name == "sitting_posture_detection":
        split_csv = PROJECT_ROOT / "03_metadata" / "final_split" / "sitting_posture_detection.csv"
        image_dir = PROJECT_ROOT / "02_data" / "raw" / "Sitting Posture Detection.v2i.coco"
        classes = ["good_posture", "leaning_backward", "leaning_forward", "slouch"]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # Load annotation mapping
        image_classes = {}
        for orig in ["train", "valid", "test"]:
            ann_file = image_dir / orig / "_annotations.coco.json"
            if not ann_file.exists():
                continue
            with open(ann_file) as f:
                coco = json.load(f)
            imgs = {im["id"]: im["file_name"] for im in coco["images"]}
            cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
            for ann in coco["annotations"]:
                fname = imgs.get(ann["image_id"], "")
                cname = cats.get(ann["category_id"], "")
                if fname and cname in class_to_idx:
                    image_classes[fname] = cname

        df_split = pd.read_csv(split_csv)
        data_by_split = {}

        for s in ["train", "valid", "test"]:
            sub = df_split[df_split["split"] == s]
            paths, labels = [], []
            for _, r in sub.iterrows():
                fname = r["image_id"]
                cname = image_classes.get(fname)
                if not cname:
                    continue
                for orig in ["train", "valid", "test"]:
                    p = image_dir / orig / fname
                    if p.exists():
                        paths.append(str(p))
                        labels.append(class_to_idx[cname])
                        break
            data_by_split[s] = (paths, np.array(labels), classes)
        return data_by_split

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def run_exp_03(dataset_name, yolo_model="yolov8n-pose.pt"):
    exp_id = f"EXP-YOLO-POSE-{dataset_name.upper()}"
    exp_dir = RESULTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUNNING {exp_id} ({yolo_model})")
    print(f"{'='*60}")

    data_by_split = load_dataset_paths(dataset_name)
    classes = data_by_split["train"][2]

    # 1. Pose Feature Extraction for all splits
    features_by_split = {}
    for s in ["train", "valid", "test"]:
        paths, labels, _ = data_by_split[s]
        print(f"\nProcessing {s} split ({len(paths)} images)...")
        if len(paths) == 0:
            print(f"Warning: split {s} is empty!")
            continue

        df_kps = extract_pose_keypoints(paths, model_name=yolo_model)
        df_feats = engineer_yolo_pose_features(df_kps)
        X = df_feats.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        features_by_split[s] = (X, labels)

    X_train, y_train = features_by_split["train"]
    X_valid, y_valid = features_by_split["valid"]
    X_test, y_test = features_by_split["test"]

    print(f"\nFeature matrix shapes -> Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_test_s = scaler.transform(X_test)

    # 2. Train Classifiers
    results = []

    # Model A: MLP
    print("\n--- Training MLP Classifier ---")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True,
                        validation_fraction=0.15, random_state=42, learning_rate_init=0.001)
    mlp.fit(X_train_s, y_train)
    y_pred_mlp = mlp.predict(X_test_s)

    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)
    print(f"MLP Test Accuracy: {acc_mlp:.4f} | F1 Macro: {f1_mlp:.4f}")
    results.append({"model": "YOLO-Pose + MLP", "accuracy": round(acc_mlp, 4), "f1_macro": round(f1_mlp, 4)})

    # Model B: XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        print("\n--- Training XGBoost Classifier ---")
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        xgb.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)], verbose=False)
        y_pred_xgb = xgb.predict(X_test_s)

        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)
        print(f"XGBoost Test Accuracy: {acc_xgb:.4f} | F1 Macro: {f1_xgb:.4f}")
        results.append({"model": "YOLO-Pose + XGBoost", "accuracy": round(acc_xgb, 4), "f1_macro": round(f1_xgb, 4)})
    except ImportError:
        pass

    # Save summary
    out_file = exp_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")
    return results


def main():
    print("=== EXP-03: YOLO Pose + Classifier ===")
    for dname in ["project_design", "sitting_posture_detection"]:
        try:
            run_exp_03(dname)
        except Exception as e:
            print(f"Error running {dname}: {e}")


if __name__ == "__main__":
    main()
