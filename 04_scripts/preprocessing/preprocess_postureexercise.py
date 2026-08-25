"""
T2B: Preprocessing keypoint Postureexercise.

Reads the interim keypoint CSV, performs:
1. Bbox-relative normalization (keypoints relative to bounding box)
2. Feature engineering (angles, distances, symmetry ratios)
3. Saves preprocessed feature vectors for classifier training

Keypoint mapping (confirmed):
  0: left_shoulder  1: right_shoulder  2: left_eye
  3: left_ear       4: nose            5: right_ear  6: right_eye
"""
import csv
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "02_data" / "interim" / "postureexercise_keypoints" / "postureexercise_keypoints.csv"
SPLIT_CSV = PROJECT_ROOT / "03_metadata" / "final_split" / "postureexercise.csv"
OUTPUT_DIR = PROJECT_ROOT / "02_data" / "processed" / "postureexercise"

KP_NAMES = ["left_shoulder", "right_shoulder", "left_eye", "left_ear", "nose", "right_ear", "right_eye"]


def normalize_to_bbox(row):
    """Normalize keypoints relative to bounding box center and size."""
    cx, cy = row["bbox_cx"], row["bbox_cy"]
    bw, bh = row["bbox_w"], row["bbox_h"]
    if bw == 0 or bh == 0:
        return None

    normalized = {}
    for kp in KP_NAMES:
        kx = (row[f"{kp}_x"] - cx) / bw
        ky = (row[f"{kp}_y"] - cy) / bh
        normalized[f"{kp}_nx"] = kx
        normalized[f"{kp}_ny"] = ky
    return normalized


def compute_angle(p1, p2):
    """Compute angle in degrees from p1 to p2 relative to horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def compute_distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def extract_features(row):
    """Extract pose features from normalized keypoints."""
    features = {}

    # Get normalized points
    pts = {}
    for kp in KP_NAMES:
        pts[kp] = (row[f"{kp}_nx"], row[f"{kp}_ny"])

    # === Shoulder features ===
    shoulder_mid_x = (pts["left_shoulder"][0] + pts["right_shoulder"][0]) / 2
    shoulder_mid_y = (pts["left_shoulder"][1] + pts["right_shoulder"][1]) / 2
    shoulder_width = compute_distance(pts["left_shoulder"], pts["right_shoulder"])
    shoulder_angle = compute_angle(pts["left_shoulder"], pts["right_shoulder"])

    features["shoulder_mid_x"] = shoulder_mid_x
    features["shoulder_mid_y"] = shoulder_mid_y
    features["shoulder_width"] = shoulder_width
    features["shoulder_angle"] = shoulder_angle

    # === Eye features ===
    eye_mid_x = (pts["left_eye"][0] + pts["right_eye"][0]) / 2
    eye_mid_y = (pts["left_eye"][1] + pts["right_eye"][1]) / 2
    eye_width = compute_distance(pts["left_eye"], pts["right_eye"])
    eye_angle = compute_angle(pts["left_eye"], pts["right_eye"])

    features["eye_mid_x"] = eye_mid_x
    features["eye_mid_y"] = eye_mid_y
    features["eye_width"] = eye_width
    features["eye_angle"] = eye_angle

    # === Ear features ===
    ear_mid_x = (pts["left_ear"][0] + pts["right_ear"][0]) / 2
    ear_mid_y = (pts["left_ear"][1] + pts["right_ear"][1]) / 2
    ear_width = compute_distance(pts["left_ear"], pts["right_ear"])
    ear_angle = compute_angle(pts["left_ear"], pts["right_ear"])

    features["ear_mid_x"] = ear_mid_x
    features["ear_mid_y"] = ear_mid_y
    features["ear_width"] = ear_width
    features["ear_angle"] = ear_angle

    # === Nose relative to midpoints ===
    features["nose_x"] = pts["nose"][0]
    features["nose_y"] = pts["nose"][1]
    features["nose_to_shoulder_mid_dx"] = pts["nose"][0] - shoulder_mid_x
    features["nose_to_shoulder_mid_dy"] = pts["nose"][1] - shoulder_mid_y
    features["nose_to_eye_mid_dy"] = pts["nose"][1] - eye_mid_y

    # === Head tilt (angle from nose to ear midpoint) ===
    head_tilt = compute_angle((ear_mid_x, ear_mid_y), pts["nose"])
    features["head_tilt"] = head_tilt

    # === Symmetry ratios ===
    # Left vs right shoulder height difference
    features["shoulder_dy"] = pts["right_shoulder"][1] - pts["left_shoulder"][1]
    # Left vs right eye height difference
    features["eye_dy"] = pts["right_eye"][1] - pts["left_eye"][1]
    # Left vs right ear height difference
    features["ear_dy"] = pts["right_ear"][1] - pts["left_ear"][1]

    # === Lateral offset (nose relative to shoulder center) ===
    features["lateral_offset"] = pts["nose"][0] - shoulder_mid_x

    # === Vertical alignment (nose-shoulder vertical distance) ===
    features["vertical_alignment"] = shoulder_mid_y - pts["nose"][1]

    # === All raw normalized coordinates ===
    for kp in KP_NAMES:
        features[f"raw_{kp}_nx"] = pts[kp][0]
        features[f"raw_{kp}_ny"] = pts[kp][1]

    return features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(INPUT_CSV)
    split_df = pd.read_csv(SPLIT_CSV)

    print(f"Loaded {len(df)} records")

    # Normalize to bbox
    norm_records = []
    skipped = 0
    for _, row in df.iterrows():
        norm = normalize_to_bbox(row)
        if norm is None:
            skipped += 1
            continue
        record = {"image_id": row["image_id"], "split": row["split"],
                  "class_id": row["class_id"], "class_name": row["class_name"]}
        record.update(norm)
        norm_records.append(record)

    print(f"Normalized: {len(norm_records)}, skipped: {skipped}")

    norm_df = pd.DataFrame(norm_records)

    # Extract features
    feature_records = []
    for _, row in norm_df.iterrows():
        feats = extract_features(row)
        record = {"image_id": row["image_id"], "split": row["split"],
                  "class_id": row["class_id"], "class_name": row["class_name"]}
        record.update(feats)
        feature_records.append(record)

    feat_df = pd.DataFrame(feature_records)
    print(f"Features extracted: {len(feat_df)} records, {len(feat_df.columns)} columns")

    # Split into train/valid/test
    for split_name in ["train", "valid", "test"]:
        split_data = feat_df[feat_df["split"] == split_name]
        # Feature columns (exclude metadata)
        meta_cols = ["image_id", "split", "class_id", "class_name"]
        feature_cols = [c for c in feat_df.columns if c not in meta_cols]

        X = split_data[feature_cols].values
        y = split_data["class_id"].values
        labels = split_data["class_name"].values

        np.save(OUTPUT_DIR / f"X_{split_name}.npy", X.astype(np.float32))
        np.save(OUTPUT_DIR / f"y_{split_name}.npy", y.astype(np.int64))
        print(f"  {split_name}: X={X.shape}, y={y.shape}")

    # Save full CSV
    feat_df.to_csv(OUTPUT_DIR / "features.csv", index=False)

    # Save feature names
    meta_cols = ["image_id", "split", "class_id", "class_name"]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save class mapping
    class_map = df.drop_duplicates("class_id").set_index("class_id")["class_name"].to_dict()
    with open(OUTPUT_DIR / "class_map.json", "w") as f:
        json.dump({str(k): v for k, v in sorted(class_map.items())}, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")


if __name__ == "__main__":
    main()
