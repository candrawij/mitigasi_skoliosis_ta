"""
T2C: Preprocessing keypoint IKORN 4-Keypoint.

Reads interim keypoint CSV, performs:
1. Bbox-relative normalization
2. Feature engineering (angles, distances)
3. Saves preprocessed feature vectors for classifier training

Keypoint mapping: 0: bottom, 1: shoulder, 2: head, 3: back
"""
import csv
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_CSV = Path(r"D:\.Candra\Project\TA\02_data\interim\ikorn_4kp_keypoints\ikorn_4kp_keypoints.csv")
SPLIT_CSV = Path(r"D:\.Candra\Project\TA\03_metadata\final_split\ikorn_4kp.csv")
OUTPUT_DIR = Path(r"D:\.Candra\Project\TA\02_data\processed\ikorn_4kp")

KP_NAMES = ["bottom", "shoulder", "head", "back"]


def compute_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def compute_distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def extract_features(row):
    """Extract pose features from normalized keypoints."""
    features = {}

    pts = {}
    for kp in KP_NAMES:
        pts[kp] = (row[f"{kp}_x"], row[f"{kp}_y"])

    # Normalize relative to bbox
    bx, by, bw, bh = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
    if bw > 0 and bh > 0:
        bbox_cx = bx + bw/2
        bbox_cy = by + bh/2
        for kp in KP_NAMES:
            features[f"{kp}_nx"] = (pts[kp][0] - bbox_cx) / bw
            features[f"{kp}_ny"] = (pts[kp][1] - bbox_cy) / bh
    else:
        for kp in KP_NAMES:
            features[f"{kp}_nx"] = pts[kp][0]
            features[f"{kp}_ny"] = pts[kp][1]

    # === Spine angles ===
    # Angle from bottom to shoulder (spine lower)
    spine_lower_angle = compute_angle(pts["bottom"], pts["shoulder"])
    features["spine_lower_angle"] = spine_lower_angle

    # Angle from shoulder to head (spine upper)
    spine_upper_angle = compute_angle(pts["shoulder"], pts["head"])
    features["spine_upper_angle"] = spine_upper_angle

    # Angle from bottom to head (overall spine)
    spine_full_angle = compute_angle(pts["bottom"], pts["head"])
    features["spine_full_angle"] = spine_full_angle

    # Angle from shoulder to back
    back_angle = compute_angle(pts["shoulder"], pts["back"])
    features["back_angle"] = back_angle

    # === Distances ===
    features["dist_bottom_shoulder"] = compute_distance(pts["bottom"], pts["shoulder"])
    features["dist_shoulder_head"] = compute_distance(pts["shoulder"], pts["head"])
    features["dist_bottom_head"] = compute_distance(pts["bottom"], pts["head"])
    features["dist_shoulder_back"] = compute_distance(pts["shoulder"], pts["back"])
    features["dist_bottom_back"] = compute_distance(pts["bottom"], pts["back"])

    # === Relative positions ===
    features["head_shoulder_dx"] = pts["head"][0] - pts["shoulder"][0]
    features["head_shoulder_dy"] = pts["head"][1] - pts["shoulder"][1]
    features["back_shoulder_dx"] = pts["back"][0] - pts["shoulder"][0]
    features["back_shoulder_dy"] = pts["back"][1] - pts["shoulder"][1]

    # === Curvature (angle difference between upper and lower spine) ===
    features["spine_curvature"] = spine_upper_angle - spine_lower_angle

    # === Forward lean indicator ===
    features["head_forward_offset"] = pts["head"][0] - pts["bottom"][0]

    # === Raw normalized coordinates ===
    for kp in KP_NAMES:
        features[f"raw_{kp}_x"] = pts[kp][0]
        features[f"raw_{kp}_y"] = pts[kp][1]

    return features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} records")

    # Use the new split from split manifest
    split_df = pd.read_csv(SPLIT_CSV)
    split_map = dict(zip(split_df["image_id"], split_df["split"]))
    df["split"] = df["image_id"].map(split_map).fillna(df.get("split", "train"))

    # Extract features
    feature_records = []
    for _, row in df.iterrows():
        feats = extract_features(row)
        record = {
            "image_id": row["image_id"],
            "split": row["split"],
            "class_name": row["class_name"],
            "class_id": 0 if row["class_name"] == "Bad" else 1,
        }
        record.update(feats)
        feature_records.append(record)

    feat_df = pd.DataFrame(feature_records)
    print(f"Features extracted: {len(feat_df)} records, {len(feat_df.columns)} columns")

    # Split into train/valid/test
    meta_cols = ["image_id", "split", "class_id", "class_name"]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    for split_name in ["train", "valid", "test"]:
        split_data = feat_df[feat_df["split"] == split_name]
        X = split_data[feature_cols].values
        y = split_data["class_id"].values

        np.save(OUTPUT_DIR / f"X_{split_name}.npy", X.astype(np.float32))
        np.save(OUTPUT_DIR / f"y_{split_name}.npy", y.astype(np.int64))
        print(f"  {split_name}: X={X.shape}, y={y.shape}")

    feat_df.to_csv(OUTPUT_DIR / "features.csv", index=False)

    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(OUTPUT_DIR / "class_map.json", "w") as f:
        json.dump({"0": "Bad", "1": "Good"}, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")


if __name__ == "__main__":
    main()
