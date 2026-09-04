"""
extract_private_2d_features.py — Extract 36 Normalized 2D Multi-View Features (CAM01 + CAM02)
for Private Dataset 6-Class Postures.

Pipeline:
  1. Load private_6class_all.csv (727 captures)
  2. Load 2D keypoints and confidences from selected_person annotations
  3. Apply CAM02 lateral canonicalization (flip horizontal coords if lateral_side == 'left')
  4. Perform hip-centered pose normalization and scale by S
  5. Extract 18 geometric features for CAM01 and 18 for CAM02 (Total 36 features)
  6. Perform data integrity audit and check stop conditions
  7. Save private_features_2d.csv and feature_2d_audit.csv
"""

import os
import sys
import json
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
from private_feature_common import (
    CLASS_TO_ID,
    MAIN_CLASSES,
    FEATURE_NAMES_2D,
    extract_single_view_2d_features,
    combine_2d_multi_view_features
)

MANIFESTS_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "manifests"
FEATURES_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
AUDIT_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "audit"
SELECTED_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "selected_person"
KEYPOINTS_2D_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "keypoints_2d"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def load_person_annotation(image_path_str: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, str]:
    """Load keypoints (17,2) and confidences (17,) from selected_person or fallback to keypoints_2d."""
    img_stem = Path(image_path_str).stem
    f_sel = SELECTED_DIR / f"{img_stem}_selected_person.json"
    f_k2d = KEYPOINTS_2D_DIR / f"{img_stem}_keypoints.json"

    data = None
    if f_sel.exists():
        with open(f_sel, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    elif f_k2d.exists():
        with open(f_k2d, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    else:
        return None, None, False, f"Annotation file missing for {img_stem}"

    has_target = data.get("has_target", data.get("has_pose", False))
    if not has_target:
        return None, None, False, f"No valid target person in {img_stem}"

    kpts = np.array(data.get("keypoints", []), dtype=np.float64)
    confs = np.array(data.get("confidences", []), dtype=np.float64)

    if kpts.shape != (17, 2) or confs.shape != (17,):
        return None, None, False, f"Invalid shape for {img_stem}: kpts={kpts.shape}, confs={confs.shape}"

    return kpts, confs, True, "Success"


def run_extraction():
    print("=" * 80)
    print("  STEP 5: EXTRACT 36 2D MULTI-VIEW FEATURES (727 CAPTURES)")
    print("=" * 80)

    manifest_file = MANIFESTS_DIR / "private_6class_all.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}. Run build_private_6class_manifest.py first!")

    df_manifest = pd.read_csv(manifest_file)
    print(f"Loaded manifest: {len(df_manifest)} captures")
    assert len(df_manifest) == 727, f"Manifest must contain exactly 727 captures, found {len(df_manifest)}"

    feature_rows = []
    audit_records = []

    valid_count = 0
    invalid_count = 0

    for idx, row in df_manifest.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        sess_id = row["session_id"]
        label = row["label"]
        class_id = int(row["class_id"])
        cal_id = row["calibration_id"]
        lat_side = row["lateral_side"]
        subset = row["subset"]

        cam01_path = row["cam01_path"]
        cam02_path = row["cam02_path"]

        # 1. Load annotations for CAM01
        kpts1, confs1, ok1, reason1 = load_person_annotation(cam01_path)
        # 2. Load annotations for CAM02
        kpts2, confs2, ok2, reason2 = load_person_annotation(cam02_path)

        feat_row = {
            "capture_id": cap_id,
            "subject_id": sub_id,
            "session_id": sess_id,
            "label": label,
            "class_id": class_id,
            "calibration_id": cal_id,
            "lateral_side": lat_side,
            "subset": subset,
            "status_2d": "USABLE"
        }

        audit_rec = {
            "capture_id": cap_id,
            "subject_id": sub_id,
            "label": label,
            "class_id": class_id,
            "cam01_valid": ok1,
            "cam02_valid": ok2,
            "cam01_reason": reason1,
            "cam02_reason": reason2,
            "status_2d": "USABLE",
            "failure_detail": None
        }

        if not ok1 or not ok2:
            feat_row["status_2d"] = "INVALID_ANNOTATION"
            audit_rec["status_2d"] = "INVALID_ANNOTATION"
            audit_rec["failure_detail"] = f"CAM01: {reason1} | CAM02: {reason2}"
            for f_name in FEATURE_NAMES_2D:
                feat_row[f_name] = np.nan
            invalid_count += 1
            feature_rows.append(feat_row)
            audit_records.append(audit_rec)
            continue

        # 3. Extract CAM01 features (Frontal view)
        c1_feat, c1_ok, c1_msg = extract_single_view_2d_features(
            kpts1, confs1, view_role="frontal", lateral_side=lat_side
        )

        # 4. Extract CAM02 features (Lateral view with canonicalization)
        c2_feat, c2_ok, c2_msg = extract_single_view_2d_features(
            kpts2, confs2, view_role="lateral", lateral_side=lat_side
        )

        if not c1_ok or not c2_ok:
            feat_row["status_2d"] = "INVALID_CORE_KEYPOINTS"
            audit_rec["status_2d"] = "INVALID_CORE_KEYPOINTS"
            audit_rec["failure_detail"] = f"CAM01 geom: {c1_msg} | CAM02 geom: {c2_msg}"
            for f_name in FEATURE_NAMES_2D:
                feat_row[f_name] = np.nan
            invalid_count += 1
        else:
            # 5. Combine into 36 features
            combined_36 = combine_2d_multi_view_features(c1_feat, c2_feat)
            feat_row.update(combined_36)
            valid_count += 1

        feature_rows.append(feat_row)
        audit_records.append(audit_rec)

    df_features_2d = pd.DataFrame(feature_rows)
    df_audit_2d = pd.DataFrame(audit_records)

    # Reorder columns: metadata first, then 36 features in fixed schema
    meta_cols = ["capture_id", "subject_id", "session_id", "label", "class_id",
                 "calibration_id", "lateral_side", "subset", "status_2d"]
    df_features_2d = df_features_2d[meta_cols + FEATURE_NAMES_2D]

    # ==============================================================================
    # ACCEPTANCE CHECKS & AUDIT SUMMARY
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  STEP 6: AUDIT 2D FEATURES QUALITY & SANITY")
    print("=" * 80)

    print(f"Total 6-Class Captures Processed: {len(df_features_2d)}")
    print(f"  - 2D Usable (Valid):   {valid_count} ({valid_count/len(df_features_2d)*100:.2f}%)")
    print(f"  - 2D Invalid:          {invalid_count} ({invalid_count/len(df_features_2d)*100:.2f}%)")

    # Check 1: No forward_head or reject
    assert "forward_head" not in df_features_2d["label"].values, "Violation: forward_head present!"
    assert "reject" not in df_features_2d["label"].values, "Violation: reject present!"
    print("  ✓ Acceptance Passed: Zero forward_head and Zero reject in features")

    # Check 2: class_id strictly 0..5
    assert set(df_features_2d["class_id"].unique()).issubset({0, 1, 2, 3, 4, 5}), "Violation: class_id outside 0..5!"
    print("  ✓ Acceptance Passed: class_id values strictly in [0..5]")

    # Check 3: Zero duplicate capture_id
    assert not df_features_2d["capture_id"].duplicated().any(), "Violation: duplicate capture_id found!"
    print("  ✓ Acceptance Passed: Zero duplicate capture_id")

    # Check 4: No inf or -inf in numeric feature columns
    feat_matrix = df_features_2d[FEATURE_NAMES_2D].values
    assert not np.isinf(feat_matrix).any(), "Violation: inf or -inf found in feature values!"
    print("  ✓ Acceptance Passed: Zero inf/-inf values across all 36 features")

    # Check 5: Class distribution of usable samples
    df_usable = df_features_2d[df_features_2d["status_2d"] == "USABLE"]
    print("\nUsable Sample Count per Posture Class:")
    for c_name in MAIN_CLASSES:
        cnt = (df_usable["label"] == c_name).sum()
        total_raw = (df_features_2d["label"] == c_name).sum()
        print(f"  - {c_name:18s}: {cnt:4d} / {total_raw:4d} usable ({cnt/total_raw*100:.1f}%)")

    # Check 6: NaN rates per feature
    print("\nFeature Missing Rates (NaN % on Usable Subset):")
    nan_rates = df_usable[FEATURE_NAMES_2D].isna().mean() * 100
    nan_features = nan_rates[nan_rates > 0]
    if len(nan_features) == 0:
        print("  ✓ All 36 features have 0.0% NaN on usable samples!")
    else:
        for f_name, rate in nan_features.items():
            print(f"  - {f_name:32s}: {rate:5.2f}% NaN (acceptable if nose occluded in lateral view)")

    # Save outputs
    out_features_csv = FEATURES_DIR / "private_features_2d.csv"
    out_audit_csv = AUDIT_DIR / "feature_2d_audit.csv"

    df_features_2d.to_csv(out_features_csv, index=False)
    df_audit_2d.to_csv(out_audit_csv, index=False)

    print(f"\n[SAVED] 2D Feature Table saved to: {out_features_csv}")
    print(f"[SAVED] 2D Audit Log saved to:    {out_audit_csv}")

    return df_features_2d, df_audit_2d


if __name__ == "__main__":
    run_extraction()
