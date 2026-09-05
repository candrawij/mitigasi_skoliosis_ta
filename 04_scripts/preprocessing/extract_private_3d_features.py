"""
extract_private_3d_features.py — Extract 25 Normalized Stereo 3D Spatial Geometry Features
for Private Dataset 6-Class Postures.

Pipeline:
  1. Load private_6class_all.csv (727 captures)
  2. Filter by QC decision: only INCLUDE_3D_FULL and INCLUDE_3D_WITH_MASKING are usable
  3. Load 3D keypoints from 02_data/private_annotations/keypoints_3d/<capture_id>_3d_keypoints.json
  4. Perform hip-centered 3D normalization and scale by S3
  5. Extract 15 normalized 3D coordinates and 10 spatial geometry features (Total 25 features)
  6. Perform data integrity audit and check stop conditions
  7. Save private_features_3d.csv and feature_3d_audit.csv
"""

import os
import sys
import json
from typing import Tuple, Optional, Dict, Any
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
    FEATURE_NAMES_3D,
    extract_3d_features
)

MANIFESTS_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "manifests"
FEATURES_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
AUDIT_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "audit"
ANNOT_3D_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "keypoints_3d"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def load_3d_annotation(capture_id: str) -> Tuple[Optional[np.ndarray], int, bool, str]:
    """Load 3D keypoints in meters from keypoints_3d/<capture_id>_3d_keypoints.json."""
    fpath = ANNOT_3D_DIR / f"{capture_id}_3d_keypoints.json"
    if not fpath.exists():
        return None, 0, False, f"3D keypoint file not found: {fpath.name}"

    try:
        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as e:
        return None, 0, False, f"Failed to parse JSON: {e}"

    raw_kpts = data.get("keypoints_3d_m", [])
    valid_core = data.get("valid_core_joints", 0)

    if not raw_kpts or len(raw_kpts) < 17:
        return None, 0, False, f"Invalid 3D keypoints length ({len(raw_kpts)}) in {fpath.name}"

    # Convert to float64 numpy array, preserving NaNs
    kpts_3d = np.array(raw_kpts, dtype=np.float64)
    if kpts_3d.shape != (17, 3):
        return None, 0, False, f"Invalid shape {kpts_3d.shape} for 3D keypoints"

    return kpts_3d, valid_core, True, "Success"


def run_extraction():
    print("=" * 80)
    print("  STEP 7: EXTRACT 25 STEREO 3D SPATIAL GEOMETRY FEATURES (727 CAPTURES)")
    print("=" * 80)

    manifest_file = MANIFESTS_DIR / "private_6class_all.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}. Run build_private_6class_manifest.py first!")

    df_manifest = pd.read_csv(manifest_file)
    print(f"Loaded manifest: {len(df_manifest)} captures")
    assert len(df_manifest) == 727, f"Manifest must contain exactly 727 captures, found {len(df_manifest)}"

    # Load QC 3D status from private_3d_qc_final.csv
    qc_3d_file = RESULTS_DIR / "private_3d_qc_final.csv"
    qc3d_lookup = {}
    if qc_3d_file.exists():
        df_qc3d = pd.read_csv(qc_3d_file)
        for _, r in df_qc3d.iterrows():
            qc3d_lookup[r["capture_id"]] = {
                "decision": r.get("decision", "UNKNOWN"),
                "core_reproj_error": r.get("core_reproj_error_640p_px", None),
                "reason": r.get("reason", "")
            }
        print(f"Loaded 3D QC database: {len(qc3d_lookup)} captures")
    else:
        print("Warning: private_3d_qc_final.csv not found, using manifest qc_3d_status.")

    feature_rows = []
    audit_records = []

    usable_count = 0
    excluded_count = 0

    for idx, row in df_manifest.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        sess_id = row["session_id"]
        label = row["label"]
        class_id = int(row["class_id"])
        cal_id = row["calibration_id"]
        lat_side = row["lateral_side"]
        subset = row["subset"]

        qc_info = qc3d_lookup.get(cap_id, {
            "decision": row.get("qc_3d_status", "UNKNOWN"),
            "core_reproj_error": row.get("reprojection_error", None),
            "reason": ""
        })
        qc_decision = qc_info["decision"]
        reproj_err = qc_info["core_reproj_error"]

        feat_row = {
            "capture_id": cap_id,
            "subject_id": sub_id,
            "session_id": sess_id,
            "label": label,
            "class_id": class_id,
            "calibration_id": cal_id,
            "lateral_side": lat_side,
            "subset": subset,
            "qc_3d_status": qc_decision,
            "core_reproj_error": reproj_err,
            "status_3d": "USABLE"
        }

        audit_rec = {
            "capture_id": cap_id,
            "subject_id": sub_id,
            "label": label,
            "class_id": class_id,
            "calibration_id": cal_id,
            "qc_3d_status": qc_decision,
            "core_reproj_error": reproj_err,
            "valid_core_joints": 0,
            "status_3d": "USABLE",
            "failure_detail": None
        }

        # 1. Rule-based check: Must be INCLUDE_3D_FULL or INCLUDE_3D_WITH_MASKING
        if qc_decision not in ["INCLUDE_3D_FULL", "INCLUDE_3D_WITH_MASKING"]:
            feat_row["status_3d"] = "EXCLUDED"
            audit_rec["status_3d"] = "EXCLUDED"
            audit_rec["failure_detail"] = f"QC Decision: {qc_decision} ({qc_info.get('reason', '')})"
            for f_name in FEATURE_NAMES_3D:
                feat_row[f_name] = np.nan
            excluded_count += 1
            feature_rows.append(feat_row)
            audit_records.append(audit_rec)
            continue

        # 2. Load 3D Annotation JSON
        kpts3d, valid_core, ok_annot, msg_annot = load_3d_annotation(cap_id)
        audit_rec["valid_core_joints"] = valid_core

        if not ok_annot:
            feat_row["status_3d"] = "INVALID_ANNOTATION"
            audit_rec["status_3d"] = "INVALID_ANNOTATION"
            audit_rec["failure_detail"] = msg_annot
            for f_name in FEATURE_NAMES_3D:
                feat_row[f_name] = np.nan
            excluded_count += 1
            feature_rows.append(feat_row)
            audit_records.append(audit_rec)
            continue

        # 3. Extract 25 3D spatial features
        feat_3d, ok_feat, msg_feat = extract_3d_features(kpts3d)

        if not ok_feat:
            feat_row["status_3d"] = "INVALID_3D_GEOMETRY"
            audit_rec["status_3d"] = "INVALID_3D_GEOMETRY"
            audit_rec["failure_detail"] = msg_feat
            for f_name in FEATURE_NAMES_3D:
                feat_row[f_name] = np.nan
            excluded_count += 1
        else:
            feat_row.update(feat_3d)
            usable_count += 1

        feature_rows.append(feat_row)
        audit_records.append(audit_rec)

    df_features_3d = pd.DataFrame(feature_rows)
    df_audit_3d = pd.DataFrame(audit_records)

    # Reorder columns: metadata first, then 25 features
    meta_cols = ["capture_id", "subject_id", "session_id", "label", "class_id",
                 "calibration_id", "lateral_side", "subset", "qc_3d_status",
                 "core_reproj_error", "status_3d"]
    df_features_3d = df_features_3d[meta_cols + FEATURE_NAMES_3D]

    # ==============================================================================
    # ACCEPTANCE CHECKS & AUDIT SUMMARY
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  AUDIT 3D FEATURES QUALITY & INTEGRITY")
    print("=" * 80)

    print(f"Total 6-Class Captures Processed: {len(df_features_3d)}")
    print(f"  - 3D Usable (Valid):   {usable_count} ({usable_count/len(df_features_3d)*100:.2f}%)")
    print(f"  - 3D Excluded/Invalid: {excluded_count} ({excluded_count/len(df_features_3d)*100:.2f}%)")

    # Check 1: No forward_head or reject
    assert "forward_head" not in df_features_3d["label"].values, "Violation: forward_head present in 3D!"
    assert "reject" not in df_features_3d["label"].values, "Violation: reject present in 3D!"
    print("  ✓ Acceptance Passed: Zero forward_head and Zero reject in 3D features")

    # Check 2: class_id strictly in 0..5
    assert set(df_features_3d["class_id"].unique()).issubset({0, 1, 2, 3, 4, 5}), "Violation: class_id outside 0..5!"
    print("  ✓ Acceptance Passed: class_id values strictly in [0..5]")

    # Check 3: Zero duplicate capture_id
    assert not df_features_3d["capture_id"].duplicated().any(), "Violation: duplicate capture_id found in 3D!"
    print("  ✓ Acceptance Passed: Zero duplicate capture_id")

    # Check 4: No inf or -inf in numeric feature values
    feat_matrix = df_features_3d[FEATURE_NAMES_3D].values
    assert not np.isinf(feat_matrix).any(), "Violation: inf or -inf found in 3D feature values!"
    print("  ✓ Acceptance Passed: Zero inf/-inf values across all 25 3D features")

    # Check 5: Class distribution of usable samples
    df_usable_3d = df_features_3d[df_features_3d["status_3d"] == "USABLE"]
    print("\nUsable Sample Count per Posture Class (3D):")
    for c_name in MAIN_CLASSES:
        cnt = (df_usable_3d["label"] == c_name).sum()
        total_raw = (df_features_3d["label"] == c_name).sum()
        print(f"  - {c_name:18s}: {cnt:4d} / {total_raw:4d} usable ({cnt/total_raw*100:.1f}%)")

    # Check 6: Usable Sample Count per Subject
    print("\nUsable Sample Count per Subject (3D):")
    sub_counts = df_usable_3d["subject_id"].value_counts().sort_index()
    for s_id, cnt in sub_counts.items():
        total_s = (df_features_3d["subject_id"] == s_id).sum()
        print(f"  - {s_id}: {cnt:2d} / {total_s:2d} usable")

    # Check 7: Feature Missing Rates on Usable Subset
    print("\nFeature Missing Rates (NaN % on Usable Subset):")
    nan_rates = df_usable_3d[FEATURE_NAMES_3D].isna().mean() * 100
    nan_features = nan_rates[nan_rates > 0]
    if len(nan_features) == 0:
        print("  ✓ All 25 3D features have 0.0% NaN on usable samples!")
    else:
        for f_name, rate in nan_features.items():
            print(f"  - {f_name:32s}: {rate:5.2f}% NaN (acceptable for masked/occluded nose)")

    # Save outputs
    out_features_csv = FEATURES_DIR / "private_features_3d.csv"
    out_audit_csv = AUDIT_DIR / "feature_3d_audit.csv"

    df_features_3d.to_csv(out_features_csv, index=False)
    df_audit_3d.to_csv(out_audit_csv, index=False)

    print(f"\n[SAVED] 3D Feature Table saved to: {out_features_csv}")
    print(f"[SAVED] 3D Audit Log saved to:    {out_audit_csv}")

    return df_features_3d, df_audit_3d


if __name__ == "__main__":
    run_extraction()
