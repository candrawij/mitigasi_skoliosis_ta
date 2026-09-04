"""
build_private_6class_manifest.py — Build 6-Class Private Dataset Manifest
Filters raw 885 captures to 727 captures belonging to the 6 primary posture classes.

Excludes:
  - forward_head (class_removed_after_supervisor_review)
  - reject (negative/invalid-input gate)

Output:
  02_data/private_processed/manifests/private_6class_all.csv
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
from private_feature_common import CLASS_TO_ID, MAIN_CLASSES

META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"
SELECTED_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "selected_person"
MANIFESTS_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "manifests"
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


def build_manifest():
    print("=" * 80)
    print("  STEP 2: BUILD PRIVATE 6-CLASS MANIFEST (TARGET: 727 CAPTURES)")
    print("=" * 80)

    captures_csv = META_DIR / "captures.csv"
    images_csv = META_DIR / "images.csv"
    calib_map_csv = META_DIR / "calibration_map.csv"
    qc_3d_csv = RESULTS_DIR / "private_3d_qc_final.csv"

    if not captures_csv.exists() or not images_csv.exists():
        raise FileNotFoundError(f"Metadata files missing in {META_DIR}")

    df_cap = pd.read_csv(captures_csv)
    df_img = pd.read_csv(images_csv)
    df_qc3d = pd.read_csv(qc_3d_csv) if qc_3d_csv.exists() else None

    print(f"Loaded raw captures: {len(df_cap)} rows across {df_cap['subject_id'].nunique()} subjects")
    print(f"Loaded raw images: {len(df_img)} rows")

    # Map QC 3D status if available
    qc3d_lookup = {}
    if df_qc3d is not None:
        for _, r in df_qc3d.iterrows():
            qc3d_lookup[r["capture_id"]] = {
                "decision": r.get("decision", "UNKNOWN"),
                "status_3d": r.get("status_3d", "UNKNOWN"),
                "core_reproj_error": r.get("core_reproj_error_640p_px", None),
                "correspondence_status": r.get("correspondence_status", "UNKNOWN")
            }

    # Group images by capture_id
    img_pairs = {}
    for _, r in df_img.iterrows():
        cid = r["capture_id"]
        cam = r["camera_id"]
        if cid not in img_pairs:
            img_pairs[cid] = {}
        img_pairs[cid][cam] = r

    manifest_rows = []

    for idx, row in df_cap.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        posture = row["primary_posture"]
        sess_id = row.get("session_id", "SE01")
        cal_id = row.get("calibration_id", "")
        lateral_side = row.get("lateral_side", "right")
        subset = row.get("subset", "controlled")

        # Exclude forward_head and reject
        if posture not in MAIN_CLASSES:
            continue

        if cap_id not in img_pairs:
            raise ValueError(f"Capture {cap_id} has no image entries in images.csv")

        pair = img_pairs[cap_id]
        if "CAM01" not in pair or "CAM02" not in pair:
            raise ValueError(f"Capture {cap_id} does not have complete CAM01 and CAM02 pair!")

        row_c1 = pair["CAM01"]
        row_c2 = pair["CAM02"]

        cam01_path = str(row_c1["image_path"]).replace("\\", "/")
        cam02_path = str(row_c2["image_path"]).replace("\\", "/")

        # Check file existence
        full_c1_path = PROJECT_ROOT / cam01_path
        full_c2_path = PROJECT_ROOT / cam02_path
        if not full_c1_path.exists() or not full_c2_path.exists():
            raise FileNotFoundError(f"Missing image file for {cap_id}: {full_c1_path} or {full_c2_path}")

        # Check selected person annotations
        f_sel1 = SELECTED_DIR / f"{row_c1['image_id']}_selected_person.json"
        f_sel2 = SELECTED_DIR / f"{row_c2['image_id']}_selected_person.json"
        sel1_valid = f_sel1.exists()
        sel2_valid = f_sel2.exists()

        status_2d = "USABLE" if (sel1_valid and sel2_valid) else "INVALID_ANNOTATION"

        # 3D QC info
        qc_info = qc3d_lookup.get(cap_id, {})
        qc_3d_decision = qc_info.get("decision", "UNKNOWN")
        status_3d = "USABLE" if qc_3d_decision in ["INCLUDE_3D_FULL", "INCLUDE_3D_WITH_MASKING"] else "EXCLUDED"
        reproj_err = qc_info.get("core_reproj_error", None)

        class_id = CLASS_TO_ID[posture]

        manifest_rows.append({
            "capture_id": cap_id,
            "subject_id": sub_id,
            "session_id": sess_id,
            "label": posture,
            "class_id": class_id,
            "cam01_path": cam01_path,
            "cam02_path": cam02_path,
            "calibration_id": cal_id,
            "lateral_side": lateral_side,
            "subset": subset,
            "status_2d": status_2d,
            "status_3d": status_3d,
            "qc_3d_status": qc_3d_decision,
            "selected_person_valid_cam01": sel1_valid,
            "selected_person_valid_cam02": sel2_valid,
            "reprojection_error": reproj_err
        })

    df_manifest = pd.DataFrame(manifest_rows)

    # ==============================================================================
    # ACCEPTANCE CHECKS & ASSERTIONS
    # ==============================================================================
    print("\n[Running Acceptance Checks...]")

    # Check 1: No forward_head or reject
    invalid_labels = df_manifest[~df_manifest["label"].isin(MAIN_CLASSES)]
    if len(invalid_labels) > 0:
        raise AssertionError(f"Found invalid labels in manifest: {invalid_labels['label'].unique()}")
    print("  ✓ No forward_head or reject in labels")

    # Check 2: Class IDs are strictly 0..5
    if not set(df_manifest["class_id"].unique()).issubset({0, 1, 2, 3, 4, 5}):
        raise AssertionError(f"class_id contains values outside 0..5: {df_manifest['class_id'].unique()}")
    print("  ✓ class_id values are strictly within [0..5]")

    # Check 3: Subject ID not empty
    if df_manifest["subject_id"].isna().any() or (df_manifest["subject_id"] == "").any():
        raise AssertionError("Found empty subject_id in manifest")
    print(f"  ✓ subject_id valid across {df_manifest['subject_id'].nunique()} subjects")

    # Check 4: No duplicate capture_id
    if df_manifest["capture_id"].duplicated().any():
        dup = df_manifest[df_manifest["capture_id"].duplicated()]["capture_id"].tolist()
        raise AssertionError(f"Duplicate capture_id found: {dup}")
    print("  ✓ Zero duplicate capture_id")

    # Check 5: Total count strictly equals 727
    total_count = len(df_manifest)
    print(f"\nManifest total captures: {total_count}")
    print("Per-class distribution:")
    class_dist = df_manifest["label"].value_counts()[MAIN_CLASSES]
    for cls_name, cnt in class_dist.items():
        print(f"  - {cls_name:18s}: {cnt:4d}")

    assert total_count == 727, f"STOP CONDITION FAILED: Expected total 727 captures, but got {total_count}!"
    print("  ✓ CRITICAL ASSERTION PASSED: len(df_manifest) == 727")

    # Save manifest
    out_path = MANIFESTS_DIR / "private_6class_all.csv"
    df_manifest.to_csv(out_path, index=False)
    print(f"\n[SAVED] 6-Class manifest saved to: {out_path}")
    return df_manifest


if __name__ == "__main__":
    build_manifest()
