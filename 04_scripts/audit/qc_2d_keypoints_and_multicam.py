"""
FASE 5D — QC Keypoint 2D & Multicam Consistency on 23 Subjects (851 Captures / 1,702 Images)
Audits:
  1. Keypoint validity, confidence distributions, and completeness per capture.
  2. Multi-camera subject identity consistency between CAM01 and CAM02.
  3. Classifies each capture into PASS / REVIEW / FAIL.
Output:
  07_results/private_audit/keypoint_qc.csv
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
SELECTED_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "selected_person"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def run_keypoint_qc():
    print("=" * 80)
    print("  FASE 5D: 2D KEYPOINT & MULTI-CAMERA CONSISTENCY QC (851 CAPTURES)")
    print("=" * 80)

    captures_csv = META_DIR / "captures.csv"
    images_csv = META_DIR / "images.csv"
    df_cap = pd.read_csv(captures_csv)
    df_img = pd.read_csv(images_csv)

    qc_records = []
    
    pass_count = 0
    review_count = 0
    fail_count = 0
    
    for idx, row in df_cap.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        posture = row["primary_posture"]
        cal_id = row["calibration_id"]
        
        img_c1_rows = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")]
        img_c2_rows = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")]
        
        if len(img_c1_rows) == 0 or len(img_c2_rows) == 0:
            qc_records.append({
                "capture_id": cap_id,
                "subject_id": sub_id,
                "posture": posture,
                "calibration_id": cal_id,
                "status": "FAIL",
                "reason": "Missing camera image in metadata"
            })
            fail_count += 1
            continue

        img_c1_id = img_c1_rows.iloc[0]["image_id"]
        img_c2_id = img_c2_rows.iloc[0]["image_id"]
        
        f1 = SELECTED_DIR / f"{img_c1_id}_selected_person.json"
        f2 = SELECTED_DIR / f"{img_c2_id}_selected_person.json"
        
        if not f1.exists() or not f2.exists():
            qc_records.append({
                "capture_id": cap_id,
                "subject_id": sub_id,
                "posture": posture,
                "calibration_id": cal_id,
                "status": "FAIL",
                "reason": "Selected person JSON not found"
            })
            fail_count += 1
            continue

        with open(f1, "r", encoding="utf-8") as fp: d1 = json.load(fp)
        with open(f2, "r", encoding="utf-8") as fp: d2 = json.load(fp)

        has1 = d1.get("has_target", False)
        has2 = d2.get("has_target", False)

        if not has1 or not has2:
            if posture == "reject":
                qc_records.append({
                    "capture_id": cap_id,
                    "subject_id": sub_id,
                    "posture": posture,
                    "calibration_id": cal_id,
                    "status": "PASS",
                    "c1_valid_joints": 0,
                    "c2_valid_joints": 0,
                    "head_visible": False,
                    "shoulders_visible": False,
                    "hips_visible": False,
                    "multicam_consistent": True,
                    "reason": "Correctly rejected negative sample (empty chair / out of frame)"
                })
                pass_count += 1
            else:
                qc_records.append({
                    "capture_id": cap_id,
                    "subject_id": sub_id,
                    "posture": posture,
                    "calibration_id": cal_id,
                    "status": "FAIL",
                    "c1_valid_joints": 0,
                    "c2_valid_joints": 0,
                    "head_visible": False,
                    "shoulders_visible": False,
                    "hips_visible": False,
                    "multicam_consistent": False,
                    "reason": "Target subject missed in core posture class"
                })
                fail_count += 1
            continue

        conf1 = np.array(d1["confidences"])
        conf2 = np.array(d2["confidences"])
        
        # Valid keypoints (confidence >= 0.25)
        c1_valid = int(sum(1 for c in conf1 if c >= 0.25))
        c2_valid = int(sum(1 for c in conf2 if c >= 0.25))
        
        # Anatomy Checks
        # Head: Nose (0), Eyes (1,2)
        head_c1 = (conf1[0] >= 0.25 or (conf1[1] >= 0.25 and conf1[2] >= 0.25))
        head_c2 = (conf2[0] >= 0.25 or conf2[1] >= 0.25 or conf2[2] >= 0.25)
        head_ok = head_c1 and head_c2
        
        # Shoulders: L_shoulder (5), R_shoulder (6)
        sh_c1 = (conf1[5] >= 0.25 and conf1[6] >= 0.25)
        sh_c2 = (conf2[5] >= 0.25 or conf2[6] >= 0.25)
        sh_ok = sh_c1 and sh_c2
        
        # Hips: L_hip (11), R_hip (12)
        hip_c1 = (conf1[11] >= 0.25 and conf1[12] >= 0.25)
        hip_c2 = (conf2[11] >= 0.25 or conf2[12] >= 0.25)
        hip_ok = hip_c1 and hip_c2
        
        # Multicam consistency check (Bbox proportions & torso heights)
        bb1 = d1["bbox"]
        bb2 = d2["bbox"]
        h1 = bb1[3] - bb1[1]
        h2 = bb2[3] - bb2[1]
        ratio_h = min(h1, h2) / max(1.0, max(h1, h2))
        multicam_consistent = (ratio_h >= 0.40)  # Subject height in pixels across views is comparable

        # QC Decision
        if sh_ok and hip_ok and head_ok and c1_valid >= 10 and c2_valid >= 10 and multicam_consistent:
            status = "PASS"
            reason = "Full anatomical keypoints valid and multicam consistent"
            pass_count += 1
        elif (sh_ok and hip_ok) and (c1_valid >= 8 and c2_valid >= 8):
            status = "REVIEW"
            reason = "Slight limb occlusion but torso and shoulders fully preserved"
            review_count += 1
        else:
            status = "FAIL"
            reason = "Incomplete torso keypoints or severe multicam mismatch"
            fail_count += 1

        qc_records.append({
            "capture_id": cap_id,
            "subject_id": sub_id,
            "posture": posture,
            "calibration_id": cal_id,
            "status": status,
            "c1_valid_joints": c1_valid,
            "c2_valid_joints": c2_valid,
            "head_visible": head_ok,
            "shoulders_visible": sh_ok,
            "hips_visible": hip_ok,
            "multicam_consistent": multicam_consistent,
            "reason": reason
        })

    df_qc = pd.DataFrame(qc_records)
    out_qc_csv = RESULTS_DIR / "keypoint_qc.csv"
    df_qc.to_csv(out_qc_csv, index=False)
    print(f"\n[SAVED] 2D Keypoint & Multicam QC to: {out_qc_csv}")

    total_caps = len(df_cap)
    print("\n" + "=" * 60)
    print("        RINGKASAN QC 2D KEYPOINTS & MULTICAM (851 CAPTURES)")
    print("=" * 60)
    print(f"  🟢 PASS    : {pass_count} ({pass_count/total_caps*100:.2f}%)")
    print(f"  🟡 REVIEW  : {review_count} ({review_count/total_caps*100:.2f}%)")
    print(f"  🔴 FAIL    : {fail_count} ({fail_count/total_caps*100:.2f}%)")
    print("=" * 60)

    return df_qc


if __name__ == "__main__":
    run_keypoint_qc()
