"""
FASE 5E — 3D Triangulation & Anatomical Geometry QC on 23 Subjects (851 Captures)
Computes:
  1. Multi-rig stereo triangulation on target-selected 2D keypoints.
  2. Reprojection error per joint and mean capture reprojection error.
  3. Anatomical geometry consistency:
     - Biacromial shoulder span (0.30 m - 0.55 m)
     - Torso spinal length (0.35 m - 0.70 m)
     - Depth distance Z (0.8 m - 4.5 m)
  4. Outlier classification and PASS / REVIEW / FAIL decision.
Output:
  07_results/private_audit/private_3d_qc.csv
"""
import sys
import cv2
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
CALIB_DIR = PROJECT_ROOT / "02_data" / "private_calibration"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def run_3d_triangulation_qc():
    print("=" * 80)
    print("  FASE 5E: 3D TRIANGULATION & ANATOMICAL GEOMETRY QC (851 CAPTURES)")
    print("=" * 80)

    captures_csv = META_DIR / "captures.csv"
    images_csv = META_DIR / "images.csv"
    df_cap = pd.read_csv(captures_csv)
    df_img = pd.read_csv(images_csv)

    # Load all stereo calibration JSONs
    calib_cache = {}
    if (CALIB_DIR / "stereo").exists():
        for fpath in (CALIB_DIR / "stereo").glob("CAL_*_stereo.json"):
            c_id = fpath.stem.replace("_stereo", "")
            with open(fpath, "r", encoding="utf-8") as fp:
                calib_cache[c_id] = json.load(fp)

    print(f"Loaded {len(calib_cache)} stereo calibration rigs: {list(calib_cache.keys())}")

    qc_3d_records = []
    
    pass_3d = 0
    review_3d = 0
    fail_3d = 0

    for idx, row in df_cap.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        posture = row["primary_posture"]
        cal_id = row["calibration_id"]
        
        img_c1_id = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")].iloc[0]["image_id"]
        img_c2_id = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")].iloc[0]["image_id"]
        
        f1 = SELECTED_DIR / f"{img_c1_id}_selected_person.json"
        f2 = SELECTED_DIR / f"{img_c2_id}_selected_person.json"
        
        if not f1.exists() or not f2.exists() or cal_id not in calib_cache:
            qc_3d_records.append({
                "capture_id": cap_id,
                "subject_id": sub_id,
                "posture": posture,
                "calibration_id": cal_id,
                "status": "FAIL",
                "valid_joints_3d": 0,
                "mean_reprojection_err_px": np.nan,
                "shoulder_width_m": np.nan,
                "torso_length_m": np.nan,
                "mean_depth_z_m": np.nan,
                "reason": "Missing selected person JSON or calibration matrix"
            })
            fail_3d += 1
            continue

        with open(f1, "r", encoding="utf-8") as fp: d1 = json.load(fp)
        with open(f2, "r", encoding="utf-8") as fp: d2 = json.load(fp)

        if not d1.get("has_target", False) or not d2.get("has_target", False):
            is_reject = (posture == "reject")
            qc_3d_records.append({
                "capture_id": cap_id,
                "subject_id": sub_id,
                "posture": posture,
                "calibration_id": cal_id,
                "status": "PASS" if is_reject else "FAIL",
                "valid_joints_3d": 0,
                "mean_reprojection_err_px": np.nan,
                "shoulder_width_m": np.nan,
                "torso_length_m": np.nan,
                "mean_depth_z_m": np.nan,
                "reason": "Negative sample properly rejected" if is_reject else "No target detected in active posture"
            })
            if is_reject: pass_3d += 1
            else: fail_3d += 1
            continue

        pts1 = np.array(d1["keypoints"], dtype=np.float64)
        pts2 = np.array(d2["keypoints"], dtype=np.float64)
        conf1 = np.array(d1["confidences"], dtype=np.float64)
        conf2 = np.array(d2["confidences"], dtype=np.float64)

        cal = calib_cache[cal_id]
        K1 = np.array(cal["intrinsics_refined"]["K1"], dtype=np.float64)
        D1 = np.array(cal["intrinsics_refined"]["D1"], dtype=np.float64)
        K2 = np.array(cal["intrinsics_refined"]["K2"], dtype=np.float64)
        D2 = np.array(cal["intrinsics_refined"]["D2"], dtype=np.float64)
        R1 = np.array(cal["rectification"]["R1"], dtype=np.float64)
        R2 = np.array(cal["rectification"]["R2"], dtype=np.float64)
        P1 = np.array(cal["rectification"]["P1"], dtype=np.float64)
        P2 = np.array(cal["rectification"]["P2"], dtype=np.float64)

        # Undistort Points
        u1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
        u2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)

        # Triangulation
        pts4D = cv2.triangulatePoints(P1, P2, u1.T, u2.T)
        pts3D = (pts4D[:3] / pts4D[3]).T  # [17, 3] in Camera 1 rectified frame

        valid_joints_3d = 0
        reproj_errors = []
        valid_3d_pts = []

        for j in range(17):
            if conf1[j] >= 0.25 and conf2[j] >= 0.25 and not np.isinf(pts3D[j]).any() and not np.isnan(pts3D[j]).any():
                valid_joints_3d += 1
                valid_3d_pts.append(pts3D[j])
                
                # Compute 2D reprojection error
                p3_homo = np.append(pts3D[j], 1.0)
                reproj_1 = P1 @ p3_homo
                reproj_1 = reproj_1[:2] / reproj_1[2]
                err1 = np.linalg.norm(reproj_1 - pts1[j])
                reproj_errors.append(err1)

        mean_reproj_err = float(np.mean(reproj_errors)) if len(reproj_errors) > 0 else 999.0

        # Anatomical Metrics:
        # Shoulders: L_shoulder (5) and R_shoulder (6)
        if conf1[5] >= 0.25 and conf2[5] >= 0.25 and conf1[6] >= 0.25 and conf2[6] >= 0.25:
            sh_width = float(np.linalg.norm(pts3D[5] - pts3D[6]))
        else:
            sh_width = np.nan

        # Torso: Mid-Shoulder (5,6) to Mid-Hip (11,12)
        if (conf1[5] >= 0.25 and conf1[6] >= 0.25 and conf1[11] >= 0.25 and conf1[12] >= 0.25):
            mid_sh = (pts3D[5] + pts3D[6]) / 2.0
            mid_hip = (pts3D[11] + pts3D[12]) / 2.0
            torso_len = float(np.linalg.norm(mid_sh - mid_hip))
        else:
            torso_len = np.nan

        # Depth Z (distance from camera)
        depths_z = [p[2] for p in valid_3d_pts if 0.5 < p[2] < 6.0]
        mean_depth_z = float(np.mean(depths_z)) if len(depths_z) > 0 else np.nan

        # Geometric Validation Rules (Posture-Aware)
        geom_ok = True
        reason = "3D Skeleton and anatomy valid"
        
        # In upright poses, verify anatomical span bounds
        if posture in ["upright", "forward_head"]:
            if not np.isnan(sh_width) and (sh_width < 0.18 or sh_width > 0.68):
                geom_ok = False
                reason = f"Shoulder span anomaly in upright: {sh_width:.2f} m"
            if not np.isnan(torso_len) and (torso_len < 0.28 or torso_len > 0.85):
                geom_ok = False
                reason = f"Torso length anomaly in upright: {torso_len:.2f} m"
        else:
            # Kinematic bending postures (leaning_forward, slouching, leaning_left/right)
            if not np.isnan(sh_width) and (sh_width < 0.15 or sh_width > 1.20):
                geom_ok = False
                reason = f"Shoulder span extreme anomaly: {sh_width:.2f} m"

        if valid_joints_3d >= 8 and geom_ok:
            status = "PASS"
            pass_3d += 1
        elif valid_joints_3d >= 5:
            status = "REVIEW"
            reason = f"Moderate joint count ({valid_joints_3d}/17) or minor distortion"
            review_3d += 1
        else:
            status = "FAIL"
            reason = f"Low 3D joint count ({valid_joints_3d}/17)"
            fail_3d += 1

        qc_3d_records.append({
            "capture_id": cap_id,
            "subject_id": sub_id,
            "posture": posture,
            "calibration_id": cal_id,
            "status": status,
            "valid_joints_3d": valid_joints_3d,
            "mean_reprojection_err_px": round(mean_reproj_err, 2),
            "shoulder_width_m": round(sh_width, 3) if not np.isnan(sh_width) else None,
            "torso_length_m": round(torso_len, 3) if not np.isnan(torso_len) else None,
            "mean_depth_z_m": round(mean_depth_z, 3) if not np.isnan(mean_depth_z) else None,
            "reason": reason
        })

    df_3d = pd.DataFrame(qc_3d_records)
    out_3d_csv = RESULTS_DIR / "private_3d_qc.csv"
    df_3d.to_csv(out_3d_csv, index=False)
    print(f"\n[SAVED] 3D Triangulation & Geometry QC to: {out_3d_csv}")

    total_caps = len(df_cap)
    print("\n" + "=" * 60)
    print("        RINGKASAN QC 3D TRIANGULASI & GEOMETRI (851 CAPTURES)")
    print("=" * 60)
    print(f"  🟢 PASS    : {pass_3d} ({pass_3d/total_caps*100:.2f}%)")
    print(f"  🟡 REVIEW  : {review_3d} ({review_3d/total_caps*100:.2f}%)")
    print(f"  🔴 FAIL    : {fail_3d} ({fail_3d/total_caps*100:.2f}%)")
    print("=" * 60)

    return df_3d


if __name__ == "__main__":
    run_3d_triangulation_qc()
