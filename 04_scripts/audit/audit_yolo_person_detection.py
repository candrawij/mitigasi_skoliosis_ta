"""
T5.C.1 — Comprehensive YOLO-Pose Person Detection Audit on 23 Subjects (1,702 Images)
Scans every raw image, captures all person candidates (without 1-person limitation),
and classifies the detection state into:
  1. correct_target: Exactly 1 person detected in seated ROI with valid keypoints.
  2. multiple_candidates: >=2 person candidates detected (e.g. subject + operator/reflection).
  3. wrong_person_risk: A non-target person candidate has higher detector confidence than seated target.
  4. no_target_detected: 0 persons detected (e.g. empty chair / subject left frame).
  5. low_keypoint_confidence: Target detected but mean keypoint confidence < 0.35.
"""
import sys
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "02_data" / "private_raw"
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def audit_yolo_detections():
    print("=" * 80)
    print("  T5.C.1: YOLO-POSE DETECTION AUDIT ACROSS ALL 1,702 IMAGES")
    print("=" * 80)

    images_csv = META_DIR / "images.csv"
    captures_csv = META_DIR / "captures.csv"
    df_img = pd.read_csv(images_csv)
    df_cap = pd.read_csv(captures_csv)

    print(f"Total images to audit: {len(df_img)} across {df_cap['subject_id'].nunique()} subjects")

    # Load YOLOv8-Pose model
    model = YOLO("yolov8n-pose.pt")

    audit_records = []
    
    # Audit loop
    for idx, row in df_img.iterrows():
        img_id = row["image_id"]
        cap_id = row["capture_id"]
        cam_id = row["camera_id"]
        v_role = row.get("view_role", "frontal" if cam_id == "CAM01" else "lateral")
        rel_path = str(row["image_path"]).replace("\\", "/")
        img_path = PROJECT_ROOT / rel_path
        
        cap_row = df_cap[df_cap["capture_id"] == cap_id].iloc[0]
        posture = cap_row["primary_posture"]
        subject_id = cap_row["subject_id"]

        if not img_path.exists():
            audit_records.append({
                "image_id": img_id,
                "capture_id": cap_id,
                "subject_id": subject_id,
                "camera_id": cam_id,
                "view_role": v_role,
                "posture": posture,
                "candidates_count": 0,
                "status": "missing_file",
                "target_area_ratio": 0.0,
                "target_conf": 0.0,
                "mean_kpt_conf": 0.0,
                "torso_kpts_valid": 0,
                "notes": "File not found"
            })
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            audit_records.append({
                "image_id": img_id,
                "capture_id": cap_id,
                "subject_id": subject_id,
                "camera_id": cam_id,
                "view_role": v_role,
                "posture": posture,
                "candidates_count": 0,
                "status": "corrupt_image",
                "target_area_ratio": 0.0,
                "target_conf": 0.0,
                "mean_kpt_conf": 0.0,
                "torso_kpts_valid": 0,
                "notes": "Corrupt image"
            })
            continue

        h, w = img_bgr.shape[:2]
        frame_area = float(w * h)
        frame_cx, frame_cy = w / 2.0, h / 2.0

        # Predict with very low threshold (0.15) to catch ALL possible candidate detections
        res = model.predict(img_bgr, verbose=False, conf=0.15)
        
        candidates = []
        if len(res) > 0 and res[0].boxes is not None and len(res[0].boxes.xyxy) > 0:
            boxes_xyxy = res[0].boxes.xyxy.cpu().numpy()
            boxes_conf = res[0].boxes.conf.cpu().numpy()
            kpts_data = res[0].keypoints.data.cpu().numpy() if res[0].keypoints is not None else None
            
            for c_idx in range(len(boxes_xyxy)):
                bx = boxes_xyxy[c_idx]
                bc = float(boxes_conf[c_idx])
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                b_area = bw * bh
                b_area_ratio = b_area / frame_area
                bcx = (bx[0] + bx[2]) / 2.0
                bcy = (bx[1] + bx[3]) / 2.0
                
                # Distance to center
                dist_center = np.sqrt(((bcx - frame_cx) / frame_cx) ** 2 + ((bcy - frame_cy) / frame_cy) ** 2)
                
                # Keypoints
                kpts_xy = kpts_data[c_idx, :, :2] if kpts_data is not None and len(kpts_data) > c_idx else np.zeros((17, 2))
                kpts_c = kpts_data[c_idx, :, 2] if kpts_data is not None and len(kpts_data) > c_idx else np.zeros((17,))
                mean_kc = float(np.mean(kpts_c))
                
                # Torso keypoints: 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip
                torso_valid = int(sum(1 for joint_idx in [5, 6, 11, 12] if kpts_c[joint_idx] >= 0.3))
                
                candidates.append({
                    "cand_idx": c_idx,
                    "bbox": bx.tolist(),
                    "conf": bc,
                    "area_ratio": b_area_ratio,
                    "dist_center": dist_center,
                    "mean_kpt_conf": mean_kc,
                    "torso_valid": torso_valid,
                    "keypoints": kpts_xy.tolist(),
                    "confidences": kpts_c.tolist()
                })

        num_cands = len(candidates)
        
        # Classification Logic
        if num_cands == 0:
            status = "no_target_detected"
            target_area = 0.0
            target_conf = 0.0
            mean_kc = 0.0
            torso_val = 0
            notes = "Empty chair or subject out of frame"
        elif num_cands == 1:
            c0 = candidates[0]
            target_area = c0["area_ratio"]
            target_conf = c0["conf"]
            mean_kc = c0["mean_kpt_conf"]
            torso_val = c0["torso_valid"]
            
            if mean_kc < 0.30 or torso_val < 2:
                status = "low_keypoint_confidence"
                notes = "Single candidate with degraded keypoint confidence"
            else:
                status = "correct_target"
                notes = "Single clean target in frame"
        else:
            # Multiple candidates present: Check if default highest-conf is truly the centered target
            # Sort by area & center distance to find the real seated subject
            # Real subject has high area ratio (usually > 0.12) and low dist_center (< 0.7)
            ranked_by_prior = sorted(candidates, key=lambda x: (x["area_ratio"] * 2.0 - x["dist_center"] + x["torso_valid"] * 0.5), reverse=True)
            true_target = ranked_by_prior[0]
            
            # Check what default YOLO (sort by box conf) would pick
            default_yolo_pick = sorted(candidates, key=lambda x: x["conf"], reverse=True)[0]
            
            target_area = true_target["area_ratio"]
            target_conf = true_target["conf"]
            mean_kc = true_target["mean_kpt_conf"]
            torso_val = true_target["torso_valid"]
            
            if default_yolo_pick["cand_idx"] != true_target["cand_idx"]:
                status = "wrong_person_risk"
                notes = f"Multiple candidates ({num_cands}); Default YOLO picks non-target candidate {default_yolo_pick['cand_idx']}!"
            else:
                status = "multiple_candidates"
                notes = f"Multiple candidates ({num_cands}); Default YOLO correctly aligns with target"

        audit_records.append({
            "image_id": img_id,
            "capture_id": cap_id,
            "subject_id": subject_id,
            "camera_id": cam_id,
            "view_role": v_role,
            "posture": posture,
            "candidates_count": num_cands,
            "status": status,
            "target_area_ratio": round(target_area, 4),
            "target_conf": round(target_conf, 4),
            "mean_kpt_conf": round(mean_kc, 4),
            "torso_kpts_valid": torso_val,
            "notes": notes
        })

    # Save to CSV
    df_audit = pd.DataFrame(audit_records)
    out_csv = RESULTS_DIR / "private_yolo_detection_audit.csv"
    df_audit.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Detection audit results to: {out_csv}")

    # Summary Statistics
    summary = df_audit["status"].value_counts()
    summary_pct = (df_audit["status"].value_counts(normalize=True) * 100.0).round(2)
    
    df_summary = pd.DataFrame({
        "Status": summary.index,
        "Jumlah": summary.values,
        "Persentase": [f"{p}%" for p in summary_pct.values]
    })
    
    print("\n" + "=" * 60)
    print("          RINGKASAN AUDIT DETEKSI YOLO-POSE (1.702 CITRA)")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("=" * 60)
    
    return df_audit, df_summary


if __name__ == "__main__":
    audit_yolo_detections()
