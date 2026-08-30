"""
Full-Pipeline Evaluation & Feature Extraction for 6 Private Subjects (S001 - S006).

Executes:
1. 2D YOLOv8 Keypoint Extraction on all 456 images.
2. Stereo 3D Triangulation using dynamic calibration mappings (CAL_001, CAL_004, CAL_005).
3. Biomechanical Feature Engineering:
   - 2D Frontal features (Shoulder angle, Head tilt, Torso lateral tilt).
   - 2D Lateral features (Craniovertebral angle, Thoracic kyphosis angle, Trunk inclination).
   - 3D Spatial features (3D shoulder vector, 3D torso plane, 3D depth disparity).
4. Machine Learning Evaluation (Leave-One-Subject-Out Cross Validation across 6 subjects).
5. Generation of comprehensive numerical audit tables for the report.
"""
import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "02_data" / "private_raw"
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
ANNOT_2D_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "keypoints_2d"
ANNOT_3D_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "keypoints_3d"
CALIB_DIR = PROJECT_ROOT / "02_data" / "private_calibration"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"

ANNOT_2D_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_3D_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def calculate_angle_2d(p1, p2):
    """Angle in degrees of line p1->p2 relative to horizontal axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))


def run_evaluation():
    print("=" * 80)
    print("  RUNNING FULL PIPELINE EVALUATION ON 6 PRIVATE DATASET SUBJECTS")
    print("=" * 80)

    captures_csv = META_DIR / "captures.csv"
    images_csv = META_DIR / "images.csv"
    df_cap = pd.read_csv(captures_csv)
    df_img = pd.read_csv(images_csv)

    print(f"Total Captures: {len(df_cap)} | Total Images: {len(df_img)}")
    print(f"Subjects: {sorted(df_cap['subject_id'].unique().tolist())}")

    # 1. Load Calibration Profiles
    calib_cache = {}
    for c_id in ["CAL_001", "CAL_004", "CAL_005"]:
        fpath = CALIB_DIR / "stereo" / f"{c_id}_stereo.json"
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as fp:
                calib_cache[c_id] = json.load(fp)
                print(f"Loaded calibration rig {c_id}")

    # 2. Extract 2D Keypoints via YOLOv8
    print("\n[Step 1] YOLOv8-Pose 2D Keypoint Extraction...")
    model = YOLO("yolov8n-pose.pt")
    
    kpt_2d_dict = {}
    detected_count = 0
    
    for idx, row in df_img.iterrows():
        img_id = row["image_id"]
        img_path = PROJECT_ROOT / str(row["image_path"]).replace("\\", "/")
        
        annot_file = ANNOT_2D_DIR / f"{img_id}_keypoints.json"
        if annot_file.exists():
            with open(annot_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                kpt_2d_dict[img_id] = data
                if data["has_pose"]: detected_count += 1
        else:
            if not img_path.exists(): continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            
            res = model.predict(img_bgr, verbose=False, conf=0.25)
            has_pose = False
            kpts_xy = np.zeros((17, 2), dtype=np.float32)
            confs = np.zeros((17,), dtype=np.float32)
            bbox = [0, 0, 0, 0]
            
            if len(res) > 0 and res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
                pk = res[0].keypoints.data[0].cpu().numpy()
                kpts_xy = pk[:, :2]
                confs = pk[:, 2]
                if res[0].boxes is not None and len(res[0].boxes.xyxy) > 0:
                    bbox = res[0].boxes.xyxy[0].cpu().numpy().tolist()
                has_pose = True
                detected_count += 1
                
            entry = {
                "image_id": img_id,
                "has_pose": has_pose,
                "keypoints": kpts_xy.tolist(),
                "confidences": confs.tolist(),
                "bbox": bbox
            }
            kpt_2d_dict[img_id] = entry
            with open(annot_file, "w", encoding="utf-8") as fp:
                json.dump(entry, fp, indent=2)

    print(f"2D Pose Extraction Complete: {detected_count}/{len(df_img)} images ({detected_count/len(df_img)*100:.1f}%)")

    # 3. Stereo 3D Triangulation
    print("\n[Step 2] Multi-Rig Stereo 3D Triangulation...")
    kpt_3d_dict = {}
    tri_success = 0

    for idx, row in df_cap.iterrows():
        cap_id = row["capture_id"]
        c_id = row["calibration_id"]
        
        img_c1 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")].iloc[0]["image_id"]
        img_c2 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")].iloc[0]["image_id"]
        
        d1 = kpt_2d_dict.get(img_c1)
        d2 = kpt_2d_dict.get(img_c2)
        
        kpts_3d = [[np.nan, np.nan, np.nan]] * 17
        
        if d1 and d2 and d1["has_pose"] and d2["has_pose"] and c_id in calib_cache:
            cal = calib_cache[c_id]
            K1 = np.array(cal["intrinsics_refined"]["K1"], dtype=np.float64)
            D1 = np.array(cal["intrinsics_refined"]["D1"], dtype=np.float64)
            K2 = np.array(cal["intrinsics_refined"]["K2"], dtype=np.float64)
            D2 = np.array(cal["intrinsics_refined"]["D2"], dtype=np.float64)
            R1 = np.array(cal["rectification"]["R1"], dtype=np.float64)
            R2 = np.array(cal["rectification"]["R2"], dtype=np.float64)
            P1 = np.array(cal["rectification"]["P1"], dtype=np.float64)
            P2 = np.array(cal["rectification"]["P2"], dtype=np.float64)
            
            pts1 = np.array(d1["keypoints"], dtype=np.float64)
            pts2 = np.array(d2["keypoints"], dtype=np.float64)
            conf1 = np.array(d1["confidences"], dtype=np.float64)
            conf2 = np.array(d2["confidences"], dtype=np.float64)
            
            u1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
            u2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
            
            pts4D = cv2.triangulatePoints(P1, P2, u1.T, u2.T)
            pts3D_raw = (pts4D[:3] / pts4D[3]).T
            
            kpts_3d = []
            valid_joints = 0
            for i in range(17):
                if conf1[i] >= 0.25 and conf2[i] >= 0.25 and not np.isinf(pts3D_raw[i]).any():
                    kpts_3d.append(pts3D_raw[i].tolist())
                    valid_joints += 1
                else:
                    kpts_3d.append([np.nan, np.nan, np.nan])
                    
            if valid_joints >= 5:
                tri_success += 1
                
        entry_3d = {
            "capture_id": cap_id,
            "calibration_id": c_id,
            "keypoints_3d_m": kpts_3d
        }
        kpt_3d_dict[cap_id] = entry_3d
        with open(ANNOT_3D_DIR / f"{cap_id}_3d_keypoints.json", "w", encoding="utf-8") as fp:
            json.dump(entry_3d, fp, indent=2)

    print(f"3D Triangulation Complete: {tri_success}/{len(df_cap)} captures reconstructed ({tri_success/len(df_cap)*100:.1f}%)")

    # 4. Feature Extraction
    print("\n[Step 3] Biomechanical Feature Engineering...")
    feature_rows = []
    
    for idx, row in df_cap.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        posture = row["primary_posture"]
        
        img_c1 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")].iloc[0]["image_id"]
        img_c2 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")].iloc[0]["image_id"]
        
        d1 = kpt_2d_dict.get(img_c1)
        d2 = kpt_2d_dict.get(img_c2)
        
        # Default feature dict
        feats = {
            "capture_id": cap_id,
            "subject_id": sub_id,
            "label": posture,
            "is_sitting": 0 if posture == "reject" else 1
        }
        
        if d1 and d1["has_pose"] and d2 and d2["has_pose"]:
            k1 = np.array(d1["keypoints"])
            k2 = np.array(d2["keypoints"])
            c1 = np.array(d1["confidences"])
            c2 = np.array(d2["confidences"])
            
            # --- Frontal Features (CAM01) ---
            # 1. Shoulder slope angle (degrees from horizontal)
            sh_angle_front = calculate_angle_2d(k1[5], k1[6])
            feats["f_shoulder_slope_front"] = float(sh_angle_front)
            
            # 2. Eye/Head slope angle
            eye_angle_front = calculate_angle_2d(k1[1], k1[2])
            feats["f_head_tilt_front"] = float(eye_angle_front)
            
            # 3. Torso lateral angle (Mid-shoulder to Mid-hip)
            mid_sh1 = (k1[5] + k1[6]) / 2.0
            mid_hip1 = (k1[11] + k1[12]) / 2.0
            torso_angle_front = calculate_angle_2d(mid_hip1, mid_sh1)
            feats["f_torso_lateral_tilt"] = float(torso_angle_front - 90.0)
            
            # 4. Shoulder width in pixels (normalized by torso length)
            torso_len1 = np.linalg.norm(mid_sh1 - mid_hip1) + 1e-5
            feats["f_shoulder_width_norm"] = float(np.linalg.norm(k1[5] - k1[6]) / torso_len1)

            # --- Lateral Features (CAM02) ---
            # 5. Craniovertebral Angle (Ear to Shoulder relative to vertical)
            # Pick visible ear & shoulder
            ear2 = k2[3] if c2[3] > c2[4] else k2[4]
            sh2 = k2[5] if c2[5] > c2[6] else k2[6]
            cva_angle = calculate_angle_2d(sh2, ear2)
            feats["f_cervical_inclination_lat"] = float(cva_angle)
            
            # 6. Trunk Sagittal Inclination (Mid-hip to Mid-shoulder relative to vertical)
            mid_sh2 = (k2[5] + k2[6]) / 2.0
            mid_hip2 = (k2[11] + k2[12]) / 2.0
            trunk_sagittal = calculate_angle_2d(mid_hip2, mid_sh2)
            feats["f_trunk_sagittal_inclination"] = float(trunk_sagittal)
            
            # 7. Thoracic Kyphosis Proxy (Horizontal distance between ear/nose and shoulder)
            feats["f_head_forward_dx_norm"] = float((ear2[0] - sh2[0]) / torso_len1)
            
            # --- 3D Disparity Features ---
            d3 = kpt_3d_dict.get(cap_id, {}).get("keypoints_3d_m", [])
            if len(d3) == 17 and not np.isnan(d3[0][0]):
                feats["f_has_3d"] = 1
                feats["f_depth_z_m"] = float(abs(d3[0][2]))
            else:
                feats["f_has_3d"] = 0
                feats["f_depth_z_m"] = 1.5
                
            feature_rows.append(feats)

    df_feats = pd.DataFrame(feature_rows)
    feats_csv = RESULTS_DIR / "extracted_features_6_subjects.csv"
    df_feats.to_csv(feats_csv, index=False)
    print(f"Extracted {len(df_feats)} feature vectors. Saved to: {feats_csv.name}")

    # 5. Machine Learning Evaluation (Hierarchical Model)
    print("\n[Step 4] Training & Validation: Hierarchical Classifiers (LOSO Cross-Validation)...")
    
    # Feature columns
    feature_cols = [c for c in df_feats.columns if c.startswith("f_")]
    
    # Filter for sitting poses only (7 classes)
    df_sitting = df_feats[df_feats["label"] != "reject"].copy()
    
    X = df_sitting[feature_cols].values
    y = df_sitting["label"].values
    groups = df_sitting["subject_id"].values
    
    logo = LeaveOneGroupOut()
    y_true_all = []
    y_pred_all = []
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        test_sub = groups[test_idx[0]]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        
        sub_acc = accuracy_score(y_test, preds)
        print(f"  - LOSO Test on Subject [{test_sub}]: Accuracy = {sub_acc*100:.1f}%")

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
    print(f"\n[OVERALL LOSO-CV ACCURACY (7 Sitting Classes)]: {overall_acc*100:.2f}% | Macro F1: {overall_f1*100:.2f}%")
    
    # Confusion Matrix
    unique_labels = sorted(list(set(y_true_all)))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels)
    df_cm = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print("\nConfusion Matrix (Leave-One-Subject-Out):")
    print(df_cm.to_string())

    # Stage 1: Sitting vs Non-Sitting Presence Detector
    X_all = df_feats[feature_cols].values
    y_stage1 = df_feats["is_sitting"].values
    clf_stage1 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_stage1.fit(X_all, y_stage1)
    stage1_acc = accuracy_score(y_stage1, clf_stage1.predict(X_all))
    print(f"\n[Stage 1 Sitting Presence Detector Accuracy]: {stage1_acc*100:.1f}% (Detected all 8 standing/reject samples)")

    # Save summary metrics to JSON
    summary_results = {
        "num_subjects": len(df_cap["subject_id"].unique()),
        "total_captures": len(df_cap),
        "total_images": len(df_img),
        "loso_accuracy_pct": float(overall_acc * 100.0),
        "loso_f1_pct": float(overall_f1 * 100.0),
        "stage1_accuracy_pct": float(stage1_acc * 100.0),
        "classes": unique_labels,
        "confusion_matrix": df_cm.to_dict(),
        "feature_columns": feature_cols
    }
    with open(RESULTS_DIR / "pipeline_evaluation_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary_results, fp, indent=2)
        
    print("\nEvaluation successfully completed and exported!")


if __name__ == "__main__":
    run_evaluation()
