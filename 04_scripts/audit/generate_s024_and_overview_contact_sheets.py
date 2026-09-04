"""
Generate Contact Sheet for S024 and 24-Subject Master Overview Contact Sheet
Uses existing 2D keypoints in selected_person annotations.
"""
import sys
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
SELECTED_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "selected_person"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"
SHEETS_DIR = RESULTS_DIR / "contact_sheets"
SHEETS_DIR.mkdir(parents=True, exist_ok=True)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def draw_skeleton(img, keypoints, confidences, threshold=0.25):
    annotated = img.copy()
    kpts = np.array(keypoints)
    confs = np.array(confidences)
    
    for p1_idx, p2_idx in SKELETON_EDGES:
        if confs[p1_idx] >= threshold and confs[p2_idx] >= threshold:
            pt1 = (int(kpts[p1_idx][0]), int(kpts[p1_idx][1]))
            pt2 = (int(kpts[p2_idx][0]), int(kpts[p2_idx][1]))
            cv2.line(annotated, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
            
    for i, (pt, conf) in enumerate(zip(kpts, confs)):
        if conf >= threshold:
            color = (0, 255, 0) if i in [5, 6, 11, 12] else (0, 0, 255)
            radius = 5 if i in [5, 6, 11, 12] else 3
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), radius, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
            
    return annotated


def generate_contact_sheets():
    df_cap = pd.read_csv(META_DIR / "captures.csv")
    df_img = pd.read_csv(META_DIR / "images.csv")
    subjects = sorted(df_cap["subject_id"].unique().tolist())
    print(f"Total subjects: {len(subjects)} | Total captures: {len(df_cap)}")

    # 1. Contact Sheet for S024
    print("\n[1/2] Generating Contact Sheet for S024...")
    sub_caps = df_cap[df_cap["subject_id"] == "S024"]
    postures_order = ["upright", "forward_head", "leaning_forward", "slouching", "leaning_left", "leaning_right", "reject"]
    sample_caps = []
    for p in postures_order:
        m = sub_caps[sub_caps["primary_posture"] == p]
        if len(m) > 0:
            sample_caps.append(m.iloc[0])
            
    thumb_w, thumb_h = 320, 180
    comp_h = thumb_h * 2 + 70
    comp_w = thumb_w * max(1, len(sample_caps))
    composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
    
    cv2.rectangle(composite, (0, 0), (comp_w, 60), (30, 30, 30), -1)
    cv2.putText(composite, f"CONTACT SHEET AUDIT QC: SUBJECT [S024] ({len(sub_caps)} captures | Rig CAL_011)", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
    for c_idx, cap_row in enumerate(sample_caps):
        cap_id = cap_row["capture_id"]
        post = cap_row["primary_posture"]
        cal_id = cap_row["calibration_id"]
        
        # CAM01
        row_c1 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")].iloc[0]
        i1_p = PROJECT_ROOT / str(row_c1["image_path"]).replace("\\", "/")
        img1 = cv2.imread(str(i1_p))
        if img1 is not None:
            f1 = SELECTED_DIR / f"{row_c1['image_id']}_selected_person.json"
            if f1.exists():
                with open(f1, "r", encoding="utf-8") as fp: d1 = json.load(fp)
                if d1.get("has_target"):
                    img1 = draw_skeleton(img1, d1["keypoints"], d1["confidences"])
            t1 = cv2.resize(img1, (thumb_w, thumb_h))
            cv2.putText(t1, f"CAM01: {post}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            composite[65:65+thumb_h, c_idx*thumb_w:(c_idx+1)*thumb_w] = t1

        # CAM02
        row_c2 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")].iloc[0]
        i2_p = PROJECT_ROOT / str(row_c2["image_path"]).replace("\\", "/")
        img2 = cv2.imread(str(i2_p))
        if img2 is not None:
            f2 = SELECTED_DIR / f"{row_c2['image_id']}_selected_person.json"
            if f2.exists():
                with open(f2, "r", encoding="utf-8") as fp: d2 = json.load(fp)
                if d2.get("has_target"):
                    img2 = draw_skeleton(img2, d2["keypoints"], d2["confidences"])
            t2 = cv2.resize(img2, (thumb_w, thumb_h))
            cv2.putText(t2, f"CAM02: {cap_id} ({cal_id})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            composite[65+thumb_h:65+thumb_h*2, c_idx*thumb_w:(c_idx+1)*thumb_w] = t2

    out_s24 = SHEETS_DIR / "contact_sheet_S024.jpg"
    cv2.imwrite(str(out_s24), composite)
    print(f"  [SAVED] {out_s24.name}")

    # 2. Master Overview Contact Sheet for All 24 Subjects
    print("\n[2/2] Generating Master Overview for All 24 Subjects...")
    # 24 subjects * 2 thumbnails (CAM01 frontal + CAM02 lateral) = 48 thumbnails
    # Grid: 6 columns x 8 rows
    grid_cols = 6
    grid_rows = 8
    thumb_w, thumb_h = 300, 168
    overview_comp = np.zeros((thumb_h * grid_rows + 70, thumb_w * grid_cols, 3), dtype=np.uint8)
    
    cv2.rectangle(overview_comp, (0, 0), (thumb_w * grid_cols, 60), (30, 30, 30), -1)
    cv2.putText(overview_comp, f"PRIVATE DATASET OVERVIEW: 24 SUBJECTS (S001 - S024) [{len(df_cap)} CAPTURES / {len(df_img)} IMAGES]", 
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

    thumb_idx = 0
    for sub in subjects:
        sub_c = df_cap[df_cap["subject_id"] == sub]
        rep_cap = sub_c[sub_c["primary_posture"] == "upright"]
        if len(rep_cap) == 0: rep_cap = sub_c
        rep_row = rep_cap.iloc[0]
        cap_id = rep_row["capture_id"]
        cal_id = rep_row["calibration_id"]

        # CAM01
        r_c1 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")].iloc[0]
        i1_p = PROJECT_ROOT / str(r_c1["image_path"]).replace("\\", "/")
        img1 = cv2.imread(str(i1_p))
        if img1 is not None:
            f1 = SELECTED_DIR / f"{r_c1['image_id']}_selected_person.json"
            if f1.exists():
                with open(f1, "r", encoding="utf-8") as fp: d1 = json.load(fp)
                if d1.get("has_target"):
                    img1 = draw_skeleton(img1, d1["keypoints"], d1["confidences"])
            t1 = cv2.resize(img1, (thumb_w, thumb_h))
            cv2.putText(t1, f"{sub} Front", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            r_idx = thumb_idx // grid_cols
            c_idx = thumb_idx % grid_cols
            overview_comp[65+r_idx*thumb_h:65+(r_idx+1)*thumb_h, c_idx*thumb_w:(c_idx+1)*thumb_w] = t1
            thumb_idx += 1

        # CAM02
        r_c2 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")].iloc[0]
        i2_p = PROJECT_ROOT / str(r_c2["image_path"]).replace("\\", "/")
        img2 = cv2.imread(str(i2_p))
        if img2 is not None:
            f2 = SELECTED_DIR / f"{r_c2['image_id']}_selected_person.json"
            if f2.exists():
                with open(f2, "r", encoding="utf-8") as fp: d2 = json.load(fp)
                if d2.get("has_target"):
                    img2 = draw_skeleton(img2, d2["keypoints"], d2["confidences"])
            t2 = cv2.resize(img2, (thumb_w, thumb_h))
            lat_side = r_c2.get('lateral_side', '')
            cv2.putText(t2, f"{sub} Side ({cal_id})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            r_idx = thumb_idx // grid_cols
            c_idx = thumb_idx % grid_cols
            overview_comp[65+r_idx*thumb_h:65+(r_idx+1)*thumb_h, c_idx*thumb_w:(c_idx+1)*thumb_w] = t2
            thumb_idx += 1

    out_all24 = SHEETS_DIR / "contact_sheet_all_24_subjects_overview.jpg"
    cv2.imwrite(str(out_all24), overview_comp)
    print(f"  [SAVED] {out_all24.name}")
    print("\nAll contact sheets generated successfully!")


if __name__ == "__main__":
    generate_contact_sheets()
