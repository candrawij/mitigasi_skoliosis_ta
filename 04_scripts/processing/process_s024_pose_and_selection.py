"""
Process S024 Images: YOLOv8-Pose Extraction + Target-Person Selection
Generates:
  - 02_data/private_annotations/keypoints_2d/S024_*_keypoints.json
  - 02_data/private_annotations/selected_person/S024_*_selected_person.json
"""
import sys
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
KEYPOINTS_2D_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "keypoints_2d"
SELECTED_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "selected_person"

KEYPOINTS_2D_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT / "04_scripts" / "processing"))
from target_person_selector import TargetPersonSelector


def process_s024():
    print("=" * 70)
    print("  PROCESSING S024: YOLOV8-POSE & TARGET-PERSON SELECTION (68 IMAGES)")
    print("=" * 70)

    df_img = pd.read_csv(META_DIR / "images.csv")
    df_cap = pd.read_csv(META_DIR / "captures.csv")

    s024_imgs = df_img[df_img["image_id"].str.startswith("S024")].copy()
    print(f"Found {len(s024_imgs)} images for S024")

    selector = TargetPersonSelector()
    model = YOLO("yolov8n-pose.pt")

    cap_lookup = df_cap.set_index("capture_id").to_dict("index")

    processed = 0
    selected_target_count = 0
    rejected_count = 0

    for idx, row in s024_imgs.iterrows():
        img_id = row["image_id"]
        cap_id = row["capture_id"]
        cam_id = row["camera_id"]
        v_role = row["view_role"] if pd.notna(row["view_role"]) else ("frontal" if cam_id == "CAM01" else "lateral")
        
        cap_info = cap_lookup.get(cap_id, {})
        posture = cap_info.get("primary_posture", "unknown")
        subject_id = cap_info.get("subject_id", "S024")

        img_path = PROJECT_ROOT / str(row["image_path"]).replace("\\", "/")
        if not img_path.exists():
            print(f"Warning: Image file not found: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Failed to read image: {img_path}")
            continue

        h, w = img_bgr.shape[:2]

        # 1. YOLO Inference
        res = model.predict(img_bgr, verbose=False, conf=0.15)
        has_cands = len(res) > 0 and res[0].boxes is not None and len(res[0].boxes.xyxy) > 0

        if not has_cands:
            # No candidates
            k2d_entry = {
                "image_id": img_id,
                "has_pose": False,
                "keypoints": np.zeros((17, 2), dtype=np.float32).tolist(),
                "confidences": np.zeros((17,), dtype=np.float32).tolist(),
                "bbox": [0.0, 0.0, 0.0, 0.0]
            }
            sel_entry = {
                "image_id": img_id,
                "capture_id": cap_id,
                "subject_id": subject_id,
                "posture": posture,
                "has_target": False,
                "selected_cand_idx": -1,
                "composite_score": 0.0,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "keypoints": np.zeros((17, 2), dtype=np.float32).tolist(),
                "confidences": np.zeros((17,), dtype=np.float32).tolist(),
                "candidates_count": 0,
                "decision_reason": "No candidate detected by YOLO"
            }
            rejected_count += 1
        else:
            boxes_xyxy = res[0].boxes.xyxy.cpu().numpy()
            boxes_conf = res[0].boxes.conf.cpu().numpy()
            kpts_data = res[0].keypoints.data.cpu().numpy() if res[0].keypoints is not None else np.zeros((len(boxes_xyxy), 17, 3))

            # Target person selection
            sel_result = selector.select_target(boxes_xyxy, boxes_conf, kpts_data, (h, w), view_role=v_role)

            sel_entry = {
                "image_id": img_id,
                "capture_id": cap_id,
                "subject_id": subject_id,
                "posture": posture,
                "has_target": sel_result["has_target"],
                "selected_cand_idx": sel_result["selected_cand_idx"],
                "composite_score": sel_result["composite_score"],
                "bbox": sel_result["bbox"],
                "keypoints": sel_result["keypoints"],
                "confidences": sel_result["confidences"],
                "candidates_count": sel_result["candidates_count"],
                "decision_reason": sel_result["decision_reason"]
            }

            if sel_result["has_target"]:
                selected_target_count += 1
                k2d_entry = {
                    "image_id": img_id,
                    "has_pose": True,
                    "keypoints": sel_result["keypoints"],
                    "confidences": sel_result["confidences"],
                    "bbox": sel_result["bbox"]
                }
            else:
                rejected_count += 1
                k2d_entry = {
                    "image_id": img_id,
                    "has_pose": False,
                    "keypoints": np.zeros((17, 2), dtype=np.float32).tolist(),
                    "confidences": np.zeros((17,), dtype=np.float32).tolist(),
                    "bbox": sel_result["bbox"]
                }

        # Save files
        with open(KEYPOINTS_2D_DIR / f"{img_id}_keypoints.json", "w", encoding="utf-8") as fp:
            json.dump(k2d_entry, fp, indent=2)

        with open(SELECTED_DIR / f"{img_id}_selected_person.json", "w", encoding="utf-8") as fp:
            json.dump(sel_entry, fp, indent=2)

        processed += 1
        if processed % 10 == 0 or processed == len(s024_imgs):
            print(f"  Processed {processed}/{len(s024_imgs)} images (Target Selected: {selected_target_count}, Rejected/Empty: {rejected_count})")

    print(f"\n[DONE] S024 Complete! Processed: {processed} | Selected Target: {selected_target_count} | Empty/Rejected: {rejected_count}")


if __name__ == "__main__":
    process_s024()
