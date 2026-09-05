"""
private_inference_common.py — Common Reusable Engine for 2D Multi-View & Stereo 3D Posture Inference
Handles:
  1. YOLOv8n-Pose detection & TargetPersonSelector
  2. Reject / Invalid-Input Gate for 2D and 3D
  3. Stereo triangulation & reprojection error QC
  4. Feature extraction & normalization (using private_feature_common.py)
  5. Deployment model loading & XGBoost probabilistic inference
"""

import os
import sys
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure project root and preprocessing/processing in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "processing"))

from private_feature_common import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    MAIN_CLASSES,
    NUM_CLASSES,
    FEATURE_NAMES_2D,
    FEATURE_NAMES_3D,
    COCO_NOSE,
    COCO_LEFT_SHOULDER,
    COCO_RIGHT_SHOULDER,
    COCO_LEFT_HIP,
    COCO_RIGHT_HIP,
    MANDATORY_KP_INDICES,
    extract_single_view_2d_features,
    combine_2d_multi_view_features,
    extract_3d_features
)
from target_person_selector import TargetPersonSelector

CALIBRATION_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "stereo"
MODELS_DIR = PROJECT_ROOT / "06_models"


# ==============================================================================
# 1. MODEL & CALIBRATION LOADERS
# ==============================================================================

_YOLO_CACHE = None
_SELECTOR_CACHE = None
_DEPLOYMENT_CACHE = {}
_CALIB_CACHE = {}


def get_yolo_pose(weights_path: Optional[str] = None):
    """Load or retrieve cached YOLOv8n-pose model."""
    global _YOLO_CACHE
    if _YOLO_CACHE is None:
        from ultralytics import YOLO
        w_path = weights_path or str(PROJECT_ROOT / "yolov8n-pose.pt")
        _YOLO_CACHE = YOLO(w_path)
    return _YOLO_CACHE


def get_target_selector():
    """Retrieve cached TargetPersonSelector instance."""
    global _SELECTOR_CACHE
    if _SELECTOR_CACHE is None:
        _SELECTOR_CACHE = TargetPersonSelector()
    return _SELECTOR_CACHE


def load_deployment_pipeline(mode: str):
    """Load cached deployment pipeline (scaler + XGBoost) and metadata."""
    global _DEPLOYMENT_CACHE
    mode = mode.lower()
    if mode in _DEPLOYMENT_CACHE:
        return _DEPLOYMENT_CACHE[mode]

    m_dir = MODELS_DIR / f"keypoint_{mode}" / "private_final"
    pipeline_path = m_dir / "pipeline.pkl"
    meta_path = m_dir / "model_metadata.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline artifact missing in {m_dir}. Run fit_private_deployment_models.py first!")

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    _DEPLOYMENT_CACHE[mode] = (pipeline, metadata)
    return pipeline, metadata


def load_stereo_calibration(cal_id_or_name: str) -> Dict[str, Any]:
    """Load stereo calibration JSON by calibration ID (e.g. 'CAL_009') or path."""
    global _CALIB_CACHE
    cal_id = cal_id_or_name.replace("_stereo.json", "").replace(".json", "")
    if cal_id in _CALIB_CACHE:
        return _CALIB_CACHE[cal_id]

    # Try direct file path or lookup in CALIBRATION_DIR
    fpath = Path(cal_id_or_name)
    if not fpath.exists():
        fpath = CALIBRATION_DIR / f"{cal_id}_stereo.json"

    if not fpath.exists():
        raise FileNotFoundError(f"Calibration file not found for {cal_id_or_name} in {CALIBRATION_DIR}")

    with open(fpath, "r", encoding="utf-8") as f:
        cal = json.load(f)

    _CALIB_CACHE[cal_id] = cal
    return cal


# ==============================================================================
# 2. KEYPOINT DETECTION & TARGET PERSON SELECTION
# ==============================================================================

def detect_target_person_keypoints(
    image_bgr: np.ndarray,
    view_role: str = "frontal",
    conf_threshold: float = 0.25
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Detects pose and selects the seated participant from image.
    Returns: (is_valid, kpts_17x2, confs_17, selection_meta)
    """
    h, w = image_bgr.shape[:2]
    yolo = get_yolo_pose()
    selector = get_target_selector()

    results = yolo(image_bgr, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return False, None, None, {"has_target": False, "reason": "No person detected"}

    boxes_xyxy = boxes.xyxy.cpu().numpy()
    boxes_conf = boxes.conf.cpu().numpy()

    if results.keypoints is None or results.keypoints.data is None:
        return False, None, None, {"has_target": False, "reason": "No keypoints predicted"}

    kpts_data = results.keypoints.data.cpu().numpy()  # [N, 17, 3]

    sel_res = selector.select_target(
        boxes_xyxy=boxes_xyxy,
        boxes_conf=boxes_conf,
        kpts_data=kpts_data,
        frame_shape=(h, w),
        view_role=view_role
    )

    if not sel_res.get("has_target", False):
        return False, None, None, {"has_target": False, "reason": sel_res.get("decision_reason", "Target selector rejected candidates")}

    kpts = np.array(sel_res["keypoints"], dtype=np.float64)  # [17, 2]
    confs = np.array(sel_res["confidences"], dtype=np.float64)  # [17]

    return True, kpts, confs, sel_res


# ==============================================================================
# 3. REJECT / INVALID-INPUT GATE (2D & 3D)
# ==============================================================================

def check_2d_reject_gate(
    kpts_cam01: Optional[np.ndarray],
    confs_cam01: Optional[np.ndarray],
    kpts_cam02: Optional[np.ndarray],
    confs_cam02: Optional[np.ndarray],
    min_conf: float = 0.25
) -> Tuple[bool, str]:
    """
    Step 21: Reject Gate for 2D Multi-View.
    Enforces that core anatomical landmarks exist with sufficient confidence on BOTH views.
    """
    if kpts_cam01 is None or confs_cam01 is None:
        return False, "REJECT_NO_PERSON_CAM01"
    if kpts_cam02 is None or confs_cam02 is None:
        return False, "REJECT_NO_PERSON_CAM02"

    # Check mandatory core keypoints: Left/Right Shoulder (5,6) and Left/Right Hip (11,12)
    for idx in MANDATORY_KP_INDICES:
        if confs_cam01[idx] < min_conf:
            return False, f"REJECT_CAM01_LOW_CONF_JOINT_{idx}"
        if np.isnan(kpts_cam01[idx, 0]) or np.isnan(kpts_cam01[idx, 1]):
            return False, f"REJECT_CAM01_NAN_JOINT_{idx}"

        if confs_cam02[idx] < min_conf:
            return False, f"REJECT_CAM02_LOW_CONF_JOINT_{idx}"
        if np.isnan(kpts_cam02[idx, 0]) or np.isnan(kpts_cam02[idx, 1]):
            return False, f"REJECT_CAM02_NAN_JOINT_{idx}"

    # Check torso scale factor S > 0 on both views
    sh_c_1 = (kpts_cam01[COCO_LEFT_SHOULDER] + kpts_cam01[COCO_RIGHT_SHOULDER]) / 2.0
    hip_c_1 = (kpts_cam01[COCO_LEFT_HIP] + kpts_cam01[COCO_RIGHT_HIP]) / 2.0
    s1 = np.linalg.norm(sh_c_1 - hip_c_1)
    if s1 < 5.0:  # in pixels
        return False, "REJECT_CAM01_DEGENERATE_TORSO_SCALE"

    sh_c_2 = (kpts_cam02[COCO_LEFT_SHOULDER] + kpts_cam02[COCO_RIGHT_SHOULDER]) / 2.0
    hip_c_2 = (kpts_cam02[COCO_LEFT_HIP] + kpts_cam02[COCO_RIGHT_HIP]) / 2.0
    s2 = np.linalg.norm(sh_c_2 - hip_c_2)
    if s2 < 5.0:
        return False, "REJECT_CAM02_DEGENERATE_TORSO_SCALE"

    return True, "VALID_2D"


# ==============================================================================
# 4. STEREO TRIANGULATION & 3D QC GATE
# ==============================================================================

def triangulate_stereo_pair(
    kpts_cam01: np.ndarray,
    confs_cam01: np.ndarray,
    kpts_cam02: np.ndarray,
    confs_cam02: np.ndarray,
    cal_data: Dict[str, Any],
    orig_img_shape: Tuple[int, int] = (1080, 1920)
) -> Tuple[bool, Optional[np.ndarray], float, str]:
    """
    Performs camera-rectified stereo triangulation from Camera 1 and Camera 2 keypoints.
    Returns: (is_valid_3d, kpts_3d_m [17, 3], core_reproj_error, reason)
    """
    cal_w = cal_data.get("image_width", 640)
    cal_h = cal_data.get("image_height", 480)
    h_orig, w_orig = orig_img_shape

    scale_x = cal_w / float(w_orig)
    scale_y = cal_h / float(h_orig)

    # Scale 2D points to calibration resolution (640x480)
    pts1_cal = kpts_cam01.copy()
    pts1_cal[:, 0] *= scale_x
    pts1_cal[:, 1] *= scale_y

    pts2_cal = kpts_cam02.copy()
    pts2_cal[:, 0] *= scale_x
    pts2_cal[:, 1] *= scale_y

    K1 = np.array(cal_data["intrinsics_refined"]["K1"], dtype=np.float64)
    D1 = np.array(cal_data["intrinsics_refined"]["D1"], dtype=np.float64)
    K2 = np.array(cal_data["intrinsics_refined"]["K2"], dtype=np.float64)
    D2 = np.array(cal_data["intrinsics_refined"]["D2"], dtype=np.float64)
    R1 = np.array(cal_data["rectification"]["R1"], dtype=np.float64)
    R2 = np.array(cal_data["rectification"]["R2"], dtype=np.float64)
    P1 = np.array(cal_data["rectification"]["P1"], dtype=np.float64)
    P2 = np.array(cal_data["rectification"]["P2"], dtype=np.float64)

    # Undistort points to rectified camera planes
    u1_rect = cv2.undistortPoints(pts1_cal.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
    u2_rect = cv2.undistortPoints(pts2_cal.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)

    # Stereo triangulation (in rectified 3D frame)
    pts4D = cv2.triangulatePoints(P1, P2, u1_rect.T, u2_rect.T)
    w_coords = pts4D[3]
    w_coords[np.abs(w_coords) < 1e-7] = 1e-7
    pts3D_rect = (pts4D[:3] / w_coords).T  # [17, 3]

    # Transform back to original CAM01 camera coordinate frame
    X_orig_cam1 = (R1.T @ pts3D_rect.T).T  # [17, 3] in meters

    # Reproject to CAM01 to evaluate geometric accuracy
    reproj_cam1, _ = cv2.projectPoints(X_orig_cam1, np.zeros(3), np.zeros(3), K1, D1)
    reproj_cam1 = reproj_cam1.reshape(-1, 2)

    core_errors = []
    for j in MANDATORY_KP_INDICES:
        err = np.linalg.norm(pts1_cal[j] - reproj_cam1[j])
        core_errors.append(err)

    mean_core_reproj = float(np.mean(core_errors))

    # Mask low confidence or invalid joints as NaN
    kpts_3d_out = np.full((17, 3), np.nan, dtype=np.float64)
    for j in range(17):
        if confs_cam01[j] >= 0.25 and confs_cam02[j] >= 0.25:
            # Positive depth check (Z > 0)
            if X_orig_cam1[j, 2] > 0.2:
                kpts_3d_out[j] = X_orig_cam1[j]

    # Step 21 3D QC Gate Checks:
    # 1. All mandatory core joints must have valid 3D points
    for j in MANDATORY_KP_INDICES:
        if np.isnan(kpts_3d_out[j, 0]):
            return False, None, mean_core_reproj, f"REJECT_3D_INVALID_CORE_JOINT_{j}"

    # 2. Torso 3D length anatomical validity check (0.15m <= L <= 0.90m)
    sh_c_3d = (kpts_3d_out[COCO_LEFT_SHOULDER] + kpts_3d_out[COCO_RIGHT_SHOULDER]) / 2.0
    hip_c_3d = (kpts_3d_out[COCO_LEFT_HIP] + kpts_3d_out[COCO_RIGHT_HIP]) / 2.0
    torso_len_3d = np.linalg.norm(sh_c_3d - hip_c_3d)
    if torso_len_3d < 0.15 or torso_len_3d > 0.90:
        return False, None, mean_core_reproj, f"REJECT_3D_ANATOMICAL_TORSO_LEN_{torso_len_3d:.2f}m"

    # 3. Mean core reprojection error threshold (< 35 px at 640p)
    if mean_core_reproj > 35.0:
        return False, None, mean_core_reproj, f"REJECT_3D_HIGH_REPROJECTION_ERROR_{mean_core_reproj:.1f}px"

    return True, kpts_3d_out, mean_core_reproj, "VALID_3D"


# ==============================================================================
# 5. END-TO-END INFERENCE FUNCTIONS
# ==============================================================================

def infer_pair_2d(
    img_cam01: np.ndarray,
    img_cam02: np.ndarray,
    lateral_side: str = "right"
) -> Dict[str, Any]:
    """
    End-to-End 2D Multi-View Inference on image pair.
    """
    # 1. Pose Detection
    ok1, kpts1, confs1, meta1 = detect_target_person_keypoints(img_cam01, view_role="frontal")
    ok2, kpts2, confs2, meta2 = detect_target_person_keypoints(img_cam02, view_role="lateral")

    # 2. Reject Gate
    is_valid, reason = check_2d_reject_gate(kpts1, confs1, kpts2, confs2)
    if not is_valid:
        return {
            "status": "REJECTED",
            "reason": reason,
            "prediction": "REJECT",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in MAIN_CLASSES},
            "meta": {"cam01": meta1, "cam02": meta2}
        }

    # 3. Extract 36 Features
    c1_feat, c1_ok, c1_msg = extract_single_view_2d_features(
        kpts1, confs1, view_role="frontal", lateral_side=lateral_side
    )
    c2_feat, c2_ok, c2_msg = extract_single_view_2d_features(
        kpts2, confs2, view_role="lateral", lateral_side=lateral_side
    )
    if not c1_ok or not c2_ok:
        return {
            "status": "REJECTED",
            "reason": f"REJECT_2D_FEATURE_EXTRACTION_FAILED: {c1_msg} | {c2_msg}",
            "prediction": "REJECT",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in MAIN_CLASSES}
        }

    combined_dict = combine_2d_multi_view_features(c1_feat, c2_feat)
    f36 = np.array([combined_dict[fn] for fn in FEATURE_NAMES_2D], dtype=np.float64)

    # 4. Predict using Deployment Pipeline
    pipeline, meta = load_deployment_pipeline("2d")
    x_input = f36.reshape(1, -1)
    y_pred = int(pipeline.predict(x_input)[0])
    y_prob = pipeline.predict_proba(x_input)[0]

    pred_class = ID_TO_CLASS[y_pred]
    conf = float(y_prob[y_pred])

    return {
        "status": "VALID",
        "reason": "OK",
        "prediction": pred_class,
        "confidence": conf,
        "class_id": y_pred,
        "probabilities": {MAIN_CLASSES[i]: float(y_prob[i]) for i in range(NUM_CLASSES)},
        "features_count": 36
    }


def infer_pair_3d(
    img_cam01: np.ndarray,
    img_cam02: np.ndarray,
    calibration_id_or_path: str = "CAL_009"
) -> Dict[str, Any]:
    """
    End-to-End Stereo 3D Inference on image pair.
    """
    # 1. Pose Detection
    ok1, kpts1, confs1, meta1 = detect_target_person_keypoints(img_cam01, view_role="frontal")
    ok2, kpts2, confs2, meta2 = detect_target_person_keypoints(img_cam02, view_role="lateral")

    if not ok1 or not ok2:
        return {
            "status": "REJECTED",
            "reason": "REJECT_NO_PERSON_DETECTED",
            "prediction": "REJECT / INVALID_3D",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in MAIN_CLASSES}
        }

    # 2. Calibration & Triangulation
    try:
        cal_data = load_stereo_calibration(calibration_id_or_path)
    except Exception as e:
        return {
            "status": "REJECTED",
            "reason": f"REJECT_CALIBRATION_ERROR: {e}",
            "prediction": "REJECT / INVALID_3D",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in MAIN_CLASSES}
        }

    h1, w1 = img_cam01.shape[:2]
    is_valid_3d, kpts_3d, reproj_err, reason = triangulate_stereo_pair(
        kpts1, confs1, kpts2, confs2, cal_data, orig_img_shape=(h1, w1)
    )

    if not is_valid_3d or kpts_3d is None:
        return {
            "status": "REJECTED",
            "reason": reason,
            "reprojection_error": reproj_err,
            "prediction": "REJECT / INVALID_3D",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in MAIN_CLASSES}
        }

    # 3. Extract 25 Features
    f25_dict, is_feat_valid, msg = extract_3d_features(kpts_3d)
    if not is_feat_valid or f25_dict is None:
        return {
            "status": "REJECTED",
            "reason": f"REJECT_3D_FEATURE_EXTRACTION_FAILED: {msg}",
            "reprojection_error": reproj_err,
            "prediction": "REJECT / INVALID_3D",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in MAIN_CLASSES}
        }
    f25 = np.array([f25_dict[fn] for fn in FEATURE_NAMES_3D], dtype=np.float64)

    # 4. Predict using Deployment Pipeline
    pipeline, meta = load_deployment_pipeline("3d")
    x_input = f25.reshape(1, -1)
    y_pred = int(pipeline.predict(x_input)[0])
    y_prob = pipeline.predict_proba(x_input)[0]

    pred_class = ID_TO_CLASS[y_pred]
    conf = float(y_prob[y_pred])

    return {
        "status": "VALID",
        "reason": "OK",
        "prediction": pred_class,
        "confidence": conf,
        "class_id": y_pred,
        "reprojection_error": round(reproj_err, 2),
        "probabilities": {MAIN_CLASSES[i]: float(y_prob[i]) for i in range(NUM_CLASSES)},
        "features_count": 25
    }
