"""
private_feature_common.py — Common Reusable Feature Engineering & Normalization Module
for Private Dataset 6-Class 2D Multi-View & Stereo 3D Pipeline (XGBoost).

Used identically in:
  - Feature extraction for training (extract_private_2d_features.py, extract_private_3d_features.py)
  - Deployment model fitting (fit_private_deployment_models.py)
  - Single capture & batch evaluation (test_private_single_capture.py)
  - Offline pair inference (infer_private_pair_2d.py, infer_private_pair_3d.py)
  - Real-time webcam inference (infer_realtime_2d.py, infer_realtime_stereo_3d.py)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# ==============================================================================
# 1. TAXONOMY & CONSTANTS
# ==============================================================================

CLASS_TO_ID = {
    "upright": 0,
    "leaning_forward": 1,
    "leaning_backward": 2,
    "leaning_left": 3,
    "leaning_right": 4,
    "slouching": 5,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

MAIN_CLASSES = [
    "upright",
    "leaning_forward",
    "leaning_backward",
    "leaning_left",
    "leaning_right",
    "slouching",
]

NUM_CLASSES = 6

# COCO-17 Keypoint Indices
COCO_NOSE = 0
COCO_LEFT_EYE = 1
COCO_RIGHT_EYE = 2
COCO_LEFT_EAR = 3
COCO_RIGHT_EAR = 4
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

CORE_KP_INDICES = [COCO_NOSE, COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]
MANDATORY_KP_INDICES = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]

# 2D Feature schema (18 per view, 36 total)
BASE_2D_FEATURES = [
    "nose_x", "nose_y",
    "left_shoulder_x", "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "left_hip_x", "left_hip_y",
    "right_hip_x", "right_hip_y",
    "shoulder_slope_deg",
    "hip_slope_deg",
    "torso_inclination_deg",
    "head_torso_angle_deg",
    "head_to_shoulder_norm",
    "torso_length_norm",
    "head_horizontal_offset_norm",
    "torso_horizontal_offset_norm"
]

FEATURE_NAMES_2D = [f"cam01_{f}" for f in BASE_2D_FEATURES] + [f"cam02_{f}" for f in BASE_2D_FEATURES]

# 3D Feature schema (25 total)
FEATURE_NAMES_3D = [
    "nose_x", "nose_y", "nose_z",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
    "left_hip_x", "left_hip_y", "left_hip_z",
    "right_hip_x", "right_hip_y", "right_hip_z",
    "shoulder_roll_deg",
    "hip_roll_deg",
    "torso_lateral_lean_deg",
    "torso_sagittal_lean_deg",
    "torso_3d_inclination_deg",
    "head_torso_angle_3d_deg",
    "head_depth_offset_norm",
    "head_lateral_offset_norm",
    "shoulder_depth_asymmetry_norm",
    "hip_depth_asymmetry_norm"
]

# Coordinate Convention for 3D
COORDINATE_CONVENTION_3D = {
    "reference_frame": "CAM01_camera_frame",
    "axis_x": "lateral_horizontal_pointing_right",
    "axis_y": "vertical_pointing_downwards",
    "axis_z": "depth_optical_axis_pointing_forward",
    "units": "meters_before_normalization"
}


# ==============================================================================
# 2. GEOMETRIC UTILITIES
# ==============================================================================

def angle_between_vectors_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute unsigned angle between two 2D vectors in degrees [0, 180]."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return float("nan")
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def angle_between_vectors_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute unsigned angle between two 3D vectors in degrees [0, 180]."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return float("nan")
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


# ==============================================================================
# 3. 2D POSE NORMALIZATION & FEATURE EXTRACTION
# ==============================================================================

def validate_core_keypoints_2d(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    conf_thresh: float = 0.25
) -> Tuple[bool, str, bool]:
    """
    Validate that mandatory core keypoints (shoulders and hips) are detected.
    Returns:
        (is_valid, reason, is_nose_valid)
    """
    if len(keypoints) < 17 or len(confidences) < 17:
        return False, "Insufficient keypoints length (<17)", False

    for idx in MANDATORY_KP_INDICES:
        if np.isnan(keypoints[idx, 0]) or np.isnan(keypoints[idx, 1]) or confidences[idx] < conf_thresh:
            kp_name = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"][MANDATORY_KP_INDICES.index(idx)]
            return False, f"Mandatory keypoint missing or low confidence: {kp_name}", False

    is_nose_valid = (
        not np.isnan(keypoints[COCO_NOSE, 0])
        and not np.isnan(keypoints[COCO_NOSE, 1])
        and confidences[COCO_NOSE] >= conf_thresh
    )

    return True, "Core keypoints valid", is_nose_valid


def extract_single_view_2d_features(
    keypoints_17: np.ndarray,
    confidences_17: np.ndarray,
    view_role: str = "frontal",
    lateral_side: str = "right",
    conf_thresh: float = 0.25,
    eps: float = 1e-6
) -> Tuple[Optional[Dict[str, float]], bool, str]:
    """
    Extract 18 normalized 2D features for a single view (CAM01 or CAM02).
    Applies lateral canonicalization for CAM02 when lateral_side == 'left'.
    """
    is_valid, reason, valid_nose = validate_core_keypoints_2d(keypoints_17, confidences_17, conf_thresh)
    if not is_valid:
        return None, False, reason

    kpts = keypoints_17.copy()

    # Step 1: Centering relative to hip_center
    left_hip = kpts[COCO_LEFT_HIP]
    right_hip = kpts[COCO_RIGHT_HIP]
    hip_center = (left_hip + right_hip) / 2.0

    centered = kpts - hip_center

    # Step 2: Canonicalization for CAM02
    # If lateral camera is looking from the left side, flip horizontal coordinate
    # so all lateral views follow the same canonical orientation
    if view_role == "lateral" and lateral_side == "left":
        centered[:, 0] = -centered[:, 0]

    # Step 3: Scale normalization
    # S = max Euclidean distance from hip_center to all valid core landmarks
    core_valid_indices = [idx for idx in MANDATORY_KP_INDICES]
    if valid_nose:
        core_valid_indices.append(COCO_NOSE)

    distances = [np.linalg.norm(centered[idx]) for idx in core_valid_indices]
    scale_s = max(distances)

    if scale_s < eps or np.isnan(scale_s):
        return None, False, f"Invalid scale factor S={scale_s}"

    normalized = centered / scale_s

    # Core normalized points
    nose_p = normalized[COCO_NOSE] if valid_nose else np.array([np.nan, np.nan])
    l_sh = normalized[COCO_LEFT_SHOULDER]
    r_sh = normalized[COCO_RIGHT_SHOULDER]
    l_hip = normalized[COCO_LEFT_HIP]
    r_hip = normalized[COCO_RIGHT_HIP]

    # Derived geometric points & vectors
    shoulder_center = (l_sh + r_sh) / 2.0
    hip_center_norm = (l_hip + r_hip) / 2.0  # equals [0, 0]
    torso_vector = shoulder_center - hip_center_norm  # vector pointing up from hip to shoulder
    head_vector = (nose_p - shoulder_center) if valid_nose else np.array([np.nan, np.nan])

    # In image coordinates, Y increases downward.
    # Therefore, upward vertical axis is [0, -1]
    upward_vertical = np.array([0.0, -1.0])

    # Slopes
    # Right shoulder relative to left shoulder
    dx_sh = r_sh[0] - l_sh[0]
    dy_sh = r_sh[1] - l_sh[1]
    shoulder_slope_deg = float(np.degrees(np.arctan2(dy_sh, dx_sh)))

    dx_hip = r_hip[0] - l_hip[0]
    dy_hip = r_hip[1] - l_hip[1]
    hip_slope_deg = float(np.degrees(np.arctan2(dy_hip, dx_hip)))

    # Torso inclination angle relative to upward vertical
    torso_inclination_deg = angle_between_vectors_2d(torso_vector, upward_vertical)

    # Torso length normalized
    torso_length_norm = float(np.linalg.norm(torso_vector))

    # Torso horizontal offset (shoulder_center.x - hip_center.x)
    torso_horizontal_offset_norm = float(shoulder_center[0] - hip_center_norm[0])

    # Head features (conditional on valid nose)
    if valid_nose:
        head_torso_angle_deg = angle_between_vectors_2d(head_vector, torso_vector)
        head_to_shoulder_norm = float(np.linalg.norm(head_vector))
        head_horizontal_offset_norm = float(nose_p[0] - shoulder_center[0])
    else:
        head_torso_angle_deg = float("nan")
        head_to_shoulder_norm = float("nan")
        head_horizontal_offset_norm = float("nan")

    feat_dict = {
        "nose_x": float(nose_p[0]),
        "nose_y": float(nose_p[1]),
        "left_shoulder_x": float(l_sh[0]),
        "left_shoulder_y": float(l_sh[1]),
        "right_shoulder_x": float(r_sh[0]),
        "right_shoulder_y": float(r_sh[1]),
        "left_hip_x": float(l_hip[0]),
        "left_hip_y": float(l_hip[1]),
        "right_hip_x": float(r_hip[0]),
        "right_hip_y": float(r_hip[1]),
        "shoulder_slope_deg": shoulder_slope_deg,
        "hip_slope_deg": hip_slope_deg,
        "torso_inclination_deg": torso_inclination_deg,
        "head_torso_angle_deg": head_torso_angle_deg,
        "head_to_shoulder_norm": head_to_shoulder_norm,
        "torso_length_norm": torso_length_norm,
        "head_horizontal_offset_norm": head_horizontal_offset_norm,
        "torso_horizontal_offset_norm": torso_horizontal_offset_norm
    }

    return feat_dict, True, "Success"


def combine_2d_multi_view_features(
    c1_feat: Dict[str, float],
    c2_feat: Dict[str, float]
) -> Dict[str, float]:
    """Combine CAM01 and CAM02 feature dictionaries with prefixes into a 36-feature dict."""
    combined = {}
    for k in BASE_2D_FEATURES:
        combined[f"cam01_{k}"] = c1_feat[k]
    for k in BASE_2D_FEATURES:
        combined[f"cam02_{k}"] = c2_feat[k]
    return combined


# ==============================================================================
# 4. 3D POSE NORMALIZATION & SPATIAL GEOMETRY EXTRACTION
# ==============================================================================

def validate_core_keypoints_3d(keypoints_3d_m: np.ndarray) -> Tuple[bool, str, bool]:
    """
    Validate that mandatory core 3D keypoints (shoulders and hips) are not NaN.
    Returns:
        (is_valid, reason, is_nose_valid)
    """
    if len(keypoints_3d_m) < 17:
        return False, "Insufficient 3D keypoints (<17)", False

    for idx in MANDATORY_KP_INDICES:
        pt = keypoints_3d_m[idx]
        if pt is None or np.isnan(pt).any():
            kp_name = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"][MANDATORY_KP_INDICES.index(idx)]
            return False, f"Mandatory 3D joint is NaN: {kp_name}", False

    nose_pt = keypoints_3d_m[COCO_NOSE]
    is_nose_valid = (nose_pt is not None) and (not np.isnan(nose_pt).any())

    return True, "Core 3D joints valid", is_nose_valid


def extract_3d_features(
    keypoints_3d_m: np.ndarray,
    eps: float = 1e-6
) -> Tuple[Optional[Dict[str, float]], bool, str]:
    """
    Extract 25 spatial geometry features from 3D triangulated keypoints in meters.
    Coordinate reference frame is CAM01 (X=lateral right, Y=downwards, Z=depth optical axis).
    """
    kpts3d = np.array(keypoints_3d_m, dtype=np.float64)
    is_valid, reason, valid_nose = validate_core_keypoints_3d(kpts3d)
    if not is_valid:
        return None, False, reason

    # Step 1: Center at hip center
    l_hip_raw = kpts3d[COCO_LEFT_HIP]
    r_hip_raw = kpts3d[COCO_RIGHT_HIP]
    hip_center_3d = (l_hip_raw + r_hip_raw) / 2.0

    centered3d = kpts3d - hip_center_3d

    # Step 2: Scale normalization S3
    core_valid = [idx for idx in MANDATORY_KP_INDICES]
    if valid_nose:
        core_valid.append(COCO_NOSE)

    distances = [np.linalg.norm(centered3d[idx]) for idx in core_valid]
    scale_s3 = max(distances)

    if scale_s3 < eps or np.isnan(scale_s3):
        return None, False, f"Invalid 3D scale S3={scale_s3}"

    norm3d = centered3d / scale_s3

    # Normalized landmark coordinates
    nose_p = norm3d[COCO_NOSE] if valid_nose else np.array([np.nan, np.nan, np.nan])
    l_sh = norm3d[COCO_LEFT_SHOULDER]
    r_sh = norm3d[COCO_RIGHT_SHOULDER]
    l_hip = norm3d[COCO_LEFT_HIP]
    r_hip = norm3d[COCO_RIGHT_HIP]

    # Vectors
    shoulder_center = (l_sh + r_sh) / 2.0
    hip_center_norm = (l_hip + r_hip) / 2.0  # [0, 0, 0]
    torso_vector = shoulder_center - hip_center_norm
    head_vector = (nose_p - shoulder_center) if valid_nose else np.array([np.nan, np.nan, np.nan])

    # Upward vertical vector in 3D: [0, -1, 0]
    upward_vertical_3d = np.array([0.0, -1.0, 0.0])

    # Spatial Geometry Features:
    # 1. Shoulder roll in (X, Y) coronal plane: angle of shoulder vector relative to horizontal [1, 0]
    dx_sh = r_sh[0] - l_sh[0]
    dy_sh = r_sh[1] - l_sh[1]
    shoulder_roll_deg = float(np.degrees(np.arctan2(dy_sh, dx_sh)))

    # 2. Hip roll in (X, Y) coronal plane
    dx_hip = r_hip[0] - l_hip[0]
    dy_hip = r_hip[1] - l_hip[1]
    hip_roll_deg = float(np.degrees(np.arctan2(dy_hip, dx_hip)))

    # 3. Torso lateral lean: projection of torso_vector onto (X, Y) plane
    # atan2(torso.x, -torso.y): positive = leaning right, negative = leaning left
    torso_lateral_lean_deg = float(np.degrees(np.arctan2(torso_vector[0], -torso_vector[1])))

    # 4. Torso sagittal lean: projection of torso_vector onto (Z, Y) plane
    # atan2(torso.z, -torso.y): positive = leaning forward, negative = leaning backward
    torso_sagittal_lean_deg = float(np.degrees(np.arctan2(torso_vector[2], -torso_vector[1])))

    # 5. Torso 3D inclination: angle between 3D torso vector and upward vertical [0, -1, 0]
    torso_3d_inclination_deg = angle_between_vectors_3d(torso_vector, upward_vertical_3d)

    # 6. Head-to-torso 3D angle
    if valid_nose:
        head_torso_angle_3d_deg = angle_between_vectors_3d(head_vector, torso_vector)
        head_depth_offset_norm = float(nose_p[2] - shoulder_center[2])
        head_lateral_offset_norm = float(nose_p[0] - shoulder_center[0])
    else:
        head_torso_angle_3d_deg = float("nan")
        head_depth_offset_norm = float("nan")
        head_lateral_offset_norm = float("nan")

    # 7. Depth asymmetries (transverse plane rotation / asymmetry)
    shoulder_depth_asymmetry_norm = float(r_sh[2] - l_sh[2])
    hip_depth_asymmetry_norm = float(r_hip[2] - l_hip[2])

    feat_dict = {
        "nose_x": float(nose_p[0]), "nose_y": float(nose_p[1]), "nose_z": float(nose_p[2]),
        "left_shoulder_x": float(l_sh[0]), "left_shoulder_y": float(l_sh[1]), "left_shoulder_z": float(l_sh[2]),
        "right_shoulder_x": float(r_sh[0]), "right_shoulder_y": float(r_sh[1]), "right_shoulder_z": float(r_sh[2]),
        "left_hip_x": float(l_hip[0]), "left_hip_y": float(l_hip[1]), "left_hip_z": float(l_hip[2]),
        "right_hip_x": float(r_hip[0]), "right_hip_y": float(r_hip[1]), "right_hip_z": float(r_hip[2]),
        "shoulder_roll_deg": shoulder_roll_deg,
        "hip_roll_deg": hip_roll_deg,
        "torso_lateral_lean_deg": torso_lateral_lean_deg,
        "torso_sagittal_lean_deg": torso_sagittal_lean_deg,
        "torso_3d_inclination_deg": torso_3d_inclination_deg,
        "head_torso_angle_3d_deg": head_torso_angle_3d_deg,
        "head_depth_offset_norm": head_depth_offset_norm,
        "head_lateral_offset_norm": head_lateral_offset_norm,
        "shoulder_depth_asymmetry_norm": shoulder_depth_asymmetry_norm,
        "hip_depth_asymmetry_norm": hip_depth_asymmetry_norm
    }

    return feat_dict, True, "Success"
