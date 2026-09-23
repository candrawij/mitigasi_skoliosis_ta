"""
canonicalize_private_3d_pose.py — 3D Pose Canonicalization for Private Dataset

Purpose:
    Remove rig-specific coordinate frame orientation from 3D poses.
    After canonicalization, all subjects share a common body-aligned reference frame
    regardless of the physical orientation of the stereo camera rig.

Canonical Reference Frame:
    - Origin: Hip center (midpoint of left_hip and right_hip)
    - Y_canonical: Points FROM hip_center TOWARD shoulder_center (torso upward direction)
    - X_canonical: Points from LEFT hip to RIGHT hip (subject's right side),
                   orthogonalized to be perpendicular to Y_canonical
    - Z_canonical: Cross product of X_canonical and Y_canonical
                   (points FORWARD from subject's front)

Scale:
    Preserved from raw normalization (S3 = max core joint distance from hip_center).
    No re-scaling after rotation to keep comparability.

Critical: Canonicalization is determined PURELY by geometry (joint positions),
NOT by class labels or prediction outcomes. This prevents data leakage.

Inputs:
    - 17 COCO 3D keypoints (meters, from JSON) per capture
    - Mandatory core joints: BOTH hips and BOTH shoulders must be non-NaN

Output:
    A function: canonicalize_3d_pose(keypoints_3d_m) -> canonical_keypoints_3d
    that can be called from extract_private_3d_features_canonical.py

Left/Right Semantics Preservation:
    X_canonical points from subject-left to subject-right:
    - We build X from (right_hip - left_hip) — the subject's anatomical right.
    - This convention is BODY-CENTERED, not camera-centered.
    - Verified: after canonicalization, right_hip.x > left_hip.x always.

Usage:
    from canonicalize_private_3d_pose import canonicalize_3d_pose
"""

import numpy as np
from typing import Optional, Tuple

# COCO-17 Keypoint Indices
COCO_NOSE = 0
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
MANDATORY_KP_INDICES = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]


def canonicalize_3d_pose(
    keypoints_3d_m: np.ndarray,
    eps: float = 1e-6
) -> Tuple[Optional[np.ndarray], bool, str]:
    """
    Canonicalize 3D pose by rotating into body-aligned coordinate frame.

    The canonical frame is determined SOLELY by joint geometry:
      - Y_can = direction from hip_center to shoulder_center (body upward)
      - X_can = direction from left_hip to right_hip, orthogonalized vs Y_can (body right)
      - Z_can = X_can x Y_can (body forward)

    This removes the dependency on camera rig physical orientation.

    Args:
        keypoints_3d_m: np.ndarray of shape (17, 3), in meters, CAM01 frame.
                        May contain NaN for non-core joints.

    Returns:
        (canonical_kpts, is_valid, reason)
        canonical_kpts: np.ndarray (17, 3), canonicalized and normalized
                        (hip-centered, scale-normalized, rotation-applied)
                        Non-core joints with NaN remain NaN.
        is_valid: bool
        reason: str describing outcome or failure reason
    """
    kpts = np.array(keypoints_3d_m, dtype=np.float64)

    if kpts.shape != (17, 3):
        return None, False, f"Invalid shape {kpts.shape}, expected (17, 3)"

    # Step 1: Validate mandatory joints
    for idx in MANDATORY_KP_INDICES:
        if np.isnan(kpts[idx]).any():
            return None, False, f"Mandatory joint {idx} is NaN — cannot canonicalize"

    l_sh = kpts[COCO_LEFT_SHOULDER]
    r_sh = kpts[COCO_RIGHT_SHOULDER]
    l_hip = kpts[COCO_LEFT_HIP]
    r_hip = kpts[COCO_RIGHT_HIP]

    # Step 2: Compute canonical axes from body geometry
    hip_center = (l_hip + r_hip) / 2.0
    shoulder_center = (l_sh + r_sh) / 2.0

    # Y_can: from hip_center toward shoulder_center (body upward direction)
    y_can_raw = shoulder_center - hip_center
    y_can_norm = np.linalg.norm(y_can_raw)
    if y_can_norm < eps:
        return None, False, f"Degenerate torso vector (norm={y_can_norm:.6f})"
    y_can = y_can_raw / y_can_norm

    # X_can: from left_hip toward right_hip (subject's anatomical right),
    #         then orthogonalized relative to Y_can via Gram-Schmidt
    x_raw = r_hip - l_hip
    x_raw_norm = np.linalg.norm(x_raw)
    if x_raw_norm < eps:
        return None, False, f"Degenerate hip-width vector (norm={x_raw_norm:.6f})"
    x_raw = x_raw / x_raw_norm

    # Gram-Schmidt orthogonalization: remove Y component from X
    x_can = x_raw - np.dot(x_raw, y_can) * y_can
    x_can_norm = np.linalg.norm(x_can)
    if x_can_norm < eps:
        return None, False, f"Degenerate X_can after orthogonalization (norm={x_can_norm:.6f})"
    x_can = x_can / x_can_norm

    # Z_can: cross product to complete right-handed frame
    # Z = X x Y (points forward from subject in anatomical convention)
    z_can = np.cross(x_can, y_can)
    z_can_norm = np.linalg.norm(z_can)
    if z_can_norm < eps:
        return None, False, f"Degenerate Z_can (norm={z_can_norm:.6f})"
    z_can = z_can / z_can_norm

    # Rotation matrix R: rows are the canonical axes
    # To rotate from CAM01 frame to canonical frame: v_can = R @ v_cam01
    R = np.stack([x_can, y_can, z_can], axis=0)  # shape (3, 3)

    # Step 3: Translate all joints to hip-centered coordinates (CAM01 frame)
    centered = kpts - hip_center  # (17, 3)

    # Step 4: Apply rotation to all joints
    # For joints with NaN, rotation would propagate NaN — that's correct behavior
    canonical = (R @ centered.T).T  # (17, 3)

    # Step 5: Scale normalization (same as extract_3d_features)
    # Use max Euclidean distance from origin (hip_center) among core mandatory joints
    valid_nose = not np.isnan(kpts[COCO_NOSE]).any()
    core_valid_indices = list(MANDATORY_KP_INDICES)
    if valid_nose:
        core_valid_indices.append(COCO_NOSE)

    distances = [np.linalg.norm(canonical[idx]) for idx in core_valid_indices]
    scale_s3 = max(distances)

    if scale_s3 < eps or np.isnan(scale_s3):
        return None, False, f"Invalid scale S3={scale_s3:.6f}"

    canonical_normalized = canonical / scale_s3

    return canonical_normalized, True, "Canonicalization successful"


def verify_semantics(canonical_kpts: np.ndarray) -> dict:
    """
    Verify that left/right semantics are preserved after canonicalization.

    Expected properties in canonical frame:
      - right_hip.x > left_hip.x (X points to subject's right)
      - right_shoulder.x > left_shoulder.x
      - shoulder_center.y > 0 (Y points upward from hip)

    Returns a dict of check results.
    """
    l_sh = canonical_kpts[COCO_LEFT_SHOULDER]
    r_sh = canonical_kpts[COCO_RIGHT_SHOULDER]
    l_hip = canonical_kpts[COCO_LEFT_HIP]
    r_hip = canonical_kpts[COCO_RIGHT_HIP]
    shoulder_center = (l_sh + r_sh) / 2.0

    return {
        "right_hip_x_gt_left_hip_x": bool(r_hip[0] > l_hip[0]),
        "right_shoulder_x_gt_left_shoulder_x": bool(r_sh[0] > l_sh[0]),
        "shoulder_center_y_positive": bool(shoulder_center[1] > 0),
        "hip_center_at_origin": bool(np.allclose((l_hip + r_hip) / 2.0, [0, 0, 0], atol=1e-6)),
    }
