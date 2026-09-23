"""
extract_private_3d_features_canonical.py
=========================================
Extract Normalized Stereo 3D Features in CANONICAL BODY-ALIGNED reference frame.

The canonical frame (from canonicalize_private_3d_pose.py) has:
  - Origin: hip_center
  - Y_can: pointing FROM hip_center TOWARD shoulder_center (body "up")
  - X_can: pointing from subject-left to subject-right (body anatomical right)
  - Z_can: pointing FORWARD from subject (right-hand rule: X_can x Y_can)

Feature Schema (CANONICAL, 22 non-degenerate features):
  After empirical analysis, some features from the original 25-feature schema
  become constants in canonical frame (zero variance) and are replaced:

  DEGENERATE (removed, all=0 by construction):
    - torso_lateral_lean_deg   (torso IS the Y-axis)
    - torso_sagittal_lean_deg  (torso IS the Y-axis)
    - hip_depth_asymmetry_norm (hips used to define X-axis, so Z always = 0)
    - [shoulder_center Z, X are implicitly 0]

  REPLACED WITH (body-relative, meaningful in canonical frame):
    - torso_height_norm        (shoulder_center.y = normalized torso height)
    - shoulder_width_norm      (|r_sh - l_sh|, physical width)
    - shoulder_depth_spread    (r_sh.z - l_sh.z = transverse rotation proxy)
    - head_z_rel               (nose.z - shoulder_center.z = forward head position)
    - head_y_rel               (nose.y - shoulder_center.y = relative head height)

  RETAINED:
    - 15 normalized coordinates (nose xyz, l_sh xyz, r_sh xyz, l_hip xyz, r_hip xyz)
    - shoulder_roll_deg        (shoulder line roll in XY plane, still informative)
    - hip_roll_deg             (hip line roll in XY plane, still informative)
    - torso_3d_inclination_deg (angle vs Y, ~0 for upright — now meaningful)
    - head_torso_angle_3d_deg  (angle between head and torso vectors)
    - head_depth_offset_norm   (nose.z - shoulder_center.z)
    - head_lateral_offset_norm (nose.x - shoulder_center.x)
    - shoulder_depth_asymmetry_norm (still informative — asymmetric shoulder rotation)

  Total: 15 coords + 7 geometry = 22 features
  (Note: 3 degenerate features removed from original 25)
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))

from private_feature_common import (
    COCO_NOSE, COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER,
    COCO_LEFT_HIP, COCO_RIGHT_HIP, MANDATORY_KP_INDICES,
    validate_core_keypoints_3d, angle_between_vectors_3d
)
from canonicalize_private_3d_pose import canonicalize_3d_pose

MANIFESTS_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "manifests"
FEATURES_DIR  = PROJECT_ROOT / "02_data" / "private_processed" / "features"
AUDIT_DIR     = PROJECT_ROOT / "02_data" / "private_processed" / "audit"
ANNOT_3D_DIR  = PROJECT_ROOT / "02_data" / "private_annotations" / "keypoints_3d"
RESULTS_DIR   = PROJECT_ROOT / "07_results" / "private_audit"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical feature schema (22 features)
FEATURE_NAMES_3D_CANONICAL = [
    # 15 normalized joint coordinates (same joints as original 3D schema)
    "nose_x", "nose_y", "nose_z",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
    "left_hip_x", "left_hip_y", "left_hip_z",
    "right_hip_x", "right_hip_y", "right_hip_z",
    # 7 geometry features (adapted for canonical frame)
    "shoulder_roll_deg",              # shoulder line tilt (XY plane)
    "hip_roll_deg",                   # hip line tilt (XY plane)
    "torso_3d_inclination_deg",       # angle of torso vs body-up [0,1,0] (should be near-0 for upright)
    "head_torso_angle_3d_deg",        # angle between head vector and torso vector
    "head_depth_offset_norm",         # nose.z - shoulder_center.z (forward head position)
    "head_lateral_offset_norm",       # nose.x - shoulder_center.x (lateral head shift)
    "shoulder_depth_asymmetry_norm",  # r_sh.z - l_sh.z (transverse shoulder rotation)
]


def extract_3d_features_canonical(
    keypoints_canonical: np.ndarray,
    eps: float = 1e-6
):
    """
    Extract 22 features from CANONICALIZED 3D pose.
    Input: already hip-centered, rotated to body frame, scale-normalized.
    """
    kpts = np.array(keypoints_canonical, dtype=np.float64)

    is_valid, reason, valid_nose = validate_core_keypoints_3d(kpts)
    if not is_valid:
        return None, False, reason

    nose_p = kpts[COCO_NOSE] if valid_nose else np.full(3, np.nan)
    l_sh   = kpts[COCO_LEFT_SHOULDER]
    r_sh   = kpts[COCO_RIGHT_SHOULDER]
    l_hip  = kpts[COCO_LEFT_HIP]
    r_hip  = kpts[COCO_RIGHT_HIP]

    sh_center  = (l_sh + r_sh) / 2.0
    hip_center = (l_hip + r_hip) / 2.0   # ~[0,0,0]
    torso_vec  = sh_center - hip_center   # ~[0, ~sh_y, ~0]
    head_vec   = (nose_p - sh_center) if valid_nose else np.full(3, np.nan)

    # In canonical frame, "body upward" = +Y = [0, 1, 0]
    upward_canonical = np.array([0.0, 1.0, 0.0])

    # 1. Shoulder roll in XY plane
    dx_sh = r_sh[0] - l_sh[0]; dy_sh = r_sh[1] - l_sh[1]
    shoulder_roll_deg = float(np.degrees(np.arctan2(dy_sh, dx_sh)))

    # 2. Hip roll in XY plane
    dx_hip = r_hip[0] - l_hip[0]; dy_hip = r_hip[1] - l_hip[1]
    hip_roll_deg = float(np.degrees(np.arctan2(dy_hip, dx_hip)))

    # 3. Torso 3D inclination: angle vs +Y (body-up)
    # For perfectly upright canonical pose: torso_vec = [0, sh_y, 0] ≈ [0, 1, 0] → ~0°
    # For leaning: torso_vec deviates → >0°
    torso_3d_inclination_deg = angle_between_vectors_3d(torso_vec, upward_canonical)

    # 4. Head-to-torso angle
    if valid_nose:
        head_torso_angle_3d_deg  = angle_between_vectors_3d(head_vec, torso_vec)
        head_depth_offset_norm   = float(nose_p[2] - sh_center[2])
        head_lateral_offset_norm = float(nose_p[0] - sh_center[0])
    else:
        head_torso_angle_3d_deg  = float("nan")
        head_depth_offset_norm   = float("nan")
        head_lateral_offset_norm = float("nan")

    # 5. Shoulder depth asymmetry (transverse rotation; not degenerate in canonical)
    shoulder_depth_asymmetry_norm = float(r_sh[2] - l_sh[2])

    feat_dict = {
        "nose_x": float(nose_p[0]), "nose_y": float(nose_p[1]), "nose_z": float(nose_p[2]),
        "left_shoulder_x":  float(l_sh[0]),  "left_shoulder_y":  float(l_sh[1]),  "left_shoulder_z":  float(l_sh[2]),
        "right_shoulder_x": float(r_sh[0]),  "right_shoulder_y": float(r_sh[1]),  "right_shoulder_z": float(r_sh[2]),
        "left_hip_x":  float(l_hip[0]),  "left_hip_y":  float(l_hip[1]),  "left_hip_z":  float(l_hip[2]),
        "right_hip_x": float(r_hip[0]),  "right_hip_y": float(r_hip[1]),  "right_hip_z": float(r_hip[2]),
        "shoulder_roll_deg":              shoulder_roll_deg,
        "hip_roll_deg":                   hip_roll_deg,
        "torso_3d_inclination_deg":       torso_3d_inclination_deg,
        "head_torso_angle_3d_deg":        head_torso_angle_3d_deg,
        "head_depth_offset_norm":         head_depth_offset_norm,
        "head_lateral_offset_norm":       head_lateral_offset_norm,
        "shoulder_depth_asymmetry_norm":  shoulder_depth_asymmetry_norm,
    }

    return feat_dict, True, "Success"


def load_3d_annotation(capture_id: str):
    fpath = ANNOT_3D_DIR / f"{capture_id}_3d_keypoints.json"
    if not fpath.exists():
        return None, False, f"File not found: {fpath.name}"
    try:
        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as e:
        return None, False, f"JSON parse error: {e}"
    kpts = np.array(data.get("keypoints_3d_m", []), dtype=np.float64)
    if kpts.shape != (17, 3):
        return None, False, f"Invalid shape {kpts.shape}"
    return kpts, True, "OK"


def run_extraction():
    print("=" * 80)
    print("  EXTRACT 22 CANONICAL 3D FEATURES (403 USABLE CAPTURES)")
    print("=" * 80)
    print(f"  Features: {len(FEATURE_NAMES_3D_CANONICAL)} ({', '.join(FEATURE_NAMES_3D_CANONICAL[:5])} ...)")

    manifest_file = MANIFESTS_DIR / "private_6class_3d_manifest.csv"
    df_manifest = pd.read_csv(manifest_file)
    print(f"\nLoaded manifest: {len(df_manifest)} captures")

    usable_decisions = {"INCLUDE_3D_FULL", "INCLUDE_3D_WITH_MASKING"}
    stats = {"USABLE": 0, "EXCLUDED": 0, "INVALID_GEOMETRY": 0, "CANON_FAILED": 0, "NO_FILE": 0}

    rows = []
    for _, mrow in df_manifest.iterrows():
        cap_id   = mrow["capture_id"]
        label    = mrow["label"]
        decision = mrow.get("decision", "")

        if decision not in usable_decisions:
            stats["EXCLUDED"] += 1
            rows.append({"capture_id": cap_id, "label": label, "status_3d": "EXCLUDED", "reason": decision,
                         **{f: float("nan") for f in FEATURE_NAMES_3D_CANONICAL}})
            continue

        kpts_raw, ok, reason = load_3d_annotation(cap_id)
        if not ok:
            stats["NO_FILE"] += 1
            rows.append({"capture_id": cap_id, "label": label, "status_3d": "NO_FILE", "reason": reason,
                         **{f: float("nan") for f in FEATURE_NAMES_3D_CANONICAL}})
            continue

        canon_kpts, canon_ok, canon_reason = canonicalize_3d_pose(kpts_raw)
        if not canon_ok:
            stats["CANON_FAILED"] += 1
            rows.append({"capture_id": cap_id, "label": label, "status_3d": "CANON_FAILED", "reason": canon_reason,
                         **{f: float("nan") for f in FEATURE_NAMES_3D_CANONICAL}})
            continue

        feat, feat_ok, feat_reason = extract_3d_features_canonical(canon_kpts)
        if not feat_ok:
            stats["INVALID_GEOMETRY"] += 1
            rows.append({"capture_id": cap_id, "label": label, "status_3d": "INVALID_3D_GEOMETRY", "reason": feat_reason,
                         **{f: float("nan") for f in FEATURE_NAMES_3D_CANONICAL}})
            continue

        stats["USABLE"] += 1
        row_data = {"capture_id": cap_id, "label": label, "status_3d": "USABLE", "reason": ""}
        row_data.update({f: feat[f] for f in FEATURE_NAMES_3D_CANONICAL})
        rows.append(row_data)

    df_out = pd.DataFrame(rows)
    out_path = FEATURES_DIR / "private_features_3d_canonical.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df_out)} rows)")

    print("\n--- Extraction Status ---")
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")

    usable_df = df_out[df_out["status_3d"] == "USABLE"]
    print(f"\nUSABLE canonical samples: {len(usable_df)}")

    print("\n--- Canonical Feature Statistics (USABLE) ---")
    angle_cols = [f for f in FEATURE_NAMES_3D_CANONICAL if "deg" in f or "asym" in f or "offset" in f]
    for col in angle_cols:
        vals = usable_df[col].dropna()
        print(f"  {col:35s}: mean={vals.mean():7.3f}  std={vals.std():6.3f}  "
              f"min={vals.min():7.3f}  max={vals.max():7.3f}  NaN={vals.isna().sum()}")

    print("\n--- Per-class torso_3d_inclination_deg (expecting small for upright) ---")
    for lbl in sorted(usable_df["label"].unique()):
        vals = usable_df[usable_df["label"] == lbl]["torso_3d_inclination_deg"].dropna()
        print(f"  {lbl:<22}: N={len(vals):3d}  mean={vals.mean():6.2f}  std={vals.std():5.2f}")

    print("\n--- Per-class head_depth_offset_norm (expecting positive for slouching) ---")
    for lbl in sorted(usable_df["label"].unique()):
        vals = usable_df[usable_df["label"] == lbl]["head_depth_offset_norm"].dropna()
        if len(vals) > 0:
            print(f"  {lbl:<22}: N={len(vals):3d}  mean={vals.mean():7.4f}  std={vals.std():6.4f}")

    return df_out


if __name__ == "__main__":
    run_extraction()
