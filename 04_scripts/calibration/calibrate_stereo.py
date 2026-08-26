"""
T5.B.3: Dual-Camera Stereo Calibration Script.

Features:
  1. Synchronized dual-view checkerboard corner matching.
  2. Computes Extrinsics: Rotation (R), Translation (T), Essential (E), Fundamental (F).
  3. Computes Stereo Rectification: R1, R2, P1, P2, and Disparity-to-Depth Matrix (Q).
  4. Decoupled calibration_id tracking for rig configurations (e.g. CAL_001).
  5. Exports full stereo calibration profile to 02_data/private_calibration/stereo/.

Usage:
  python 04_scripts/calibration/calibrate_stereo.py --calibration_id CAL_001 \
      --cam01_dir 02_data/private_calibration/raw_frames/CAM01 \
      --cam02_dir 02_data/private_calibration/raw_frames/CAM02 \
      --cam01_intrinsic 02_data/private_calibration/intrinsic/CAM01_intrinsic.json \
      --cam02_intrinsic 02_data/private_calibration/intrinsic/CAM02_intrinsic.json
"""
import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEREO_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "stereo"


def load_intrinsic_json(json_path):
    """Load K and D from an intrinsic JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix_K"]["raw_3x3"], dtype=np.float64)
    D = np.array(data["distortion_coefficients"]["raw_vector"], dtype=np.float64)
    img_shape = (data["image_width"], data["image_height"])
    return K, D, img_shape


def calibrate_stereo(
    cam01_image_paths,
    cam02_image_paths,
    cam01_intrinsic_file=None,
    cam02_intrinsic_file=None,
    cols=9,
    rows=6,
    square_size_m=0.025,
    calibration_id="CAL_001"
):
    STEREO_DIR.mkdir(parents=True, exist_ok=True)
    
    # 3D world coordinates
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    matched_pairs = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_shape = None

    print(f"\n{'='*60}")
    print(f"  STEREO CALIBRATION: {calibration_id}")
    print(f"{'='*60}")
    print(f"Cam01 pairs: {len(cam01_image_paths)}, Cam02 pairs: {len(cam02_image_paths)}")

    # Sort to ensure matching pairs
    cam01_sorted = sorted(cam01_image_paths, key=lambda x: Path(x).stem)
    cam02_sorted = sorted(cam02_image_paths, key=lambda x: Path(x).stem)

    n_pairs = min(len(cam01_sorted), len(cam02_sorted))

    for i in range(n_pairs):
        p1, p2 = cam01_sorted[i], cam02_sorted[i]
        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))

        if img1 is None or img2 is None:
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray1.shape[::-1]

        ret1, corners1 = cv2.findChessboardCorners(gray1, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret1 and ret2:
            objpoints.append(objp)
            c1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            c2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(c1)
            imgpoints_right.append(c2)
            matched_pairs.append((str(p1), str(p2)))
            print(f"  [MATCHED] Pair #{len(objpoints)}: {Path(p1).name} <-> {Path(p2).name}")
        else:
            print(f"  [UNMATCHED] Corners missing in pair: {Path(p1).name} (Cam1={ret1}, Cam2={ret2})")

    if len(objpoints) < 5:
        print(f"\n[ERROR] Found only {len(objpoints)} synchronized pairs. Need at least 5-10 pairs for stereo calibration!")
        return None

    # Load initial intrinsics if provided
    flags = 0
    if cam01_intrinsic_file and cam02_intrinsic_file:
        K1, D1, _ = load_intrinsic_json(cam01_intrinsic_file)
        K2, D2, _ = load_intrinsic_json(cam02_intrinsic_file)
        flags |= cv2.CALIB_FIX_INTRINSIC
        print("Using pre-computed intrinsics with CALIB_FIX_INTRINSIC flag.")
    else:
        # Calibrate intrinsics individually first
        _, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
        _, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

    # Perform stereo calibration
    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K1, D1, K2, D2, img_shape,
        criteria=stereo_criteria, flags=flags
    )

    # Compute stereo rectification
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_shape, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # Calculate baseline distance in meters & mm
    baseline_m = float(np.linalg.norm(T))
    baseline_mm = baseline_m * 1000.0

    # Evaluate Reprojection Error against QC guideline
    mean_stereo_error = float(round(ret, 4))
    if mean_stereo_error < 1.0:
        qc_status = "Target (<1 px) - Excellent"
    elif mean_stereo_error <= 2.0:
        qc_status = "Acceptable (1-2 px) - Good"
    else:
        qc_status = "Warning (>2 px) - Re-check stereo sync/frames"

    stereo_results = {
        "calibration_id": calibration_id,
        "calibration_timestamp": datetime.now().isoformat(),
        "camera_left_id": "CAM01",
        "camera_right_id": "CAM02",
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "stereo_quality": {
            "mean_reprojection_error_px": mean_stereo_error,
            "qc_status": qc_status,
            "matched_pairs_count": len(objpoints),
            "baseline_distance_m": round(baseline_m, 4),
            "baseline_distance_mm": round(baseline_mm, 2)
        },
        "extrinsics": {
            "rotation_matrix_R": R.tolist(),
            "translation_vector_T": T.tolist(),
            "essential_matrix_E": E.tolist(),
            "fundamental_matrix_F": F.tolist()
        },
        "intrinsics_refined": {
            "K1": K1.tolist(),
            "D1": D1.tolist(),
            "K2": K2.tolist(),
            "D2": D2.tolist()
        },
        "rectification": {
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q_disparity_to_depth": Q.tolist()
        }
    }

    out_file = STEREO_DIR / f"{calibration_id}_stereo.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stereo_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  STEREO CALIBRATION SUMMARY: {calibration_id}")
    print(f"{'='*60}")
    print(f"  Stereo Reprojection Error: {mean_stereo_error:.4f} px [{qc_status}]")
    print(f"  Camera Baseline Distance:  {baseline_mm:.2f} mm ({baseline_m:.3f} m)")
    print(f"  Translation Vector T:      [{T[0,0]:.3f}, {T[1,0]:.3f}, {T[2,0]:.3f}] m")
    print(f"  Saved Stereo JSON:         {out_file}")

    return stereo_results


def main():
    parser = argparse.ArgumentParser(description="Dual Camera Stereo Calibration")
    parser.add_argument("--calibration_id", type=str, default="CAL_001", help="Rig calibration configuration ID")
    parser.add_argument("--cam01_dir", type=str, required=False, help="Folder containing CAM01 frames")
    parser.add_argument("--cam02_dir", type=str, required=False, help="Folder containing CAM02 frames")
    parser.add_argument("--cam01_intrinsic", type=str, default=None, help="Path to CAM01 intrinsic JSON")
    parser.add_argument("--cam02_intrinsic", type=str, default=None, help="Path to CAM02 intrinsic JSON")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners horizontal")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners vertical")
    parser.add_argument("--square_size", type=float, default=0.025, help="Square size in meters")

    args = parser.parse_args()

    if args.cam01_dir and args.cam02_dir:
        p1 = list(Path(args.cam01_dir).glob("*.jpg")) + list(Path(args.cam01_dir).glob("*.png"))
        p2 = list(Path(args.cam02_dir).glob("*.jpg")) + list(Path(args.cam02_dir).glob("*.png"))
        calibrate_stereo(
            p1, p2,
            cam01_intrinsic_file=args.cam01_intrinsic,
            cam02_intrinsic_file=args.cam02_intrinsic,
            cols=args.cols, rows=args.rows,
            square_size_m=args.square_size,
            calibration_id=args.calibration_id
        )
    else:
        print("Stereo Calibration script ready. Run with --cam01_dir and --cam02_dir.")


if __name__ == "__main__":
    main()
