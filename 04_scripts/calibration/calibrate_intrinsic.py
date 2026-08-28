"""
T5.B.2: Single Camera Intrinsic Calibration Script.

Features:
  1. Offline batch calibration from a folder of checkerboard images.
  2. Interactive live webcam/USB camera capture with real-time sub-pixel corner detection.
  3. Computes Camera Matrix (K), Distortion Coefficients (D), and Per-View Reprojection Error.
  4. Exports calibration parameters to JSON in 02_data/private_calibration/intrinsic/.

Usage:
  # From existing images:
  python 04_scripts/calibration/calibrate_intrinsic.py --camera_id CAM01 --input_dir 02_data/private_calibration/raw_frames/CAM01

  # Live capture from camera device index 0:
  python 04_scripts/calibration/calibrate_intrinsic.py --camera_id CAM01 --live --cam_idx 0
"""
import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "intrinsic"


def calibrate_camera_from_images(
    image_paths,
    cols=9,
    rows=6,
    square_size_m=0.025,
    camera_id="CAM01",
    save_undistort_sample=True
):
    """Calibrate camera intrinsic parameters from a list of checkerboard images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 3D world coordinates of checkerboard corners
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    valid_paths = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_shape = None

    print(f"\n--- Calibrating Intrinsic Parameters for {camera_id} ---")
    print(f"Total images provided: {len(image_paths)}")
    print(f"Grid: {cols}x{rows} inner corners, Square size: {square_size_m*1000:.1f} mm")

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_paths.append(str(p))
            print(f"  [FOUND] Corners in: {Path(p).name}")
        else:
            print(f"  [MISS]  Corners NOT found: {Path(p).name}")

    if len(objpoints) < 5:
        print(f"\n[ERROR] Only {len(objpoints)} valid images found. Need at least 5-10 images for accurate calibration!")
        return None

    print(f"\nRunning OpenCV camera calibration on {len(objpoints)} valid views...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Compute per-view reprojection error
    total_error = 0
    per_view_errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        pts1 = np.asarray(imgpoints[i], dtype=np.float32).reshape(-1, 2)
        pts2 = np.asarray(imgpoints2, dtype=np.float32).reshape(-1, 2)
        error = float(np.mean(np.linalg.norm(pts1 - pts2, axis=1)))
        per_view_errors.append(float(round(error, 4)))
        total_error += error

    mean_error = float(round(total_error / len(objpoints), 4))

    # Evaluate against QC Guidelines
    if mean_error < 1.0:
        qc_status = "Target (<1 px) - Excellent"
    elif mean_error <= 2.0:
        qc_status = "Acceptable (1-2 px) - Good"
    else:
        qc_status = "Warning (>2 px) - Re-check frames"

    # Assemble calibration result dict
    calib_result = {
        "camera_id": camera_id,
        "calibration_timestamp": datetime.now().isoformat(),
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "camera_matrix_K": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "raw_3x3": K.tolist()
        },
        "distortion_coefficients": {
            "k1": float(dist[0, 0]),
            "k2": float(dist[0, 1]),
            "p1": float(dist[0, 2]),
            "p2": float(dist[0, 3]),
            "k3": float(dist[0, 4]) if dist.shape[1] > 4 else 0.0,
            "raw_vector": dist.tolist()[0]
        },
        "quality_metrics": {
            "mean_reprojection_error_px": mean_error,
            "qc_status": qc_status,
            "num_valid_views": len(objpoints),
            "per_view_errors_px": per_view_errors
        },
        "checkerboard_spec": {
            "inner_cols": cols,
            "inner_rows": rows,
            "square_size_m": square_size_m
        }
    }

    out_file = OUTPUT_DIR / f"{camera_id}_intrinsic.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calib_result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  CALIBRATION SUMMARY: {camera_id}")
    print(f"{'='*60}")
    print(f"  Focal Length:   fx = {K[0, 0]:.2f}, fy = {K[1, 1]:.2f} px")
    print(f"  Principal Point: cx = {K[0, 2]:.2f}, cy = {K[1, 2]:.2f} px")
    print(f"  Mean Reprojection Error: {mean_error:.4f} px [{qc_status}]")
    print(f"  Saved JSON:     {out_file}")

    # Optional: Save an undistorted verification sample
    if save_undistort_sample and valid_paths:
        sample_img = cv2.imread(valid_paths[0])
        undistorted = cv2.undistort(sample_img, K, dist, None, K)
        preview_path = OUTPUT_DIR / f"{camera_id}_undistort_preview.jpg"
        # Concatenate side by side
        comparison = np.hstack((sample_img, undistorted))
        cv2.imwrite(str(preview_path), comparison)
        print(f"  Saved Undistort Comparison: {preview_path}")

    return calib_result


def live_capture_calibration(camera_id="CAM01", cam_idx=0, cols=9, rows=6, square_size_m=0.025):
    """Interactive GUI for live checkerboard capture and instant calibration."""
    frames_dir = PROJECT_ROOT / "02_data" / "private_calibration" / "raw_frames" / camera_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing old calibration images
    old_imgs = list(frames_dir.glob(f"{camera_id}_calib_*.jpg"))
    if old_imgs:
        print(f">> Ditemukan {len(old_imgs)} foto kalibrasi lama untuk {camera_id}.")
        clean_old = input(f"   Apakah ingin MENGHAPUS foto lama dan mulai kalibrasi {camera_id} baru dari awal? (Y/n) [default: Y]: ").strip().lower()
        if clean_old != "n":
            for f in old_imgs: f.unlink()
            print(f"   [OK] Foto lama {camera_id} telah dibersihkan. Memulai dari calib_001.")

    captured_paths = []
    count = len(list(frames_dir.glob(f"{camera_id}_calib_*.jpg")))

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {cam_idx}")
        return

    print("\n--- Interactive Live Calibration Capture ---")
    print("Controls:")
    print("  [SPACE] / [C] : Capture current frame (if checkerboard detected)")
    print("  [ENTER] / [Q] : Finish capture and compute intrinsic calibration")
    print("  [ESC]         : Cancel\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            status_text = f"Pattern DETECTED! Press SPACE to capture (Captured: {count})"
            color = (0, 255, 0)
        else:
            status_text = f"Searching for {cols}x{rows} pattern... (Captured: {count})"
            color = (0, 165, 255)

        cv2.putText(display, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(f"Calibration Capture - {camera_id}", display)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord(' '), ord('c'), ord('C')]:
            if found:
                count += 1
                fname = frames_dir / f"{camera_id}_calib_{count:03d}.jpg"
                cv2.imwrite(str(fname), frame)
                captured_paths.append(fname)
                print(f"Captured view #{count}: {fname.name}")
            else:
                print("Cannot capture: checkerboard not detected clearly in current frame.")
        elif key in [13, ord('q'), ord('Q')]:  # ENTER or Q
            break
        elif key == 27:  # ESC
            print("Capture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if captured_paths:
        calibrate_camera_from_images(
            captured_paths, cols=cols, rows=rows,
            square_size_m=square_size_m, camera_id=camera_id
        )


def main():
    parser = argparse.ArgumentParser(description="Intrinsic Camera Calibration")
    parser.add_argument("--camera_id", type=str, default="CAM01", help="Camera identifier (CAM01, CAM02)")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing calibration checkerboard images")
    parser.add_argument("--live", action="store_true", help="Run interactive live webcam calibration")
    parser.add_argument("--cam_idx", type=int, default=0, help="Camera device index for live capture")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners horizontal")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners vertical")
    parser.add_argument("--square_size", type=float, default=0.025, help="Square size in meters (e.g. 0.025 = 25mm)")

    args = parser.parse_args()

    if args.live:
        live_capture_calibration(
            camera_id=args.camera_id, cam_idx=args.cam_idx,
            cols=args.cols, rows=args.rows, square_size_m=args.square_size
        )
    elif args.input_dir:
        in_path = Path(args.input_dir)
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        img_paths = []
        for ext in exts:
            img_paths.extend(list(in_path.glob(ext)))
        if not img_paths:
            print(f"[ERROR] No images found in {in_path}")
            return
        calibrate_camera_from_images(
            img_paths, cols=args.cols, rows=args.rows,
            square_size_m=args.square_size, camera_id=args.camera_id
        )
    else:
        # Default test check
        print("Calibrate Intrinsic Script Ready. Run with --live or --input_dir <path>.")


if __name__ == "__main__":
    main()
