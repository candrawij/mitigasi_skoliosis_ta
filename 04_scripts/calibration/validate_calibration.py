"""
T5.B.5: Calibration Validation & Stereo Verification Script.

Features:
  1. Epipolar line alignment verification (draws epilines across rectified stereo pairs).
  2. 3D Triangulation testing: projects 2D corresponding points to 3D world coordinates (X, Y, Z in meters).
  3. Evaluates error metrics:
       - CAM01 Intrinsic Reprojection Error
       - CAM02 Intrinsic Reprojection Error
       - Stereo Joint Reprojection Error
       - Baseline Distance (mm and m)
  4. Validates reprojection error against QC guidelines:
       - < 1.0 px: Target Quality (Excellent)
       - 1.0 - 2.0 px: Acceptable Quality (Good)
       - > 3.0 px: Investigation Needed (Re-check frames/sync)
  5. Saves visual QC proof image and JSON validation report to 02_data/private_calibration/logs/.

Usage:
  python 04_scripts/calibration/validate_calibration.py --calibration_id CAL_001 \
      --cam01_sample 02_data/private_calibration/raw_frames/CAM01/sample.jpg \
      --cam02_sample 02_data/private_calibration/raw_frames/CAM02/sample.jpg
"""
import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIB_LOGS_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "logs"
STEREO_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "stereo"
INTRINSIC_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "intrinsic"


def triangulate_2d_to_3d(pt1, pt2, P1, P2):
    """Triangulate a 2D point pair into 3D world coordinate (X, Y, Z in meters).
    pt1: (x, y) on CAM01
    pt2: (x, y) on CAM02
    P1, P2: 3x4 projection matrices from stereo rectification
    """
    pts1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    pts2 = np.array([[pt2[0]], [pt2[1]]], dtype=np.float64)
    
    pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    # Normalize homogeneous coordinates
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.flatten()


def draw_horizontal_epilines(rect1, rect2, line_interval=30):
    """Draw horizontal guide lines across rectified stereo pair to verify row alignment."""
    h, w = rect1.shape[:2]
    canvas = np.hstack((rect1, rect2))
    
    for y in range(line_interval, h, line_interval):
        # Alternate colors for visibility
        color = (0, 255, 0) if (y // line_interval) % 2 == 0 else (0, 200, 255)
        cv2.line(canvas, (0, y), (2 * w, y), color, 1)
        
    cv2.putText(canvas, "CAM01 (Rectified Frontal)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(canvas, "CAM02 (Rectified Lateral)", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(canvas, (w, 0), (w, h), (255, 255, 255), 2)
    return canvas


def validate_calibration(calibration_id="CAL_001", cam01_sample_path=None, cam02_sample_path=None):
    CALIB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stereo_file = STEREO_DIR / f"{calibration_id}_stereo.json"
    
    if not stereo_file.exists():
        print(f"[ERROR] Stereo calibration file not found: {stereo_file}")
        return None

    with open(stereo_file, "r", encoding="utf-8") as f:
        calib = json.load(f)

    # Load individual intrinsic errors if available
    cam01_error = None
    cam02_error = None
    cam01_file = INTRINSIC_DIR / "CAM01_intrinsic.json"
    cam02_file = INTRINSIC_DIR / "CAM02_intrinsic.json"
    
    if cam01_file.exists():
        with open(cam01_file, "r", encoding="utf-8") as f:
            c1_data = json.load(f)
            cam01_error = c1_data.get("quality_metrics", {}).get("mean_reprojection_error_px")
            
    if cam02_file.exists():
        with open(cam02_file, "r", encoding="utf-8") as f:
            c2_data = json.load(f)
            cam02_error = c2_data.get("quality_metrics", {}).get("mean_reprojection_error_px")

    stereo_error_px = calib["stereo_quality"]["mean_reprojection_error_px"]
    baseline_mm = calib["stereo_quality"]["baseline_distance_mm"]
    baseline_m = calib["stereo_quality"]["baseline_distance_m"]
    
    # QC evaluation
    if stereo_error_px < 1.0:
        qc_verdict = "PASSED_TARGET"
        qc_comment = "Reprojection error < 1.0 px: Calibration is optimal for 3D posture reconstruction."
    elif stereo_error_px <= 2.0:
        qc_verdict = "PASSED_ACCEPTABLE"
        qc_comment = "Reprojection error 1.0 - 2.0 px: Calibration is acceptable for posture classification."
    else:
        qc_verdict = "WARNING_RECHECK"
        qc_comment = "Reprojection error > 2.0 px: Frame synchronization or corner detection may have noise."

    print(f"\n{'='*70}")
    print(f"  CALIBRATION VALIDATION REPORT: {calibration_id}")
    print(f"{'='*70}")
    print(f"  CAM01 Reprojection Error:    {f'{cam01_error:.4f} px' if cam01_error is not None else 'N/A'}")
    print(f"  CAM02 Reprojection Error:    {f'{cam02_error:.4f} px' if cam02_error is not None else 'N/A'}")
    print(f"  Stereo Joint Error:          {stereo_error_px:.4f} px")
    print(f"  Camera Baseline Distance:    {baseline_mm:.2f} mm ({baseline_m:.3f} m)")
    print(f"  QC Status:                   [{qc_verdict}]")
    print(f"  QC Evaluation:               {qc_comment}")

    report = {
        "calibration_id": calibration_id,
        "validation_timestamp": datetime.now().isoformat(),
        "metrics": {
            "cam01_reprojection_error_px": cam01_error,
            "cam02_reprojection_error_px": cam02_error,
            "stereo_reprojection_error_px": stereo_error_px,
            "baseline_distance_mm": baseline_mm,
            "baseline_distance_m": baseline_m
        },
        "qc_status": qc_verdict,
        "qc_comment": qc_comment,
        "qc_guidelines": {
            "target_threshold_px": 1.0,
            "acceptable_threshold_px": 2.0,
            "review_threshold_px": 3.0
        }
    }

    # Test Triangulation & Rectification if sample images are provided
    if cam01_sample_path and cam02_sample_path:
        img1 = cv2.imread(str(cam01_sample_path))
        img2 = cv2.imread(str(cam02_sample_path))
        
        if img1 is not None and img2 is not None:
            K1 = np.array(calib["intrinsics_refined"]["K1"], dtype=np.float64)
            D1 = np.array(calib["intrinsics_refined"]["D1"], dtype=np.float64)
            K2 = np.array(calib["intrinsics_refined"]["K2"], dtype=np.float64)
            D2 = np.array(calib["intrinsics_refined"]["D2"], dtype=np.float64)
            R1 = np.array(calib["rectification"]["R1"], dtype=np.float64)
            R2 = np.array(calib["rectification"]["R2"], dtype=np.float64)
            P1 = np.array(calib["rectification"]["P1"], dtype=np.float64)
            P2 = np.array(calib["rectification"]["P2"], dtype=np.float64)
            
            h, w = img1.shape[:2]
            
            # Rectify images
            map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)
            rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
            rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
            
            # Draw epipolar guide lines
            epipolar_img = draw_horizontal_epilines(rect1, rect2, line_interval=40)
            epipolar_out_path = CALIB_LOGS_DIR / f"{calibration_id}_rectification_epipolar_check.jpg"
            cv2.imwrite(str(epipolar_out_path), epipolar_img)
            print(f"\n  [Visual Verification]")
            print(f"  Epipolar Lines Check Image:  {epipolar_out_path}")
            report["epipolar_check_image"] = str(epipolar_out_path.name)
            
            # 3D Triangulation on center point & shoulder sample
            pt_center1 = (w/2, h/2)
            pt_center2 = (w/2, h/2)
            pt3D_center = triangulate_2d_to_3d(pt_center1, pt_center2, P1, P2)
            
            report["triangulation_tests"] = {
                "center_point": {
                    "pt1_2d": pt_center1,
                    "pt2_2d": pt_center2,
                    "triangulated_3d_m": {
                        "X": float(round(pt3D_center[0], 4)),
                        "Y": float(round(pt3D_center[1], 4)),
                        "Z_depth": float(round(pt3D_center[2], 4))
                    }
                }
            }
            print(f"\n  [3D Triangulation Sanity Test]")
            print(f"  2D Image Center -> 3D Coordinate: X = {pt3D_center[0]:.3f} m, Y = {pt3D_center[1]:.3f} m, Z (Depth) = {pt3D_center[2]:.3f} m")

    # Save log report
    report_file = CALIB_LOGS_DIR / f"{calibration_id}_validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Validation report saved to: {report_file}")
    print(f"{'='*70}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Stereo Calibration Validator")
    parser.add_argument("--calibration_id", type=str, default="CAL_001", help="Rig calibration configuration ID")
    parser.add_argument("--cam01_sample", type=str, default=None, help="Sample image path from CAM01")
    parser.add_argument("--cam02_sample", type=str, default=None, help="Sample image path from CAM02")

    args = parser.parse_args()
    validate_calibration(
        calibration_id=args.calibration_id,
        cam01_sample_path=args.cam01_sample,
        cam02_sample_path=args.cam02_sample
    )


if __name__ == "__main__":
    main()
