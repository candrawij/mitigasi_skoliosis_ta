"""
infer_realtime_stereo_3d.py — Real-Time Dual-Camera Stereo 3D Posture Inference Prototype
Performs live synchronized pose tracking, stereo triangulation, Reject Gate validation,
spatial geometry calculation, and 3D XGBoost posture classification with HUD visualization.

Usage:
  # With physical dual webcams and calibration:
  python 04_scripts/inference/infer_realtime_stereo_3d.py --cam01-idx 0 --cam02-idx 1 --calibration CAL_001

  # Test / demo mode with image files (runs simulation loop without physical webcams):
  python 04_scripts/inference/infer_realtime_stereo_3d.py --test-mode --calibration CAL_001
"""

import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "inference"))
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))

from private_inference_common import (
    infer_pair_3d,
    detect_target_person_keypoints,
    load_stereo_calibration,
    triangulate_stereo_pair,
    MAIN_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    COCO_NOSE,
    COCO_LEFT_SHOULDER,
    COCO_RIGHT_SHOULDER,
    COCO_LEFT_HIP,
    COCO_RIGHT_HIP,
    FEATURE_NAMES_3D
)


class ThreadedCamera:
    """Thread-safe camera grabber for low-latency live streaming."""
    def __init__(self, cam_idx: int, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened() and sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_idx}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def draw_skeleton_overlay(image: np.ndarray, kpts: Optional[np.ndarray], color=(0, 255, 0)):
    """Draw essential torso and head skeleton keypoints on frame."""
    if kpts is None or len(kpts) < 17:
        return image

    for idx in [COCO_NOSE, COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]:
        pt = kpts[idx]
        if not np.isnan(pt[0]) and not np.isnan(pt[1]):
            cv2.circle(image, (int(pt[0]), int(pt[1])), 5, color, -1)
            cv2.circle(image, (int(pt[0]), int(pt[1])), 7, (255, 255, 255), 1)

    ls, rs = kpts[COCO_LEFT_SHOULDER], kpts[COCO_RIGHT_SHOULDER]
    lh, rh = kpts[COCO_LEFT_HIP], kpts[COCO_RIGHT_HIP]
    nose = kpts[COCO_NOSE]

    if not (np.isnan(ls[0]) or np.isnan(rs[0])):
        cv2.line(image, (int(ls[0]), int(ls[1])), (int(rs[0]), int(rs[1])), color, 2)
    if not (np.isnan(lh[0]) or np.isnan(rh[0])):
        cv2.line(image, (int(lh[0]), int(lh[1])), (int(rh[0]), int(rh[1])), color, 2)
    if not (np.isnan(ls[0]) or np.isnan(rs[0]) or np.isnan(lh[0]) or np.isnan(rh[0])):
        sh_c = ((ls + rs) / 2.0).astype(int)
        hip_c = ((lh + rh) / 2.0).astype(int)
        cv2.line(image, (sh_c[0], sh_c[1]), (hip_c[0], hip_c[1]), (255, 255, 0), 2)
        if not np.isnan(nose[0]):
            cv2.line(image, (int(nose[0]), int(nose[1])), (sh_c[0], sh_c[1]), (0, 255, 255), 2)

    return image


def build_realtime_hud_3d(
    frame_cam01: np.ndarray,
    frame_cam02: np.ndarray,
    result: dict,
    fps: float,
    cal_id: str
) -> np.ndarray:
    """Compose dual-camera split-screen display with modern 3D HUD overlay."""
    h1, w1 = frame_cam01.shape[:2]
    h2, w2 = frame_cam02.shape[:2]

    target_h = 480
    w1_r = int(w1 * (target_h / h1))
    w2_r = int(w2 * (target_h / h2))

    c1_resized = cv2.resize(frame_cam01, (w1_r, target_h))
    c2_resized = cv2.resize(frame_cam02, (w2_r, target_h))

    canvas = np.hstack([c1_resized, c2_resized])
    canvas_w = w1_r + w2_r

    # Top HUD Bar
    hud_h = 75
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_w, hud_h), (20, 20, 20), -1)
    # Bottom HUD Bar
    bot_h = 80
    cv2.rectangle(overlay, (0, target_h - bot_h), (canvas_w, target_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Headers
    cv2.putText(canvas, f"CAM01: FRONTAL | CAL: {cal_id}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, "CAM02: LATERAL STEREO", (w1_r + 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    status = result.get("status", "REJECTED")
    pred = result.get("prediction", "REJECT")
    conf = result.get("confidence", 0.0)
    reproj = result.get("reprojection_error", 0.0)

    if status == "VALID":
        if pred == "upright":
            badge_color = (50, 205, 50)
        elif "leaning" in pred:
            badge_color = (0, 165, 255)
        else:
            badge_color = (0, 215, 255)
        status_text = f"POSTURE: {pred.upper()}"
        conf_text = f"CONF: {conf*100:.1f}% | REPROJ: {reproj:.1f}px"
        qc_badge = "3D QC: PASS"
        qc_col = (0, 255, 0)
    else:
        badge_color = (50, 50, 220)
        status_text = "POSTURE: REJECT / INVALID_3D"
        conf_text = f"REASON: {result.get('reason', 'QC Error')[:30]}"
        qc_badge = "3D QC: REJECT"
        qc_col = (0, 0, 255)

    cv2.putText(canvas, status_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, badge_color, 2)
    cv2.putText(canvas, conf_text, (380, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
    cv2.putText(canvas, f"FPS: {fps:4.1f}", (canvas_w - 120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(canvas, qc_badge, (canvas_w - 170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, qc_col, 2)

    # Bottom HUD: Probabilities Bar Chart
    probs = result.get("probabilities", {})
    bar_x = 15
    bar_spacing = int((canvas_w - 30) / max(1, len(MAIN_CLASSES)))

    for i, c_name in enumerate(MAIN_CLASSES):
        p = probs.get(c_name, 0.0)
        bx = bar_x + i * bar_spacing
        by = target_h - 15
        bw = bar_spacing - 20

        lbl = c_name.replace("leaning_", "lean_")[:10]
        cv2.putText(canvas, f"{lbl}: {p*100:3.0f}%", (bx, by - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.rectangle(canvas, (bx, by - 18), (bx + bw, by - 6), (50, 50, 50), -1)
        fill_w = int(bw * p)
        fill_col = (0, 255, 0) if c_name == pred and status == "VALID" else (180, 180, 0)
        cv2.rectangle(canvas, (bx, by - 18), (bx + fill_w, by - 6), fill_col, -1)

    return canvas


def run_realtime_stereo_3d(
    cam01_idx: int = 0,
    cam02_idx: int = 1,
    calibration: str = "CAL_001",
    test_mode: bool = False,
    test_cam01: Optional[str] = None,
    test_cam02: Optional[str] = None
):
    print("=" * 80)
    print("  LAUNCHING REAL-TIME STEREO 3D POSTURE INFERENCE")
    print("=" * 80)
    print(f"CAM01 (Frontal) Index : {cam01_idx}")
    print(f"CAM02 (Lateral) Index : {cam02_idx}")
    print(f"Stereo Calibration    : {calibration}")
    print("Press [Q] or [ESC] on the preview window to exit.\n")

    sim_frames = None
    if test_mode or test_cam01:
        p1 = Path(test_cam01) if test_cam01 else PROJECT_ROOT / "02_data/private_raw/S001/SE01/CAM01/S001_SE01_CAP000016_CAM01.jpg"
        p2 = Path(test_cam02) if test_cam02 else PROJECT_ROOT / "02_data/private_raw/S001/SE01/CAM02/S001_SE01_CAP000016_CAM02.jpg"

        if not p1.exists() or not p2.exists():
            raise FileNotFoundError(f"Test images not found: {p1} or {p2}")

        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))
        sim_frames = (img1, img2)
        print(f"[TEST MODE] Simulating real-time feed from: {p1.name} + {p2.name}")
        cam1 = None
        cam2 = None
    else:
        try:
            cam1 = ThreadedCamera(cam01_idx, width=640, height=480)
            cam2 = ThreadedCamera(cam02_idx, width=640, height=480)
        except Exception as e:
            print(f"Warning: Failed to open webcams ({e}). Falling back to simulation test mode...")
            p1 = PROJECT_ROOT / "02_data/private_raw/S001/SE01/CAM01/S001_SE01_CAP000016_CAM01.jpg"
            p2 = PROJECT_ROOT / "02_data/private_raw/S001/SE01/CAM02/S001_SE01_CAP000016_CAM02.jpg"
            sim_frames = (cv2.imread(str(p1)), cv2.imread(str(p2)))
            cam1 = None
            cam2 = None

    fps_tracker = []
    frame_count = 0

    window_name = "Mitigasi Skoliosis — Real-Time Stereo 3D (XGBoost)"

    try:
        while True:
            t_start = time.time()

            if sim_frames is not None:
                f1, f2 = sim_frames[0].copy(), sim_frames[1].copy()
                time.sleep(0.03)
            else:
                f1 = cam1.read()
                f2 = cam2.read()
                if f1 is None or f2 is None:
                    time.sleep(0.005)
                    continue

            # Run Stereo 3D Inference
            result = infer_pair_3d(f1, f2, calibration_id_or_path=calibration)

            # Measure FPS
            t_elapsed = time.time() - t_start
            inst_fps = 1.0 / max(1e-4, t_elapsed)
            fps_tracker.append(inst_fps)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = float(np.mean(fps_tracker))

            # Compose visual HUD
            hud_view = build_realtime_hud_3d(f1, f2, result, avg_fps, cal_id=calibration)

            frame_count += 1
            if frame_count % 15 == 0:
                pred_str = result.get("prediction", "REJECT")
                reproj_val = result.get("reprojection_error", 0.0)
                conf_str = f"{result.get('confidence', 0.0)*100:.1f}%" if result.get('status') == 'VALID' else result.get('reason')
                print(f"[Frame {frame_count:04d}] Posture: {pred_str:<16} | Conf: {conf_str:<10} | Reproj: {reproj_val:4.1f}px | FPS: {avg_fps:4.1f}")

            # Display GUI window
            try:
                cv2.imshow(window_name, hud_view)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\nExit key detected. Closing real-time stream...")
                    break
            except cv2.error:
                if frame_count >= 30:
                    print("[HEADLESS] Completed 30 sample inference iterations successfully.")
                    break

            if test_mode and frame_count >= 60:
                print("\n[TEST MODE] Completed 60 frames demo.")
                break

    finally:
        if cam1:
            cam1.release()
        if cam2:
            cam2.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Stereo 3D Posture Inference")
    parser.add_argument("--cam01-idx", type=int, default=0, help="Camera index for CAM01 (Frontal)")
    parser.add_argument("--cam02-idx", type=int, default=1, help="Camera index for CAM02 (Lateral)")
    parser.add_argument("--calibration", type=str, default="CAL_001", help="Calibration ID (e.g. CAL_001) or path to stereo calibration JSON")
    parser.add_argument("--test-mode", action="store_true", help="Run simulation test mode without physical webcams")
    parser.add_argument("--test-cam01", type=str, default=None, help="Custom test image for CAM01")
    parser.add_argument("--test-cam02", type=str, default=None, help="Custom test image for CAM02")

    args = parser.parse_args()

    run_realtime_stereo_3d(
        cam01_idx=args.cam01_idx,
        cam02_idx=args.cam02_idx,
        calibration=args.calibration,
        test_mode=args.test_mode,
        test_cam01=args.test_cam01,
        test_cam02=args.test_cam02
    )


if __name__ == "__main__":
    main()
