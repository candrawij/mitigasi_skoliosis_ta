"""
infer_realtime_2d.py — Real-Time 2D Posture Inference Prototype
Supports:
  1. Single-Camera Mode (Live Laptop/Desk Testing):
     python 04_scripts/inference/infer_realtime_2d.py --single-cam --cam01-idx 0

  2. Dual-Camera Mode (Dual Physical Webcams):
     python 04_scripts/inference/infer_realtime_2d.py --cam01-idx 0 --cam02-idx 1 --lateral-side right

  3. Camera Scanner:
     python 04_scripts/inference/infer_realtime_2d.py --scan-cameras
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

# Suppress verbose OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "inference"))
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))

from private_inference_common import (
    infer_pair_2d,
    infer_single_cam_2d,
    detect_target_person_keypoints,
    check_2d_reject_gate,
    MAIN_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    COCO_NOSE,
    COCO_LEFT_SHOULDER,
    COCO_RIGHT_SHOULDER,
    COCO_LEFT_HIP,
    COCO_RIGHT_HIP
)


def scan_available_cameras(max_check: int = 4) -> List[Tuple[int, int, int]]:
    """Scan and return list of available camera tuples: (index, width, height)."""
    available = []
    print("Memindai port kamera yang terhubung...")
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened() and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((idx, w, h))
            cap.release()
    return available


class ThreadedCamera:
    """Thread-safe camera grabber for low-latency live streaming."""
    def __init__(self, cam_idx: int, width: int = 640, height: int = 480):
        self.cam_idx = cam_idx
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened() and sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak dapat membuka kamera pada index {cam_idx}")

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
            cv2.circle(image, (int(pt[0]), int(pt[1])), 6, color, -1)
            cv2.circle(image, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 1)

    ls, rs = kpts[COCO_LEFT_SHOULDER], kpts[COCO_RIGHT_SHOULDER]
    lh, rh = kpts[COCO_LEFT_HIP], kpts[COCO_RIGHT_HIP]
    nose = kpts[COCO_NOSE]

    if not (np.isnan(ls[0]) or np.isnan(rs[0])):
        cv2.line(image, (int(ls[0]), int(ls[1])), (int(rs[0]), int(rs[1])), color, 3)
    if not (np.isnan(lh[0]) or np.isnan(rh[0])):
        cv2.line(image, (int(lh[0]), int(lh[1])), (int(rh[0]), int(rh[1])), color, 3)
    if not (np.isnan(ls[0]) or np.isnan(rs[0]) or np.isnan(lh[0]) or np.isnan(rh[0])):
        sh_c = ((ls + rs) / 2.0).astype(int)
        hip_c = ((lh + rh) / 2.0).astype(int)
        cv2.line(image, (sh_c[0], sh_c[1]), (hip_c[0], hip_c[1]), (255, 255, 0), 3)
        if not np.isnan(nose[0]):
            cv2.line(image, (int(nose[0]), int(nose[1])), (sh_c[0], sh_c[1]), (0, 255, 255), 2)

    return image


def build_hud_single_cam(
    frame: np.ndarray,
    result: dict,
    fps: float
) -> np.ndarray:
    """Compose single-camera full-screen display with modern graphical HUD."""
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Draw skeleton
    kpts = result.get("kpts")
    if kpts is not None:
        canvas = draw_skeleton_overlay(canvas, kpts)

    # 1. Top HUD Bar
    hud_h = 75
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    # Bottom HUD Bar for probabilities
    bot_h = 80
    cv2.rectangle(overlay, (0, h - bot_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Status and badge color
    status = result.get("status", "REJECTED")
    pred = result.get("prediction", "REJECT")
    conf = result.get("confidence", 0.0)
    is_extrap = result.get("is_extrapolated", False)

    if status == "VALID":
        if pred == "upright":
            badge_color = (50, 205, 50)   # Green
        elif "leaning" in pred:
            badge_color = (0, 165, 255)   # Orange
        else:
            badge_color = (0, 215, 255)   # Yellow
        status_text = f"POSTURE: {pred.upper()}"
        conf_text = f"CONFIDENCE: {conf*100:.1f}%"
    else:
        badge_color = (50, 50, 220)       # Red
        status_text = "POSTURE: REJECT / INVALID"
        conf_text = f"REASON: {result.get('reason', 'QC Gate')}"

    # Header texts
    cv2.putText(canvas, "LIVE WEBCAM (FRONTAL)", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    if is_extrap:
        cv2.putText(canvas, "[Desk Mode: Hips Extrapolated]", (220, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 100), 1)

    cv2.putText(canvas, status_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, badge_color, 2)
    cv2.putText(canvas, conf_text, (min(w - 280, 350), 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(canvas, f"FPS: {fps:4.1f}", (w - 110, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Bottom HUD: Probabilities Bar Chart
    probs = result.get("probabilities", {})
    bar_x = 15
    bar_spacing = int((w - 30) / max(1, len(MAIN_CLASSES)))

    for i, c_name in enumerate(MAIN_CLASSES):
        p = probs.get(c_name, 0.0)
        bx = bar_x + i * bar_spacing
        by = h - 15
        bw = max(20, bar_spacing - 15)

        lbl = c_name.replace("leaning_", "lean_")[:10]
        cv2.putText(canvas, f"{lbl}: {p*100:3.0f}%", (bx, by - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        cv2.rectangle(canvas, (bx, by - 18), (bx + bw, by - 6), (50, 50, 50), -1)
        fill_w = int(bw * p)
        fill_col = (0, 255, 0) if c_name == pred and status == "VALID" else (180, 180, 0)
        cv2.rectangle(canvas, (bx, by - 18), (bx + fill_w, by - 6), fill_col, -1)

    return canvas


def build_hud_dual_cam(
    frame_cam01: np.ndarray,
    frame_cam02: np.ndarray,
    result: dict,
    fps: float,
    lateral_side: str
) -> np.ndarray:
    """Compose dual-camera split-screen display with modern graphical HUD overlay."""
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
    hud_h = 70
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_w, hud_h), (20, 20, 20), -1)
    # Bottom HUD Bar
    bot_h = 75
    cv2.rectangle(overlay, (0, target_h - bot_h), (canvas_w, target_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    cv2.putText(canvas, "CAM01: FRONTAL LIVE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, f"CAM02: LATERAL LIVE ({lateral_side.upper()})", (w1_r + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    status = result.get("status", "REJECTED")
    pred = result.get("prediction", "REJECT")
    conf = result.get("confidence", 0.0)

    if status == "VALID":
        badge_color = (50, 205, 50) if pred == "upright" else ((0, 165, 255) if "leaning" in pred else (0, 215, 255))
        status_text = f"POSTURE: {pred.upper()}"
        conf_text = f"CONFIDENCE: {conf*100:.1f}%"
    else:
        badge_color = (50, 50, 220)
        status_text = "POSTURE: REJECT / INVALID"
        conf_text = f"REASON: {result.get('reason', 'QC Gate')}"

    cv2.putText(canvas, status_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, badge_color, 2)
    cv2.putText(canvas, conf_text, (380, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(canvas, f"FPS: {fps:4.1f}", (canvas_w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(canvas, "MODE: Dual-Camera Live 2D", (canvas_w - 260, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Probabilities
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


def run_single_cam_live(cam_idx: int = 0):
    print("=" * 80)
    print("  MEMBUKA MODE SINGLE-CAMERA (LIVE WEBCAM ANDA)")
    print("=" * 80)
    print(f"Kamera Index      : {cam_idx}")
    print("Desk Mode         : AKTIF (Posisi panggul terestimasi jika tertutup meja)")
    print("Tekan [Q] atau [ESC] pada jendela video untuk keluar.\n")

    cam = ThreadedCamera(cam_idx, width=640, height=480)
    fps_tracker = []
    frame_count = 0
    window_name = "Mitigasi Skoliosis — Live Webcam (Single Camera Mode)"

    try:
        while True:
            t_start = time.time()
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Run single camera inference with desk mode extrapolation
            result = infer_single_cam_2d(frame, desk_mode=True)

            t_elapsed = time.time() - t_start
            inst_fps = 1.0 / max(1e-4, t_elapsed)
            fps_tracker.append(inst_fps)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = float(np.mean(fps_tracker))

            hud_frame = build_hud_single_cam(frame, result, avg_fps)

            frame_count += 1
            if frame_count % 20 == 0:
                pred_str = result.get("prediction", "REJECT")
                conf_str = f"{result.get('confidence', 0.0)*100:.1f}%" if result.get('status') == 'VALID' else result.get('reason')
                print(f"[Frame {frame_count:04d}] Posture: {pred_str:<16} | Conf: {conf_str:<15} | FPS: {avg_fps:4.1f}")

            cv2.imshow(window_name, hud_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("\nKeluar dari live webcam...")
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


def run_dual_cam_live(cam01_idx: int = 0, cam02_idx: int = 1, lateral_side: str = "right"):
    print("=" * 80)
    print("  MEMBUKA MODE DUAL-CAMERA REAL-TIME")
    print("=" * 80)
    print(f"CAM01 (Frontal) Index : {cam01_idx}")
    print(f"CAM02 (Lateral) Index : {cam02_idx}")
    print(f"Lateral Side          : {lateral_side}")
    print("Tekan [Q] atau [ESC] pada jendela video untuk keluar.\n")

    cam1 = ThreadedCamera(cam01_idx, width=640, height=480)
    cam2 = ThreadedCamera(cam02_idx, width=640, height=480)

    fps_tracker = []
    frame_count = 0
    window_name = "Mitigasi Skoliosis — Real-Time Dual-Camera (XGBoost)"

    try:
        while True:
            t_start = time.time()
            f1 = cam1.read()
            f2 = cam2.read()
            if f1 is None or f2 is None:
                time.sleep(0.005)
                continue

            result = infer_pair_2d(f1, f2, lateral_side=lateral_side)

            t_elapsed = time.time() - t_start
            inst_fps = 1.0 / max(1e-4, t_elapsed)
            fps_tracker.append(inst_fps)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = float(np.mean(fps_tracker))

            hud_frame = build_hud_dual_cam(f1, f2, result, avg_fps, lateral_side)

            frame_count += 1
            if frame_count % 20 == 0:
                pred_str = result.get("prediction", "REJECT")
                conf_str = f"{result.get('confidence', 0.0)*100:.1f}%" if result.get('status') == 'VALID' else result.get('reason')
                print(f"[Frame {frame_count:04d}] Posture: {pred_str:<16} | Conf: {conf_str:<15} | FPS: {avg_fps:4.1f}")

            cv2.imshow(window_name, hud_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("\nKeluar dari dual webcam...")
                break

    finally:
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time 2D Posture Inference Prototype")
    parser.add_argument("--single-cam", action="store_true", help="Gunakan mode satu kamera (Live Laptop/Desk Testing)")
    parser.add_argument("--cam01-idx", type=int, default=0, help="Index kamera CAM01 (Frontal)")
    parser.add_argument("--cam02-idx", type=int, default=1, help="Index kamera CAM02 (Lateral)")
    parser.add_argument("--lateral-side", type=str, default="right", choices=["left", "right"], help="Sisi lateral kamera CAM02")
    parser.add_argument("--scan-cameras", action="store_true", help="Pindai kamera yang terhubung ke PC")

    args = parser.parse_args()

    if args.scan_cameras:
        cams = scan_available_cameras()
        if cams:
            print("\nKamera yang terdeteksi pada PC Anda:")
            for idx, w, h in cams:
                print(f"  - Kamera Index [{idx}]: Resolusi {w}x{h}")
        else:
            print("Tidak ada kamera yang terdeteksi.")
        return

    if args.single_cam:
        run_single_cam_live(cam_idx=args.cam01_idx)
    else:
        # Check if both cameras can be opened, else offer single cam
        try:
            run_dual_cam_live(cam01_idx=args.cam01_idx, cam02_idx=args.cam02_idx, lateral_side=args.lateral_side)
        except Exception as e:
            print(f"\n[INFO] Gagal membuka kamera ganda ({e}).")
            print(">>> Membuka otomatis dalam mode SINGLE-CAMERA (Webcam Anda)...")
            run_single_cam_live(cam_idx=args.cam01_idx)


if __name__ == "__main__":
    main()
