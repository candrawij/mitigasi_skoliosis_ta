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
    SeatedBaselineTracker,
    MAIN_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    COCO_NOSE,
    COCO_LEFT_SHOULDER,
    COCO_RIGHT_SHOULDER,
    COCO_LEFT_HIP,
    COCO_RIGHT_HIP
)


def scan_available_cameras(max_check: int = 6) -> List[Tuple[int, int, int]]:
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


def preview_cameras_grid(available_cams: List[Tuple[int, int, int]]):
    """
    Opens a quick visual preview window showing all detected cameras with their index numbers
    so the user can immediately see which camera corresponds to which index.
    """
    if not available_cams:
        return

    print("\n[PREVIEW] Membuka jendela pratinjau kamera...")
    print(">>> Lihat nomor port pada setiap kamera pada jendela 'Pratinjau Kamera'.")
    print(">>> Tekan SEMBARANG TOMBOL atau [ESC] pada jendela video untuk lanjut memilih...\n")

    caps = []
    for idx, _, _ in available_cams:
        c = cv2.VideoCapture(idx)
        if not c.isOpened() and sys.platform.startswith("win"):
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        caps.append(c)

    win_name = "Pratinjau Kamera - Tekan Sembarang Tombol untuk Memilih"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        start_time = time.time()
        while time.time() - start_time < 30.0:  # 30s timeout
            frames = []
            for i, (idx, _, _) in enumerate(available_cams):
                cap = caps[i]
                if cap.isOpened():
                    ret, f = cap.read()
                    if ret and f is not None:
                        f_small = cv2.resize(f, (320, 240))
                        # Draw label badge
                        cv2.rectangle(f_small, (0, 0), (320, 42), (20, 20, 20), -1)
                        cv2.putText(f_small, f"KAMERA PORT [{idx}]", (15, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
                        frames.append(f_small)
                    else:
                        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
                        cv2.putText(placeholder, f"Port [{idx}] No Frame", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        frames.append(placeholder)

            if frames:
                composite = np.hstack(frames)
                cv2.imshow(win_name, composite)

            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                break
    finally:
        for c in caps:
            c.release()
        cv2.destroyWindow(win_name)


def interactive_camera_selection() -> Dict[str, Any]:
    """Interactive CLI menu to choose frontal, lateral cameras, and lateral side."""
    print("=" * 75)
    print("         SISTEM PEMILIHAN KAMERA DETEKSI POSTUR REAL-TIME")
    print("=" * 75)

    cams = scan_available_cameras(max_check=6)
    if not cams:
        print("\n[PERINGATAN] Tidak ada kamera yang terdeteksi.")
        print("Pastikan kabel USB kamera terpasang dan driver kamera aktif.")
        input("\nTekan Enter untuk keluar...")
        sys.exit(1)

    print(f"\nDitemukan {len(cams)} kamera pada komputer:")
    for idx, w, h in cams:
        print(f"  - Port [{idx}] : Resolusi {w}x{h} px")

    # If only 1 camera is detected
    if len(cams) == 1:
        c_idx = cams[0][0]
        print(f"\n[INFO] Hanya 1 kamera fisik terdeteksi (Port [{c_idx}]).")
        print("Aplikasi otomatis menjalankan mode SINGLE-CAMERA (Frontal / Meja Kerja).")
        input("Tekan Enter untuk memulai live video...")
        return {
            "mode": "single",
            "cam01_idx": c_idx,
            "cam02_idx": None,
            "lateral_side": "right"
        }

    # If multiple cameras: offer preview first
    print("\nIngin melihat jendela pratinjau (preview) semua kamera terlebih dahulu?")
    show_prev = input("Tampilkan preview kamera? (y/n) [Default: y]: ").strip().lower()
    if show_prev == "" or show_prev.startswith("y"):
        preview_cameras_grid(cams)

    print("\n" + "-" * 75)
    print("PILIHAN MODE DETEKSI:")
    print("  [1] DUAL-CAMERA (Kamera Depan + Kamera Samping) [Rekomendasi]")
    print("  [2] SINGLE-CAMERA (Hanya 1 Kamera Depan / Webcam Laptop)")
    print("-" * 75)

    m_choice = input("Pilih mode (1/2) [Default: 1]: ").strip()
    if m_choice == "2":
        cam_in = input(f"Pilih Nomor Port Kamera untuk Frontal (Pilihan: {[c[0] for c in cams]}) [Default: {cams[0][0]}]: ").strip()
        cam_idx = int(cam_in) if cam_in.isdigit() and int(cam_in) in [c[0] for c in cams] else cams[0][0]
        return {
            "mode": "single",
            "cam01_idx": cam_idx,
            "cam02_idx": None,
            "lateral_side": "right"
        }

    # Dual-camera selection
    default_c1 = cams[0][0]
    default_c2 = cams[1][0] if len(cams) > 1 else cams[0][0]

    print("\n--- Konfigurasi Kamera Ganda ---")
    c1_in = input(f"Pilih Port Kamera FRONTAL (Depan)  (Pilihan: {[c[0] for c in cams]}) [Default: {default_c1}]: ").strip()
    c1_idx = int(c1_in) if c1_in.isdigit() and int(c1_in) in [c[0] for c in cams] else default_c1

    c2_in = input(f"Pilih Port Kamera LATERAL (Samping) (Pilihan: {[c[0] for c in cams]}) [Default: {default_c2}]: ").strip()
    c2_idx = int(c2_in) if c2_in.isdigit() and int(c2_in) in [c[0] for c in cams] else default_c2

    if c1_idx == c2_idx:
        print("[INFO] Port kamera depan dan samping sama. Mengatur port samping secara otomatis ke port lain...")
        for c in cams:
            if c[0] != c1_idx:
                c2_idx = c[0]
                break

    lat_in = input("Posisikan kamera samping di sisi mana tubuh? (r = Kanan, l = Kiri) [Default: r (Kanan)]: ").strip().lower()
    lateral_side = "left" if lat_in.startswith("l") else "right"

    print("\n" + "=" * 75)
    print("KONFIGURASI KAMERA AKTIF:")
    print(f"  CAM01 (Frontal/Depan)  : Port [{c1_idx}]")
    print(f"  CAM02 (Lateral/Samping): Port [{c2_idx}] (Sisi: {lateral_side.upper()})")
    print("  *TIPS: Jika kamera tertukar saat streaming, cukup tekan tombol [S] untuk SWAP seketika!")
    print("  *TIPS: Tekan tombol [L] untuk mengubah sisi samping (Kanan <-> Kiri).")
    print("=" * 75 + "\n")

    return {
        "mode": "dual",
        "cam01_idx": c1_idx,
        "cam02_idx": c2_idx,
        "lateral_side": lateral_side
    }


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
    fps: float,
    cam_idx: int = 0,
    desk_mode: bool = True
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
    cv2.putText(canvas, f"CAM01 (FRONTAL) [PORT {cam_idx}]", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
    if is_extrap:
        cv2.putText(canvas, "[Desk Mode]", (250, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 100), 1)

    prof = result.get("profile")
    if prof:
        cv2.putText(canvas, f"[{prof.upper()}]", (350, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    # Controls hint
    cv2.putText(canvas, "[C] Kalibrasi  |  [D] Desk-Mode  |  [Q] Keluar", (w - 370, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    cv2.putText(canvas, status_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, badge_color, 2)
    cv2.putText(canvas, conf_text, (min(w - 280, 360), 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(canvas, f"FPS: {fps:4.1f}", (w - 110, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

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
    lateral_side: str,
    cam01_idx: int = 0,
    cam02_idx: int = 1,
    desk_mode: bool = True
) -> np.ndarray:
    """Compose dual-camera split-screen display with modern graphical HUD overlay."""
    h1, w1 = frame_cam01.shape[:2]
    h2, w2 = frame_cam02.shape[:2]

    target_h = 480
    w1_r = int(w1 * (target_h / h1))
    w2_r = int(w2 * (target_h / h2))

    c1_resized = cv2.resize(frame_cam01, (w1_r, target_h))
    c2_resized = cv2.resize(frame_cam02, (w2_r, target_h))

    # Draw skeleton overlay on CAM01 if keypoints available
    kpts1 = result.get("kpts1")
    if kpts1 is not None:
        k1_scaled = kpts1.copy()
        k1_scaled[:, 0] *= (w1_r / float(w1))
        k1_scaled[:, 1] *= (target_h / float(h1))
        c1_resized = draw_skeleton_overlay(c1_resized, k1_scaled, color=(0, 255, 0))

    # Draw skeleton overlay on CAM02 if keypoints available
    kpts2 = result.get("kpts2")
    if kpts2 is not None:
        k2_scaled = kpts2.copy()
        k2_scaled[:, 0] *= (w2_r / float(w2))
        k2_scaled[:, 1] *= (target_h / float(h2))
        c2_resized = draw_skeleton_overlay(c2_resized, k2_scaled, color=(0, 200, 255))

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

    desk_badge = " [Desk-Mode]" if desk_mode else ""
    cv2.putText(canvas, f"CAM01 (FRONTAL) [PORT {cam01_idx}]{desk_badge}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 2)
    cv2.putText(canvas, f"CAM02 (LATERAL {lateral_side.upper()}) [PORT {cam02_idx}]", (w1_r + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)

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
    cv2.putText(canvas, conf_text, (370, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(canvas, f"FPS: {fps:4.1f}", (canvas_w - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Hotkey hints
    cv2.putText(canvas, "[C] KALIBRASI  |  [D] DESK-MODE  |  [S] TUKAR  |  [L] SISI  |  [Q] KELUAR",
                (canvas_w - 530, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

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


def run_single_cam_live(cam_idx: int = 0, desk_mode: bool = True):
    print("=" * 80)
    print("  MEMBUKA MODE SINGLE-CAMERA (LIVE WEBCAM ANDA)")
    print("=" * 80)
    print(f"Kamera Index      : Port [{cam_idx}]")
    print(f"Desk Mode         : {'AKTIF' if desk_mode else 'NONAKTIF'} (Posisi panggul terestimasi)")
    print("Hotkeys Interaktif:")
    print("  [C] : Kalibrasi Posisi Duduk Tegak Netral (Tekan saat duduk tegak)")
    print("  [D] : Aktifkan / Nonaktifkan Desk-Mode (Ekstrapolasi Panggul)")
    print("  [Q] / [ESC] : Keluar dari live video.")
    print("")
    print("  === PENGUMPULAN DATA PENGUJIAN ===")
    print("  [1] = Rekam postur UPRIGHT (Tegak)")
    print("  [2] = Rekam postur LEANING FORWARD (Condong Depan)")
    print("  [3] = Rekam postur LEANING BACKWARD (Condong Belakang)")
    print("  [4] = Rekam postur LEANING LEFT (Miring Kiri)")
    print("  [5] = Rekam postur LEANING RIGHT (Miring Kanan)")
    print("  [6] = Rekam postur SLOUCHING (Bungkuk)")
    print("  [0] = Berhenti merekam (Idle)")
    print("")

    GT_KEY_MAP = {
        ord('1'): "upright",
        ord('2'): "leaning_forward",
        ord('3'): "leaning_backward",
        ord('4'): "leaning_left",
        ord('5'): "leaning_right",
        ord('6'): "slouching",
    }

    cam = ThreadedCamera(cam_idx, width=640, height=480)
    tracker = SeatedBaselineTracker()
    fps_tracker = []
    frame_count = 0
    window_name = f"Mitigasi Skoliosis — Live Webcam [Port {cam_idx}]"

    # Data collection state
    current_gt_label: Optional[str] = None
    collected_rows: List[Dict[str, Any]] = []

    try:
        while True:
            t_start = time.time()
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Run single camera inference with baseline tracking & desk mode
            result = infer_single_cam_2d(frame, desk_mode=desk_mode, tracker=tracker)

            t_elapsed = time.time() - t_start
            inst_fps = 1.0 / max(1e-4, t_elapsed)
            fps_tracker.append(inst_fps)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = float(np.mean(fps_tracker))

            hud_frame = build_hud_single_cam(frame, result, avg_fps, cam_idx=cam_idx, desk_mode=desk_mode)

            # Draw recording indicator on HUD
            h_frame, w_frame = hud_frame.shape[:2]
            if current_gt_label is not None:
                gt_display = current_gt_label.upper().replace("_", " ")
                # Red recording badge
                cv2.rectangle(hud_frame, (w_frame - 320, 65), (w_frame, 95), (0, 0, 180), -1)
                cv2.putText(hud_frame, f"REC GT: {gt_display}", (w_frame - 315, 87),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                # Blinking red circle
                if frame_count % 30 < 20:
                    cv2.circle(hud_frame, (w_frame - 330, 80), 6, (0, 0, 255), -1)

            # Collect data if recording
            if current_gt_label is not None and result.get("status") == "VALID":
                row = {
                    "frame": frame_count,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ground_truth": current_gt_label,
                    "prediction": result.get("prediction", "REJECT"),
                    "profile": result.get("profile", "upright"),
                    "confidence": round(result.get("confidence", 0.0), 4),
                    "correct": 1 if result.get("prediction") == current_gt_label else 0,
                }
                # Add individual class probabilities
                probs = result.get("probabilities", {})
                for c_name in MAIN_CLASSES:
                    row[f"prob_{c_name}"] = round(probs.get(c_name, 0.0), 4)
                collected_rows.append(row)

            frame_count += 1
            if frame_count % 20 == 0:
                pred_str = result.get("prediction", "REJECT")
                conf_str = f"{result.get('confidence', 0.0)*100:.1f}%" if result.get('status') == 'VALID' else result.get('reason')
                prof_str = result.get("profile", "-")
                gt_str = current_gt_label if current_gt_label else "IDLE"
                rec_count = len(collected_rows)
                print(f"[Frame {frame_count:04d}] GT: {gt_str:<16} | Pred: {pred_str:<16} | Profile: {prof_str:<16} | Conf: {conf_str:<10} | Rec: {rec_count} | FPS: {avg_fps:4.1f}")

            cv2.imshow(window_name, hud_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("\nKeluar dari live webcam...")
                break
            elif key in [ord('c'), ord('C')]:
                kpts = result.get("kpts")
                if kpts is not None:
                    tracker.calibrate(kpts, frame_h=frame.shape[0])
                    print("\n>>> [KALIBRASI] Posisi duduk netral tegak berhasil direkam!")
                    print(f"    Baseline: CX={tracker.base_cx:.1f}px, ShWidth={tracker.sh_w_base:.1f}px, HeadRatio={tracker.head_ratio_base:.3f}")
                else:
                    print("\n>>> [KALIBRASI] Peringatan: Keypoint tubuh belum terdeteksi sempurna.")
            elif key in [ord('d'), ord('D')]:
                desk_mode = not desk_mode
                print(f"\n>>> [DESK MODE] Ekstrapolasi panggul diubah: {'AKTIF' if desk_mode else 'NONAKTIF'}")
            elif key in GT_KEY_MAP:
                current_gt_label = GT_KEY_MAP[key]
                print(f"\n>>> [REKAM] Ground truth diset: {current_gt_label.upper()} — Lakukan postur ini sekarang!")
            elif key == ord('0'):
                current_gt_label = None
                print(f"\n>>> [REKAM] Perekaman dihentikan (IDLE). Total data terkumpul: {len(collected_rows)} frame.")

    finally:
        cam.release()
        cv2.destroyAllWindows()

        # Save collected data to CSV
        if collected_rows:
            import csv
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = PROJECT_ROOT / "07_results" / "live_camera_tests"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"live_test_{timestamp_str}.csv"

            fieldnames = list(collected_rows[0].keys())
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(collected_rows)

            # Print summary
            import pandas as pd
            df = pd.DataFrame(collected_rows)
            total = len(df)
            correct = df["correct"].sum()
            accuracy = correct / total * 100 if total > 0 else 0.0

            print("\n" + "=" * 80)
            print("  RINGKASAN HASIL PENGUJIAN LIVE CAMERA")
            print("=" * 80)
            print(f"  Total frame terekam  : {total}")
            print(f"  Benar (correct)      : {correct}")
            print(f"  Akurasi keseluruhan  : {accuracy:.1f}%")
            print(f"  File disimpan di     : {out_path}")
            print("")

            # Per-class breakdown
            print("  Per-Kelas Postur:")
            print(f"  {'Postur':<20} {'Total':>6} {'Benar':>6} {'Akurasi':>8}")
            print("  " + "-" * 42)
            for label in MAIN_CLASSES:
                df_c = df[df["ground_truth"] == label]
                if len(df_c) > 0:
                    c_total = len(df_c)
                    c_correct = df_c["correct"].sum()
                    c_acc = c_correct / c_total * 100
                    print(f"  {label:<20} {c_total:>6} {c_correct:>6} {c_acc:>7.1f}%")
            print("=" * 80)
        else:
            print("\n[INFO] Tidak ada data pengujian yang direkam.")


def run_dual_cam_live(cam01_idx: int = 0, cam02_idx: int = 1, lateral_side: str = "right"):
    print("=" * 80)
    print("  MEMBUKA MODE DUAL-CAMERA REAL-TIME")
    print("=" * 80)
    print(f"CAM01 (Frontal) : Port [{cam01_idx}]")
    print(f"CAM02 (Lateral) : Port [{cam02_idx}]")
    print(f"Sisi Lateral    : {lateral_side.upper()}")
    print("Hotkeys Interaktif:")
    print("  [C] : Kalibrasi Posisi Duduk Tegak Netral (Tekan saat duduk tegak)")
    print("  [D] : Aktifkan / Nonaktifkan Desk-Mode (Ekstrapolasi Panggul Meja)")
    print("  [S] : SWAP kamera! (Tukar posisi Kamera Frontal <-> Lateral seketika)")
    print("  [L] : Ganti sisi kamera samping (Kanan <-> Kiri)")
    print("  [Q] / [ESC] : Keluar dari dual webcam.")
    print("")
    print("  === PENGUMPULAN DATA PENGUJIAN ===")
    print("  [1] = Rekam postur UPRIGHT (Tegak)")
    print("  [2] = Rekam postur LEANING FORWARD (Condong Depan)")
    print("  [3] = Rekam postur LEANING BACKWARD (Condong Belakang)")
    print("  [4] = Rekam postur LEANING LEFT (Miring Kiri)")
    print("  [5] = Rekam postur LEANING RIGHT (Miring Kanan)")
    print("  [6] = Rekam postur SLOUCHING (Bungkuk)")
    print("  [0] = Berhenti merekam (Idle)")
    print("")

    GT_KEY_MAP = {
        ord('1'): "upright",
        ord('2'): "leaning_forward",
        ord('3'): "leaning_backward",
        ord('4'): "leaning_left",
        ord('5'): "leaning_right",
        ord('6'): "slouching",
    }

    cam1 = ThreadedCamera(cam01_idx, width=640, height=480)
    cam2 = ThreadedCamera(cam02_idx, width=640, height=480)

    tracker_c1 = SeatedBaselineTracker()
    tracker_c2 = SeatedBaselineTracker()
    desk_mode = True

    fps_tracker = []
    frame_count = 0
    window_name = "Mitigasi Skoliosis — Real-Time Dual-Camera (XGBoost)"

    # Data collection state
    current_gt_label: Optional[str] = None
    collected_rows: List[Dict[str, Any]] = []

    try:
        while True:
            t_start = time.time()
            f1 = cam1.read()
            f2 = cam2.read()
            if f1 is None or f2 is None:
                time.sleep(0.005)
                continue

            result = infer_pair_2d(
                f1, f2,
                lateral_side=lateral_side,
                desk_mode=desk_mode,
                tracker_c1=tracker_c1,
                tracker_c2=tracker_c2
            )

            t_elapsed = time.time() - t_start
            inst_fps = 1.0 / max(1e-4, t_elapsed)
            fps_tracker.append(inst_fps)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = float(np.mean(fps_tracker))

            hud_frame = build_hud_dual_cam(
                f1, f2, result, avg_fps, lateral_side,
                cam01_idx=cam01_idx, cam02_idx=cam02_idx,
                desk_mode=desk_mode
            )

            # Draw recording indicator on HUD
            h_frame, w_frame = hud_frame.shape[:2]
            if current_gt_label is not None:
                gt_display = current_gt_label.upper().replace("_", " ")
                # Red recording badge
                cv2.rectangle(hud_frame, (w_frame - 320, 65), (w_frame, 95), (0, 0, 180), -1)
                cv2.putText(hud_frame, f"REC GT: {gt_display}", (w_frame - 315, 87),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                # Blinking red circle
                if frame_count % 30 < 20:
                    cv2.circle(hud_frame, (w_frame - 330, 80), 6, (0, 0, 255), -1)

            # Collect data if recording
            if current_gt_label is not None and result.get("status") == "VALID":
                row = {
                    "frame": frame_count,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ground_truth": current_gt_label,
                    "prediction": result.get("prediction", "REJECT"),
                    "profile": result.get("profile", "upright"),
                    "confidence": round(result.get("confidence", 0.0), 4),
                    "correct": 1 if result.get("prediction") == current_gt_label else 0,
                }
                # Add individual class probabilities
                probs = result.get("probabilities", {})
                for c_name in MAIN_CLASSES:
                    row[f"prob_{c_name}"] = round(probs.get(c_name, 0.0), 4)
                collected_rows.append(row)

            frame_count += 1
            if frame_count % 20 == 0:
                pred_str = result.get("prediction", "REJECT")
                conf_str = f"{result.get('confidence', 0.0)*100:.1f}%" if result.get('status') == 'VALID' else result.get('reason')
                gt_str = current_gt_label if current_gt_label else "IDLE"
                rec_count = len(collected_rows)
                print(f"[Frame {frame_count:04d}] GT: {gt_str:<16} | Frontal[Port {cam01_idx}] | Lateral[Port {cam02_idx}] | Posture: {pred_str:<16} | Conf: {conf_str:<15} | Rec: {rec_count} | FPS: {avg_fps:4.1f}")

            cv2.imshow(window_name, hud_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("\nKeluar dari dual webcam...")
                break
            elif key in [ord('c'), ord('C')]:
                k1 = result.get("kpts1")
                k2 = result.get("kpts2")
                if k1 is not None and k2 is not None:
                    tracker_c1.calibrate(k1, frame_h=f1.shape[0])
                    tracker_c2.calibrate(k2, frame_h=f2.shape[0])
                    print("\n>>> [KALIBRASI] Posisi duduk frontal & lateral berhasil direkam!")
                else:
                    print("\n>>> [KALIBRASI] Peringatan: Pastikan kedua kamera melihat tubuh partisipan.")
            elif key in [ord('d'), ord('D')]:
                desk_mode = not desk_mode
                print(f"\n>>> [DESK MODE] Ekstrapolasi panggul meja diubah: {'AKTIF' if desk_mode else 'NONAKTIF'}")
            elif key in [ord('s'), ord('S')]:
                # SWAP cameras on the fly!
                cam1, cam2 = cam2, cam1
                cam01_idx, cam02_idx = cam02_idx, cam01_idx
                tracker_c1, tracker_c2 = tracker_c2, tracker_c1
                print(f"\n>>> [SWAP] Kamera berhasil ditukar!")
                print(f"    CAM01 (Frontal) sekarang : Port [{cam01_idx}]")
                print(f"    CAM02 (Lateral) sekarang : Port [{cam02_idx}]")
            elif key in [ord('l'), ord('L')]:
                lateral_side = "left" if lateral_side == "right" else "right"
                print(f"\n>>> [LATERAL] Sisi kamera lateral diubah ke: {lateral_side.upper()}")
            elif key in GT_KEY_MAP:
                current_gt_label = GT_KEY_MAP[key]
                print(f"\n>>> [REKAM] Ground truth diset: {current_gt_label.upper()} — Lakukan postur ini sekarang!")
            elif key == ord('0'):
                current_gt_label = None
                print(f"\n>>> [REKAM] Perekaman dihentikan (IDLE). Total data terkumpul: {len(collected_rows)} frame.")

    finally:
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()

        # Save collected data to CSV
        if collected_rows:
            import csv
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = PROJECT_ROOT / "07_results" / "live_camera_tests"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"dual_live_test_{timestamp_str}.csv"

            fieldnames = list(collected_rows[0].keys())
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(collected_rows)

            # Print summary
            import pandas as pd
            df = pd.DataFrame(collected_rows)
            total = len(df)
            correct = df["correct"].sum()
            accuracy = correct / total * 100 if total > 0 else 0.0

            print("\n" + "=" * 80)
            print("  RINGKASAN HASIL PENGUJIAN DUAL-CAMERA LIVE")
            print("=" * 80)
            print(f"  Total frame terekam  : {total}")
            print(f"  Benar (correct)      : {correct}")
            print(f"  Akurasi keseluruhan  : {accuracy:.1f}%")
            print(f"  File disimpan di     : {out_path}")
            print("")

            # Per-class breakdown
            print("  Per-Kelas Postur:")
            print(f"  {'Postur':<20} {'Total':>6} {'Benar':>6} {'Akurasi':>8}")
            print("  " + "-" * 42)
            for label in MAIN_CLASSES:
                df_c = df[df["ground_truth"] == label]
                if len(df_c) > 0:
                    c_total = len(df_c)
                    c_correct = df_c["correct"].sum()
                    c_acc = c_correct / c_total * 100
                    print(f"  {label:<20} {c_total:>6} {c_correct:>6} {c_acc:>7.1f}%")
            print("=" * 80)
        else:
            print("\n[INFO] Tidak ada data pengujian dual-camera yang direkam.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time 2D Posture Inference Prototype")
    parser.add_argument("--select", action="store_true", help="Pilih kamera secara interaktif melalui menu terminal dan visual preview")
    parser.add_argument("--single-cam", action="store_true", help="Gunakan mode satu kamera (Live Laptop/Desk Testing)")
    parser.add_argument("--cam01-idx", type=int, default=None, help="Index kamera CAM01 (Frontal)")
    parser.add_argument("--cam02-idx", type=int, default=None, help="Index kamera CAM02 (Lateral)")
    parser.add_argument("--lateral-side", type=str, default="right", choices=["left", "right"], help="Sisi lateral kamera CAM02")
    parser.add_argument("--scan-cameras", action="store_true", help="Pindai kamera yang terhubung ke PC")

    args = parser.parse_args()

    if args.scan_cameras:
        cams = scan_available_cameras()
        if cams:
            print("\nKamera yang terdeteksi pada PC Anda:")
            for idx, w, h in cams:
                print(f"  - Kamera Index [{idx}]: Resolusi {w}x{h} px")
        else:
            print("Tidak ada kamera yang terdeteksi.")
        return

    # If --select or no explicit camera args given:
    if args.select or (not args.single_cam and args.cam01_idx is None and args.cam02_idx is None):
        cfg = interactive_camera_selection()
        if cfg["mode"] == "single":
            run_single_cam_live(cam_idx=cfg["cam01_idx"])
        else:
            run_dual_cam_live(
                cam01_idx=cfg["cam01_idx"],
                cam02_idx=cfg["cam02_idx"],
                lateral_side=cfg["lateral_side"]
            )
        return

    # Explicit single-cam
    if args.single_cam:
        c1 = args.cam01_idx if args.cam01_idx is not None else 0
        run_single_cam_live(cam_idx=c1)
        return

    # Explicit dual-cam
    c1 = args.cam01_idx if args.cam01_idx is not None else 0
    c2 = args.cam02_idx if args.cam02_idx is not None else 1
    try:
        run_dual_cam_live(cam01_idx=c1, cam02_idx=c2, lateral_side=args.lateral_side)
    except Exception as e:
        print(f"\n[INFO] Gagal membuka kamera ganda Port {c1} & {c2}: {e}")
        print(f">>> Membuka otomatis dalam mode SINGLE-CAMERA (Port {c1})...")
        run_single_cam_live(cam_idx=c1)


if __name__ == "__main__":
    main()

