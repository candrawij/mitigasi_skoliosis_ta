"""
live_camera_test_snapshot.py — Capture live test frame from Camera 0, run trained XGBoost model, and save HUD visual.
"""
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "inference"))
from private_inference_common import infer_single_cam_2d
from infer_realtime_2d import build_hud_single_cam

OUT_DIR = PROJECT_ROOT / "07_results" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "live_camera_test_snapshot.jpg"


def capture_test_snapshot():
    print("Membuka kamera 0 untuk mengambil snapshot uji...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Tidak dapat mengakses Kamera Index 0.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Allow camera sensor to auto-adjust exposure and white balance (warm-up)
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: Gagal membaca frame dari kamera.")
        return False

    print(f"Frame berhasil ditangkap ({frame.shape[1]}x{frame.shape[0]} px).")
    print("Menjalankan inferensi model XGBoost hasil training dataset privat...")

    result = infer_single_cam_2d(frame, desk_mode=True)
    hud_frame = build_hud_single_cam(frame, result, fps=30.0)

    cv2.imwrite(str(OUT_PATH), hud_frame)
    print(f"\n[SUKSES] Snapshot visual tersimpan di: {OUT_PATH}")
    print(f"Status Deteksi : {result.get('status')}")
    print(f"Prediksi Postur: {result.get('prediction', 'REJECT').upper()}")
    print(f"Konfidensi     : {result.get('confidence', 0.0)*100:.1f}%")
    print(f"Hips Extrapol  : {result.get('is_extrapolated')}")
    print("Probabilitas Kelas:")
    for c_name, prob in result.get("probabilities", {}).items():
        print(f"  - {c_name:<18}: {prob*100:5.1f}%")

    return True


if __name__ == "__main__":
    capture_test_snapshot()
