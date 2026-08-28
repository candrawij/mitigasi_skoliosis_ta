"""
Synchronized Dual-Camera Calibration Pair Acquisition & Auto-Calibrator.

Features:
  1. Live split-screen GUI showing CAM01 & CAM02 simultaneously.
  2. Real-time checkerboard corner detection overlay on BOTH views.
  3. Single-button trigger ([SPACE]) captures matching pairs simultaneously:
     - 02_data/private_calibration/raw_frames/CAM01/pair_001.jpg
     - 02_data/private_calibration/raw_frames/CAM02/pair_001.jpg
  4. Instant auto-pipeline on finish ([ENTER]):
     - Calibrates CAM01 Intrinsic
     - Calibrates CAM02 Intrinsic
     - Computes Stereo Calibration (CAL_001)
     - Validates Reprojection Error & saves Epipolar Check image

Usage:
  python 04_scripts/calibration/capture_stereo_calibration_pairs.py \
      --calibration_id CAL_001 --cam01_idx 0 --cam02_idx 1
"""
import os
import sys
import cv2
import json
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path

# Ensure UTF-8 output
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CALIB_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "raw_frames"
SCRIPTS_DIR = PROJECT_ROOT / "04_scripts" / "calibration"


def run_stereo_pair_capture(
    calibration_id="CAL_001",
    cam01_idx=0,
    cam02_idx=1,
    cols=9,
    rows=6,
    square_size_m=0.025
):
    cam01_dir = RAW_CALIB_DIR / "CAM01"
    cam02_dir = RAW_CALIB_DIR / "CAM02"
    cam01_dir.mkdir(parents=True, exist_ok=True)
    cam02_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  SYNCHRONIZED CALIBRATION CAPTURE: {calibration_id}")
    print(f"  CAM01 (Index {cam01_idx}) <---> CAM02 (Index {cam02_idx})")
    print("=" * 70)
    print("Petunjuk Penggunaan:")
    print("  1. Pegang papan checkerboard agar TERLIHAT DI KEDUA KAMERA sekaligus.")
    print("  2. Saat kedua view mendeteksi grid (overlay HIJAU), tekan [SPACE].")
    print("  3. Ubah jarak, sudut kemiringan, dan posisi papan (kiri, kanan, atas, bawah).")
    print("  4. Kumpulkan minimal 10 - 20 pasang frame yang stabil.")
    print("  5. Tekan [ENTER] setelah selesai untuk langsung memproses kalibrasi otomatis.")
    print("  6. Tekan [ESC] untuk batal.\n")

    # Check old pairs
    existing_p1 = list(cam01_dir.glob("pair_*.jpg"))
    if existing_p1:
        print(f">> Ditemukan {len(existing_p1)} pasang foto kalibrasi lama di folder.")
        clean_old = input("   Apakah ingin MENGHAPUS foto lama dan mulai kalibrasi baru dari awal? (Y/n) [default: Y]: ").strip().lower()
        if clean_old != "n":
            for f in cam01_dir.glob("pair_*.jpg"): f.unlink()
            for f in cam02_dir.glob("pair_*.jpg"): f.unlink()
            print("   [OK] Foto kalibrasi lama telah dibersihkan. Memulai sesi baru dari pair_001.")

    pair_count = len(list(cam01_dir.glob("pair_*.jpg")))

    cap1 = cv2.VideoCapture(cam01_idx)
    cap2 = cv2.VideoCapture(cam02_idx)

    if not cap1.isOpened() or not cap2.isOpened():
        print(f"[ERROR] Gagal membuka kamera. Pastikan index CAM01={cam01_idx} dan CAM02={cam02_idx} benar.")
        if cap1.isOpened(): cap1.release()
        if cap2.isOpened(): cap2.release()
        return

    # Set resolution
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            time.sleep(0.01)
            continue

        disp1 = frame1.copy()
        disp2 = frame2.copy()

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        found1, corners1 = cv2.findChessboardCorners(gray1, (cols, rows), None)
        found2, corners2 = cv2.findChessboardCorners(gray2, (cols, rows), None)

        if found1:
            cv2.drawChessboardCorners(disp1, (cols, rows), corners1, found1)
        if found2:
            cv2.drawChessboardCorners(disp2, (cols, rows), corners2, found2)

        # Resize for display side-by-side
        disp_w, disp_h = 640, 360
        r1 = cv2.resize(disp1, (disp_w, disp_h))
        r2 = cv2.resize(disp2, (disp_w, disp_h))
        combined = np.hstack((r1, r2))

        # Status text & color
        both_ready = found1 and found2
        if both_ready:
            status_text = f"READY TO CAPTURE! Press SPACE (Captured: {pair_count} pairs)"
            status_color = (0, 255, 0)
        elif found1 or found2:
            status_text = f"Warning: Checkerboard only seen by 1 camera! (Captured: {pair_count})"
            status_color = (0, 165, 255)
        else:
            status_text = f"Searching for checkerboard in both views... (Captured: {pair_count})"
            status_color = (0, 0, 255)

        # HUD header
        cv2.rectangle(combined, (0, 0), (2 * disp_w, 40), (40, 40, 40), -1)
        cv2.putText(combined, status_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(combined, f"CAM01 ({'FOUND' if found1 else 'MISS'})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, f"CAM02 ({'FOUND' if found2 else 'MISS'})", (disp_w + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.line(combined, (disp_w, 0), (disp_w, disp_h), (255, 255, 255), 2)

        cv2.imshow("Dual-Camera Calibration Pair Capture", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE
            if both_ready:
                pair_count += 1
                fname1 = cam01_dir / f"pair_{pair_count:03d}.jpg"
                fname2 = cam02_dir / f"pair_{pair_count:03d}.jpg"
                cv2.imwrite(str(fname1), frame1)
                cv2.imwrite(str(fname2), frame2)
                print(f"  [SAVED] Pair #{pair_count:03d} -> {fname1.name} & {fname2.name}")
            else:
                print("  [BLOCKED] Papan harus terdeteksi di KEDUA kamera sekaligus sebelum menekan SPACE!")
        elif key in [13, ord('q'), ord('Q')]:  # ENTER / Q
            break
        elif key == 27:  # ESC
            print("Pengambilan dibatalkan.")
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            return

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    print(f"\n[OK] Selesai mengambil {pair_count} pasang citra kalibrasi.")

    if pair_count >= 5:
        print("\n" + "=" * 70)
        print("  MEMPROSES KALIBRASI OTOMATIS...")
        print("=" * 70)
        
        # 1. Calibrate Intrinsic CAM01
        print("\n[1/3] Menghitung Kalibrasi Intrinsik CAM01...")
        subprocess.run([sys.executable, str(SCRIPTS_DIR / "calibrate_intrinsic.py"), "--camera_id", "CAM01", "--input_dir", str(cam01_dir)])
        
        # 2. Calibrate Intrinsic CAM02
        print("\n[2/3] Menghitung Kalibrasi Intrinsik CAM02...")
        subprocess.run([sys.executable, str(SCRIPTS_DIR / "calibrate_intrinsic.py"), "--camera_id", "CAM02", "--input_dir", str(cam02_dir)])
        
        # 3. Stereo Calibration
        print("\n[3/3] Menghitung Kalibrasi Stereo Dual-Camera...")
        c1_int = PROJECT_ROOT / "02_data" / "private_calibration" / "intrinsic" / "CAM01_intrinsic.json"
        c2_int = PROJECT_ROOT / "02_data" / "private_calibration" / "intrinsic" / "CAM02_intrinsic.json"
        
        cmd_stereo = [
            sys.executable, str(SCRIPTS_DIR / "calibrate_stereo.py"),
            "--calibration_id", calibration_id,
            "--cam01_dir", str(cam01_dir),
            "--cam02_dir", str(cam02_dir),
            "--cam01_intrinsic", str(c1_int),
            "--cam02_intrinsic", str(c2_int)
        ]
        subprocess.run(cmd_stereo)
        
        # 4. Validation
        print("\n[VALIDATOR] Menjalankan Evaluasi Mutu Kalibrasi & Epipolar Line...")
        p1_sample = sorted(list(cam01_dir.glob("pair_*.jpg")))[0]
        p2_sample = sorted(list(cam02_dir.glob("pair_*.jpg")))[0]
        subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "validate_calibration.py"),
            "--calibration_id", calibration_id,
            "--cam01_sample", str(p1_sample),
            "--cam02_sample", str(p2_sample)
        ])
    else:
        print(f"[WARNING] Jumlah pasang citra ({pair_count}) kurang dari 5. Kalibrasi membutuhkan minimal 5-10 pasang.")


def main():
    config_file = PROJECT_ROOT / "03_metadata" / "camera_config.json"
    def_cam01 = 0
    def_cam02 = 2
    def_cal_id = "CAL_001"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                def_cam01 = cfg.get("cam01_idx", 0)
                def_cam02 = cfg.get("cam02_idx", 2)
                def_cal_id = cfg.get("calibration_id", "CAL_001")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Synchronized Dual-Camera Calibration Capture")
    parser.add_argument("--calibration_id", type=str, default=def_cal_id, help="Calibration setup ID")
    parser.add_argument("--cam01_idx", type=int, default=def_cam01, help=f"Device index for CAM01 (default: {def_cam01})")
    parser.add_argument("--cam02_idx", type=int, default=def_cam02, help=f"Device index for CAM02 (default: {def_cam02})")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners horizontal")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners vertical")
    parser.add_argument("--square_size", type=float, default=0.025, help="Square size in meters (0.025 = 25mm)")

    args = parser.parse_args()
    run_stereo_pair_capture(
        calibration_id=args.calibration_id,
        cam01_idx=args.cam01_idx,
        cam02_idx=args.cam02_idx,
        cols=args.cols,
        rows=args.rows,
        square_size_m=args.square_size
    )


if __name__ == "__main__":
    main()
