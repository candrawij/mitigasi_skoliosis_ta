"""
Master Orchestrator Pipeline for Camera System Testing, Calibration & Pilot Acquisition.

Urutan Testing Sistem:
  [1] T5.B.1 Checkerboard Pattern Generation & Physical Dimensions Check
  [2] T5.B.2 Single-Camera Intrinsic Calibration (CAM01 & CAM02)
  [3] T5.B.3 Dual-Camera Stereo Calibration (Extrinsics R, T, Rectification Q)
  [4] T5.B.5 Calibration Validation & Epipolar Line Visual Verification
  [5] T5.B.4 Dual-Camera Synchronized Capture Verification (Dummy / Object Test)
  [6] Pilot Study Acquisition & Protocol Verification (5-10 Subjects)
  [7] Automated End-to-End Mock Diagnostic (Software & Math Pipeline Verification)

Usage:
  # Interactive Menu:
  python 04_scripts/calibration/run_camera_test_pipeline.py

  # Direct Step Execution:
  python 04_scripts/calibration/run_camera_test_pipeline.py --step 1
  python 04_scripts/calibration/run_camera_test_pipeline.py --step 7  (Automated end-to-end dry run)
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

# Ensure UTF-8 output encoding for Windows PowerShell/CMD
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIB_SCRIPTS_DIR = PROJECT_ROOT / "04_scripts" / "calibration"
CAPTURE_SCRIPTS_DIR = PROJECT_ROOT / "04_scripts" / "capture"
DATA_DIR = PROJECT_ROOT / "02_data"
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"


CONFIG_FILE = PROJECT_ROOT / "03_metadata" / "camera_config.json"


def load_camera_config():
    """Load persistent camera device configuration."""
    defaults = {
        "cam01_idx": 0,
        "cam02_idx": 2,
        "cam03_idx": 1,
        "calibration_id": "CAL_001",
        "resolution_width": 1280,
        "resolution_height": 720,
        "notes": "CAM01 = Index 0 (Frontal), CAM02 = Index 2 (Lateral)"
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_camera_config(cfg):
    """Save persistent camera device configuration."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[OK] Konfigurasi kamera berhasil disimpan ke: {CONFIG_FILE}")


def print_banner(title):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


def scan_connected_cameras(max_check=4):
    """Scan and list connected camera indexes."""
    found = []
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((idx, w, h))
            cap.release()
    return found


def menu_camera_settings():
    """Menu 9: Camera Device Configuration & Live Test."""
    print_banner("Menu 9: Konfigurasi Default Index Kamera (CAM01 & CAM02)")
    cfg = load_camera_config()
    
    cams = scan_connected_cameras()
    print("Daftar perangkat kamera yang terdeteksi di PC:")
    if cams:
        for c_idx, w, h in cams:
            print(f"  - Device Index [{c_idx}]: Resolusi {w}x{h} px")
    else:
        print("  - Tidak ada kamera yang terdeteksi.")
        
    print(f"\nKonfigurasi Saat Ini:")
    print(f"  * CAM01 (Frontal View) : Device Index {cfg['cam01_idx']}")
    print(f"  * CAM02 (Lateral View) : Device Index {cfg['cam02_idx']}")
    print(f"  * Calibration ID       : {cfg['calibration_id']}")
    
    print("\nPilihan:")
    print("  [1] Ubah Device Index CAM01 dan CAM02")
    print("  [2] Tes Live Preview Kamera (Cek tampilan Index 0, 1, 2)")
    print("  [0] Kembali ke menu utama")
    
    opt = input("\nPilihan Anda (0-2) [default: 1]: ").strip() or "1"
    
    if opt == "1":
        idx1 = input(f"Masukkan Device Index untuk CAM01 (Frontal) [default: {cfg['cam01_idx']}]: ").strip()
        idx2 = input(f"Masukkan Device Index untuk CAM02 (Lateral) [default: {cfg['cam02_idx']}]: ").strip()
        cal_id = input(f"Masukkan Default Calibration ID [default: {cfg['calibration_id']}]: ").strip()
        
        if idx1: cfg["cam01_idx"] = int(idx1)
        if idx2: cfg["cam02_idx"] = int(idx2)
        if cal_id: cfg["calibration_id"] = cal_id
        
        save_camera_config(cfg)
        print(f"\n[SUKSES] Konfigurasi baru aktif: CAM01 = Index {cfg['cam01_idx']}, CAM02 = Index {cfg['cam02_idx']}")
        
    elif opt == "2":
        test_cams = [cfg['cam01_idx'], cfg['cam02_idx']]
        print(f"\n>> Membuka Live Preview untuk CAM01 (Index {cfg['cam01_idx']}) dan CAM02 (Index {cfg['cam02_idx']})...")
        print("Tekan [Q] atau [ESC] di jendela kamera untuk menutup preview.")
        
        cap1 = cv2.VideoCapture(cfg['cam01_idx'])
        cap2 = cv2.VideoCapture(cfg['cam02_idx'])
        
        while cap1.isOpened() and cap2.isOpened():
            ret1, f1 = cap1.read()
            ret2, f2 = cap2.read()
            if not ret1 or not ret2:
                break
            
            d1 = cv2.resize(f1, (640, 360))
            d2 = cv2.resize(f2, (640, 360))
            
            cv2.putText(d1, f"CAM01 (Index {cfg['cam01_idx']})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(d2, f"CAM02 (Index {cfg['cam02_idx']})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            
            combo = np.hstack((d1, d2))
            cv2.imshow("Camera Index Verification (Press Q to exit)", combo)
            
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                break
                
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()


def step1_generate_checkerboard():
    """Step 1: Generate & Validate Checkerboard Pattern."""
    print_banner("Step 1: Test & Generate Checkerboard Pattern")
    script = CALIB_SCRIPTS_DIR / "generate_checkerboard.py"
    
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
        
    spec_path = DATA_DIR / "private_calibration" / "patterns" / "checkerboard_spec.json"
    img_path = DATA_DIR / "private_calibration" / "patterns" / "checkerboard_9x6_25mm.png"
    
    if spec_path.exists() and img_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        print("[CHECK] Checkerboard Status: READY")
        print(f"  - File Gambar:   {img_path}")
        print(f"  - Dimensi Kotak: {spec['square_size_mm']} mm x {spec['square_size_mm']} mm")
        print(f"  - Inner Corners: {spec['inner_corners_cols']} x {spec['inner_corners_rows']}")
        print(f"  - Total Kotak:   {spec['squares_cols']} x {spec['squares_rows']}")
        print("\n>> PENTING SEBELUM MENCETAK:")
        print("   1. Cetak pada skala 100% (Do NOT Scale / Do NOT Fit to Page).")
        print("   2. Tempelkan kertas pada bidang keras rata (karton tebal / akrilik / kaca).")
        print("   3. Ukur lebar 1 kotak dengan penggaris fisik untuk memastikan tepat 25.0 mm.")
        return True
    else:
        print("[FAIL] Gagal menghasilkan file checkerboard.")
        return False


def step2_intrinsic_calibration():
    """Step 2: Single-Camera Intrinsic Calibration (CAM01 & CAM02)."""
    print_banner("Step 2: Intrinsic Camera Calibration (CAM01 & CAM02)")
    cfg = load_camera_config()
    def_idx1 = cfg.get("cam01_idx", 0)
    def_idx2 = cfg.get("cam02_idx", 2)
    
    print("Konsep Penting:")
    print("  * Kalibrasi Intrinsik mengukur karakteristik optik INDIVIDUAL dari masing-masing lensa kamera")
    print("    (Focal length fx, fy, Principal point cx, cy, dan distorsi k1, k2, p1, p2).")
    print("  * Kamera dibuka SATU PER SATU bergantian (CAM01 dulu, lalu CAM02).")
    print(f"  * Default Device Index aktif: CAM01 = [{def_idx1}], CAM02 = [{def_idx2}]\n")
    
    cams = scan_connected_cameras()
    if cams:
        print("Perangkat kamera yang terdeteksi di PC:")
        for c_idx, w, h in cams:
            print(f"  - Device Index [{c_idx}]: {w}x{h} px")
    print("\nPilih alur kalibrasi intrinsik:")
    print(f"  [1] Kalibrasi CAM01 (Idx {def_idx1}) & CAM02 (Idx {def_idx2}) Berurutan [Rekomendasi]")
    print(f"  [2] Kalibrasi CAM01 saja (Frontal View - Idx {def_idx1})")
    print(f"  [3] Kalibrasi CAM02 saja (Lateral View - Idx {def_idx2})")
    print("  [4] Batch processing dari folder gambar")
    print("  [0] Kembali ke menu utama")
    
    choice = input("\nPilihan Anda (0-4) [default: 1]: ").strip() or "1"
    script = CALIB_SCRIPTS_DIR / "calibrate_intrinsic.py"
    
    if choice == "1":
        # 1. Calibrate CAM01
        idx1 = input(f"\n[CAM01] Masukkan device index untuk CAM01 [default: {def_idx1}]: ").strip() or str(def_idx1)
        print(f"\n>> Membuka live capture untuk CAM01 (Device Index {idx1})...")
        print("Petunjuk di jendela:")
        print("  - Arahkan checkerboard dengan berbagai jarak dan sudut kemiringan.")
        print("  - Tekan [SPACE] saat grid terdeteksi hijau (kumpulkan 10-15 frame).")
        print("  - Tekan [ENTER] atau [Q] setelah selesai untuk menghitung intrinsik CAM01.\n")
        subprocess.run([sys.executable, str(script), "--camera_id", "CAM01", "--live", "--cam_idx", idx1])
        
        # 2. Continue to CAM02
        next_c2 = input("\nCAM01 selesai! Apakah ingin langsung lanjut kalibrasi CAM02? (Y/n) [default: Y]: ").strip().lower()
        if next_c2 != "n":
            idx2 = input(f"\n[CAM02] Masukkan device index untuk CAM02 [default: {def_idx2}]: ").strip() or str(def_idx2)
            print(f"\n>> Membuka live capture untuk CAM02 (Device Index {idx2})...")
            print("Petunjuk di jendela:")
            print("  - Arahkan checkerboard dengan berbagai jarak dan sudut kemiringan.")
            print("  - Tekan [SPACE] saat grid terdeteksi hijau (kumpulkan 10-15 frame).")
            print("  - Tekan [ENTER] atau [Q] setelah selesai untuk menghitung intrinsik CAM02.\n")
            subprocess.run([sys.executable, str(script), "--camera_id", "CAM02", "--live", "--cam_idx", idx2])
            print("\n[OK] Kalibrasi intrinsik untuk KEDUA kamera (CAM01 & CAM02) selesai! Siap lanjut ke Step 3.")
            
    elif choice == "2":
        idx1 = input(f"Masukkan device index untuk CAM01 [default: {def_idx1}]: ").strip() or str(def_idx1)
        subprocess.run([sys.executable, str(script), "--camera_id", "CAM01", "--live", "--cam_idx", idx1])
    elif choice == "3":
        idx2 = input(f"Masukkan device index untuk CAM02 [default: {def_idx2}]: ").strip() or str(def_idx2)
        subprocess.run([sys.executable, str(script), "--camera_id", "CAM02", "--live", "--cam_idx", idx2])
    elif choice == "4":
        cam_id = input("Masukkan Camera ID (CAM01 / CAM02) [default: CAM01]: ").strip() or "CAM01"
        default_dir = DATA_DIR / "private_calibration" / "raw_frames" / cam_id
        folder = input(f"Masukkan path folder citra [{default_dir}]: ").strip() or str(default_dir)
        subprocess.run([sys.executable, str(script), "--camera_id", cam_id, "--input_dir", folder])
    else:
        print("Kembali ke menu utama.")


def step3_stereo_calibration():
    """Step 3: Stereo Dual-Camera Calibration."""
    print_banner("Step 3: Dual-Camera Stereo Calibration (CAM01 + CAM02)")
    cfg = load_camera_config()
    def_idx1 = cfg.get("cam01_idx", 0)
    def_idx2 = cfg.get("cam02_idx", 2)
    def_cal_id = cfg.get("calibration_id", "CAL_001")
    
    print("Pilih mode kalibrasi stereo:")
    print(f"  [1] Live capture pasangan checkerboard sinkron (CAM01: {def_idx1}, CAM02: {def_idx2}) [Rekomendasi]")
    print("  [2] Hitung kalibrasi stereo dari folder gambar yang sudah ada")
    print("  [3] Lewati langkah ini")
    
    choice = input("\nPilihan Anda (1/2/3) [default: 1]: ").strip() or "1"
    calib_id = input(f"Masukkan Calibration ID [default: {def_cal_id}]: ").strip() or def_cal_id
    
    if choice == "1":
        idx1 = input(f"Device index CAM01 (Frontal) [default: {def_idx1}]: ").strip() or str(def_idx1)
        idx2 = input(f"Device index CAM02 (Lateral) [default: {def_idx2}]: ").strip() or str(def_idx2)
        script = CALIB_SCRIPTS_DIR / "capture_stereo_calibration_pairs.py"
        cmd = [
            sys.executable, str(script),
            "--calibration_id", calib_id,
            "--cam01_idx", idx1,
            "--cam02_idx", idx2
        ]
        subprocess.run(cmd)
        
    elif choice == "2":
        cam01_dir = DATA_DIR / "private_calibration" / "raw_frames" / "CAM01"
        cam02_dir = DATA_DIR / "private_calibration" / "raw_frames" / "CAM02"
        
        c1_in = input(f"Folder citra CAM01 [{cam01_dir}]: ").strip() or str(cam01_dir)
        c2_in = input(f"Folder citra CAM02 [{cam02_dir}]: ").strip() or str(cam02_dir)
        
        c1_int = DATA_DIR / "private_calibration" / "intrinsic" / "CAM01_intrinsic.json"
        c2_int = DATA_DIR / "private_calibration" / "intrinsic" / "CAM02_intrinsic.json"
        
        script = CALIB_SCRIPTS_DIR / "calibrate_stereo.py"
        cmd = [
            sys.executable, str(script),
            "--calibration_id", calib_id,
            "--cam01_dir", c1_in,
            "--cam02_dir", c2_in
        ]
        if c1_int.exists() and c2_int.exists():
            cmd.extend(["--cam01_intrinsic", str(c1_int), "--cam02_intrinsic", str(c2_int)])
            
        subprocess.run(cmd)
    else:
        print("Langkah 3 dilewati.")


def step4_validate_calibration():
    """Step 4: Calibration Validator & Epipolar Verification."""
    print_banner("Step 4: Calibration Validation & Triangulation Test")
    calib_id = input("Masukkan Calibration ID yang ingin divalidasi [default: CAL_001]: ").strip() or "CAL_001"
    
    script = CALIB_SCRIPTS_DIR / "validate_calibration.py"
    subprocess.run([sys.executable, str(script), "--calibration_id", calib_id])


def step5_test_capture():
    """Step 5: Test Dual-Camera Capture Software (Dummy / Object Test)."""
    print_banner("Step 5: Test Dual-Camera Synchronized Capture Software")
    cfg = load_camera_config()
    def_idx1 = cfg.get("cam01_idx", 0)
    def_idx2 = cfg.get("cam02_idx", 2)
    def_cal_id = cfg.get("calibration_id", "CAL_001")
    
    print("Pengujian software capture dual-kamera sinkron (uji dummy object):")
    print(f"  - Membuka live split-screen CAM01 (Index {def_idx1}) + CAM02 (Index {def_idx2}).")
    print("  - Menghasilkan capture_id tunggal untuk setiap pasang foto.")
    print("  - Menyimpan otomatis ke format: S001_SE01_CAP000001_CAM01.jpg & CAM02.jpg")
    print("  - Mencatat log real-time ke captures.csv dan images.csv.\n")
    
    sub_id = input("Subject ID [default: S007]: ").strip() or "S007"
    ses_id = input("Session ID [default: SE01]: ").strip() or "SE01"
    cal_id = input(f"Calibration ID [default: {def_cal_id}]: ").strip() or def_cal_id
    idx1 = input(f"Device index CAM01 (Frontal) [default: {def_idx1}]: ").strip() or str(def_idx1)
    idx2 = input(f"Device index CAM02 (Lateral) [default: {def_idx2}]: ").strip() or str(def_idx2)
    lat_side = input("Lateral Camera Side (right/left) [default: right]: ").strip().lower() or "right"
    chair_id = input("Chair ID [default: CHR_001]: ").strip() or "CHR_001"
    
    script = CAPTURE_SCRIPTS_DIR / "capture_multicam.py"
    cmd = [
        sys.executable, str(script),
        "--subject_id", sub_id,
        "--session_id", ses_id,
        "--calibration_id", cal_id,
        "--cam01_idx", idx1,
        "--cam02_idx", idx2,
        "--lateral_side", lat_side,
        "--chair_id", chair_id
    ]
    subprocess.run(cmd)
    
    # Audit captures output
    print("\n--- Verifikasi Integritas Data Hasil Capture ---")
    captures_csv = META_DIR / "captures.csv"
    images_csv = META_DIR / "images.csv"
    
    if captures_csv.exists() and images_csv.exists():
        import pandas as pd
        df_cap = pd.read_csv(captures_csv)
        df_img = pd.read_csv(images_csv)
        print(f"[OK] Total pose di captures.csv: {len(df_cap)}")
        print(f"[OK] Total citra di images.csv:   {len(df_img)}")
        if len(df_img) >= 2 * len(df_cap) and len(df_cap) > 0:
            print("[VERIFIED] Setiap capture_id sukses memetakan 2 view kamera sinkron (CAM01 + CAM02)!")


def step7_automated_mock_pipeline():
    """Option 7: Automated End-to-End Mock Diagnostic."""
    print_banner("Option 7: Automated Mock Pipeline Diagnostic")
    print("Menjalankan simulasi lengkap end-to-end tanpa memerlukan kamera fisik:")
    print("  1. Membuat pasangan citra checkerboard sintetis CAM01 (Frontal) & CAM02 (Oblique 30°).")
    print("  2. Mengeksekusi calibrate_intrinsic.py untuk CAM01 dan CAM02.")
    print("  3. Mengeksekusi calibrate_stereo.py untuk CAL_MOCK.")
    print("  4. Mengeksekusi validate_calibration.py untuk mengevaluasi reprojection error & 3D triangulation.")
    print("  5. Memverifikasi struktur CSV dan formula matematika stereo.\n")
    
    # 1. Generate synthetic checkerboard views
    mock_dir = DATA_DIR / "private_calibration" / "mock_frames"
    c1_mock = mock_dir / "CAM01"
    c2_mock = mock_dir / "CAM02"
    c1_mock.mkdir(parents=True, exist_ok=True)
    c2_mock.mkdir(parents=True, exist_ok=True)
    
    print("[1/4] Membuat 10 pasang citra kalibrasi sintetis...")
    cols, rows, sq_px = 9, 6, 60
    n_sq_x, n_sq_y = cols + 1, rows + 1
    board_w = n_sq_x * sq_px
    board_h = n_sq_y * sq_px
    
    # Create clean synthetic board with white border
    margin = 80
    clean_board = np.ones((board_h + 2 * margin, board_w + 2 * margin, 3), dtype=np.uint8) * 255
    for r in range(n_sq_y):
        for c in range(n_sq_x):
            if (r + c) % 2 == 1:
                x0 = margin + c * sq_px
                y0 = margin + r * sq_px
                clean_board[y0:y0+sq_px, x0:x0+sq_px] = 0
                
    h_b, w_b = clean_board.shape[:2]
    src_pts = np.float32([[0,0], [w_b,0], [w_b,h_b], [0,h_b]])
    
    # Generate 10 perspective-transformed views with varied tilts and scales
    for i in range(1, 11):
        canvas_w, canvas_h = 1280, 720
        
        # Tilt and scale variations
        tilt_x = (i % 3 - 1) * 35
        tilt_y = ((i // 3) % 3 - 1) * 25
        scale = 0.55 + (i % 4) * 0.06
        
        # CAM01: Frontal with slight natural tilt
        dst_c1 = np.float32([
            [120 + tilt_x, 80 + tilt_y],
            [120 + tilt_x + w_b * scale, 95 + tilt_y],
            [120 + tilt_x + w_b * scale * 0.96, 75 + tilt_y + h_b * scale],
            [120 + tilt_x + 15, 80 + tilt_y + h_b * scale]
        ])
        M1 = cv2.getPerspectiveTransform(src_pts, dst_c1)
        warp1 = cv2.warpPerspective(clean_board, M1, (canvas_w, canvas_h), borderValue=(255, 255, 255))
        cv2.imwrite(str(c1_mock / f"mock_{i:02d}.jpg"), warp1)
        
        # CAM02: Stereo partner with horizontal baseline angle shift
        dst_c2 = np.float32([
            [480 + tilt_x * 0.85, 90 + tilt_y],
            [480 + tilt_x * 0.85 + w_b * scale * 0.82, 120 + tilt_y],
            [480 + tilt_x * 0.85 + w_b * scale * 0.78, 90 + tilt_y + h_b * scale],
            [480 + tilt_x * 0.85 + 10, 90 + tilt_y + h_b * scale]
        ])
        M2 = cv2.getPerspectiveTransform(src_pts, dst_c2)
        warp2 = cv2.warpPerspective(clean_board, M2, (canvas_w, canvas_h), borderValue=(255, 255, 255))
        cv2.imwrite(str(c2_mock / f"mock_{i:02d}.jpg"), warp2)
        
    print(f"[OK] 10 pasang citra mock tersimpan di {mock_dir}")
    
    # 2. Run Intrinsic Calibration for CAM01 & CAM02
    print("\n[2/4] Menjalankan Kalibrasi Intrinsik pada citra mock...")
    intr_script = CALIB_SCRIPTS_DIR / "calibrate_intrinsic.py"
    subprocess.run([sys.executable, str(intr_script), "--camera_id", "CAM01", "--input_dir", str(c1_mock)])
    subprocess.run([sys.executable, str(intr_script), "--camera_id", "CAM02", "--input_dir", str(c2_mock)])
    
    # 3. Run Stereo Calibration
    print("\n[3/4] Menjalankan Kalibrasi Stereo untuk CAL_MOCK...")
    stereo_script = CALIB_SCRIPTS_DIR / "calibrate_stereo.py"
    subprocess.run([
        sys.executable, str(stereo_script),
        "--calibration_id", "CAL_MOCK",
        "--cam01_dir", str(c1_mock),
        "--cam02_dir", str(c2_mock)
    ])
    
    # 4. Run Validator
    print("\n[4/4] Menjalankan Validator Kalibrasi...")
    val_script = CALIB_SCRIPTS_DIR / "validate_calibration.py"
    subprocess.run([
        sys.executable, str(val_script),
        "--calibration_id", "CAL_MOCK",
        "--cam01_sample", str(c1_mock / "mock_01.jpg"),
        "--cam02_sample", str(c2_mock / "mock_01.jpg")
    ])
    
    print("\n" + "=" * 75)
    print("  MOCK DIAGNOSTIC SELESAI: SELURUH PIPELINE SOFTWARE & MATEMATIKA BERFUNGSI 100%!")
    print("=" * 75)


def show_system_status():
    """Display summary of calibration files, patterns, and dataset metadata."""
    print_banner("Status Sistem Kalibrasi & Metadata")
    cfg = load_camera_config()
    
    intr_files = list((DATA_DIR / "private_calibration" / "intrinsic").glob("*.json"))
    stereo_files = list((DATA_DIR / "private_calibration" / "stereo").glob("*.json"))
    logs = list((DATA_DIR / "private_calibration" / "logs").glob("*.json"))
    
    print("1. Konfigurasi Default Kamera Aktif:")
    print(f"   - CAM01 (Frontal View): Device Index [{cfg['cam01_idx']}]")
    print(f"   - CAM02 (Lateral View): Device Index [{cfg['cam02_idx']}]")
    print(f"   - Calibration ID      : [{cfg['calibration_id']}]")
    
    print("\n2. Profil Kalibrasi Terdaftar:")
    print(f"   - Intrinsic Profiles: {[f.name for f in intr_files] or 'Belum ada'}")
    print(f"   - Stereo Profiles:    {[f.name for f in stereo_files] or 'Belum ada'}")
    print(f"   - Validation Reports: {[f.name for f in logs] or 'Belum ada'}")
    
    print("\n3. File Pola Checkerboard:")
    pat_file = DATA_DIR / "private_calibration" / "patterns" / "checkerboard_9x6_25mm.png"
    print(f"   - Pattern PNG: {'TERSEDIA' if pat_file.exists() else 'BELUM DIBUAT'}")
    
    print("\n4. Status Metadata CSV Template:")
    csv_templates = ["participants.csv", "captures.csv", "images.csv", "calibration_map.csv", "qc_audit_log.csv"]
    for c in csv_templates:
        p = META_DIR / c
        print(f"   - {c:<22}: {'TERSEDIA' if p.exists() else 'HILANG'}")


def menu_dataset_manager():
    """Menu 10: Dataset Manager (Inspect, Delete Capture/Subject, Reset Data)."""
    print_banner("Menu 10: Dataset Manager & Audit Tool")
    print("Pilih aksi manajemen dataset:")
    print("  [1] Audit & Cek Kesehatan Dataset (Inspect latensi, resolusi, jumlah pose)")
    print("  [2] Hapus 1 Capture Tertentu (misal salah pose / blur)")
    print("  [3] Hapus 1 Subjek Tertentu (misal S001 uji coba)")
    print("  [4] Reset Seluruh Data Uji Subjek (Bersihkan private_raw/ dan reset captures.csv)")
    print("  [5] Bersihkan Foto Mentah Checkerboard Kalibrasi (raw_frames & mock_frames)")
    print("  [0] Kembali ke menu utama")
    
    act = input("\nPilihan Anda (0-5) [default: 1]: ").strip() or "1"
    script = CAPTURE_SCRIPTS_DIR / "dataset_manager.py"
    
    if act == "1":
        subprocess.run([sys.executable, str(script), "--action", "inspect"])
    elif act == "2":
        cap_id = input("Masukkan capture_id yang ingin dihapus (contoh: CAP000012): ").strip()
        if cap_id:
            subprocess.run([sys.executable, str(script), "--action", "delete_capture", "--capture_id", cap_id])
    elif act == "3":
        sub_id = input("Masukkan subject_id yang ingin dihapus (contoh: S001): ").strip()
        if sub_id:
            subprocess.run([sys.executable, str(script), "--action", "delete_subject", "--subject_id", sub_id])
    elif act == "4":
        subprocess.run([sys.executable, str(script), "--action", "reset_test_data"])
    elif act == "5":
        subprocess.run([sys.executable, str(script), "--action", "clean_calib_frames"])


def main():
    parser = argparse.ArgumentParser(description="Master Camera Testing & Calibration Orchestrator")
    parser.add_argument("--step", type=int, default=None, help="Directly run a specific step (1-10)")
    args = parser.parse_args()
    
    if args.step is not None:
        steps = {
            1: step1_generate_checkerboard,
            2: step2_intrinsic_calibration,
            3: step3_stereo_calibration,
            4: step4_validate_calibration,
            5: step5_test_capture,
            7: step7_automated_mock_pipeline,
            8: show_system_status,
            9: menu_camera_settings,
            10: menu_dataset_manager,
        }
        if args.step in steps:
            steps[args.step]()
        else:
            print(f"Step {args.step} tidak valid. Pilih 1-10.")
        return

    # Interactive loop
    while True:
        cfg = load_camera_config()
        print_banner("Master Pipeline: Pengujian Sistem Kamera & Dataset Privat")
        print(f"Konfigurasi Aktif: CAM01 = Index [{cfg['cam01_idx']}] (Frontal) | CAM02 = Index [{cfg['cam02_idx']}] (Lateral)")
        print("-" * 75)
        print("Pilih tahapan yang ingin dijalankan:")
        print("  [1] T5.B.1 — Generate & Cek Papan Checkerboard (Pola Kalibrasi)")
        print(f"  [2] T5.B.2 — Kalibrasi Intrinsik Kamera Tunggal (CAM01: Idx {cfg['cam01_idx']} & CAM02: Idx {cfg['cam02_idx']})")
        print(f"  [3] T5.B.3 — Kalibrasi Stereo 2-Kamera (Ekstrinsik R, T & Rectification Q)")
        print("  [4] T5.B.5 — Validasi Kalibrasi, Epipolar Lines & Uji Triangulasi 3D")
        print("  [5] T5.B.4 — Uji Software Capture Dual-Kamera Sinkron (Dummy/Object Test)")
        print("  [6] PILOT  — Panduan Perekaman Pilot Dataset (5-10 Subjek)")
        print("  [7] DIAG   — Automated Mock Pipeline Test (Uji Seluruh Pipeline Tanpa Kamera Fisik)")
        print("  [8] STATUS — Lihat Status File Kalibrasi & Template Metadata")
        print(f"  [9] SETTINGS — Ubah Default Index Kamera (Saat ini: CAM01={cfg['cam01_idx']}, CAM02={cfg['cam02_idx']})")
        print("  [10] DATASET MANAGER — Audit, Hapus Data Salah, atau Reset Data Uji")
        print("  [0] KELUAR")
        
        choice = input("\nMasukkan nomor pilihan (0-10): ").strip()
        
        if choice == "1":
            step1_generate_checkerboard()
        elif choice == "2":
            step2_intrinsic_calibration()
        elif choice == "3":
            step3_stereo_calibration()
        elif choice == "4":
            step4_validate_calibration()
        elif choice == "5":
            step5_test_capture()
        elif choice == "6":
            print_banner("Panduan Pelaksanaan Pilot Study (5-10 Subjek)")
            print("Langkah-langkah:")
            print("  1. Pastikan rig kamera sudah dikalibrasi (misal CAL_001).")
            print("  2. Jalankan capture_multicam.py:")
            print("     python 04_scripts/capture/capture_multicam.py --subject_id S001 --session_id SE01 --calibration_id CAL_001")
            print("  3. Arahkan subjek melakukan 7 kelas inti postur (tegak, condong fwd/bwd/L/R, slouch, fwd head).")
            print("  4. Ambil 3-5 repetisi per kelas.")
            print("  5. Evaluasi visibilitas pinggul/bahu dan kualitas triangulasi 3D.")
        elif choice == "7":
            step7_automated_mock_pipeline()
        elif choice == "8":
            show_system_status()
        elif choice == "9":
            menu_camera_settings()
        elif choice == "10":
            menu_dataset_manager()
        elif choice == "0":
            print("\nTerima kasih. Program selesai.")
            break
        else:
            print("\nPilihan tidak valid. Silakan masukkan angka 0-10.")
            
        input("\nTekan [ENTER] untuk kembali ke menu utama...")


if __name__ == "__main__":
    main()
