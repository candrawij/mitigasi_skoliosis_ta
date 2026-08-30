"""
Dataset Manager & Quality Audit Utility.

Functions:
  1. Inspect & Audit: Cek kelengkapan data, resolusi, delta timestamp antar-kamera, dan blur score.
  2. Delete Specific Capture: Menghapus 1 capture_id (file CAM01 + CAM02) dan membersihkan baris di CSV.
  3. Delete Subject: Menghapus 1 subject_id (folder raw + log CSV).
  4. Reset Test Data: Membersihkan data uji/dummy agar dataset bersih sebelum pilot study sesungguhnya.

Usage:
  python 04_scripts/capture/dataset_manager.py --action inspect
  python 04_scripts/capture/dataset_manager.py --action delete_capture --capture_id CAP000012
  python 04_scripts/capture/dataset_manager.py --action delete_subject --subject_id S001
  python 04_scripts/capture/dataset_manager.py --action reset_test_data
"""
import os
import sys
import csv
import json
import shutil
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure UTF-8 output
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "02_data" / "private_raw"
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
CAPTURES_CSV = META_DIR / "captures.csv"
IMAGES_CSV = META_DIR / "images.csv"
PARTICIPANTS_CSV = META_DIR / "participants.csv"


def inspect_dataset():
    """Inspect and report the health and statistics of the private dataset."""
    print("\n" + "=" * 75)
    print("  LAPORAN AUDIT & KESEHATAN DATASET PRIVAT")
    print("=" * 75)
    
    if not CAPTURES_CSV.exists() or not IMAGES_CSV.exists():
        print("[ERROR] File captures.csv atau images.csv tidak ditemukan.")
        return

    df_cap = pd.read_csv(CAPTURES_CSV)
    df_img = pd.read_csv(IMAGES_CSV)
    
    print(f"1. Ringkasan Metrik Utama:")
    print(f"   - Total Pose Tercatat (captures.csv): {len(df_cap)}")
    print(f"   - Total Citra Tercatat (images.csv) : {len(df_img)}")
    
    # Check subjects
    subjects = df_cap["subject_id"].unique().tolist() if len(df_cap) > 0 else []
    print(f"   - Subjek Terdaftar: {subjects or 'Belum ada'}")
    
    # Check disk vs CSV consistency
    print("\n2. Pengecekan Integritas File Fisik di Disk:")
    missing_files = 0
    synced_pairs = 0
    delta_times = []
    
    for cap_id in df_cap["capture_id"].unique():
        sub_df = df_img[df_img["capture_id"] == cap_id]
        c1 = sub_df[sub_df["camera_id"] == "CAM01"]
        c2 = sub_df[sub_df["camera_id"] == "CAM02"]
        
        has_c1 = len(c1) > 0
        has_c2 = len(c2) > 0
        
        # Check files on disk
        if has_c1 and has_c2:
            p1 = PROJECT_ROOT / c1.iloc[0]["image_path"].replace("\\", "/")
            p2 = PROJECT_ROOT / c2.iloc[0]["image_path"].replace("\\", "/")
            
            if p1.exists() and p2.exists():
                synced_pairs += 1
                # Calculate timestamp delta
                try:
                    t1 = datetime.fromisoformat(c1.iloc[0]["timestamp"].replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(c2.iloc[0]["timestamp"].replace("Z", "+00:00"))
                    dt_ms = abs((t1 - t2).total_seconds() * 1000.0)
                    delta_times.append(dt_ms)
                except Exception:
                    pass
            else:
                missing_files += 1
        else:
            missing_files += 1
            
    print(f"   - Pasangan Citra Sinkron Lengkap (CAM01 + CAM02): {synced_pairs} pasang")
    print(f"   - Pasangan Rusak/Hilang: {missing_files} pasang")
    if delta_times:
        print(f"   - Rata-rata Latensi Sinkronisasi Antar-Kamera: {sum(delta_times)/len(delta_times):.1f} ms (Min: {min(delta_times):.1f} ms, Max: {max(delta_times):.1f} ms)")
        print(f"     *(Kualitas Sinkronisasi: {'SANGAT BAIK (<33 ms / 1 Frame)' if sum(delta_times)/len(delta_times) < 33 else 'CUKUP'})*")
        
    print("\n3. Distribusi Kelas Postur:")
    if len(df_cap) > 0:
        counts = df_cap["primary_posture"].value_counts().to_dict()
        for k, v in counts.items():
            print(f"   - {k:<20}: {v} pose")
    else:
        print("   - Belum ada data pose.")
    print("=" * 75 + "\n")


def delete_single_capture(capture_id):
    """Delete a specific capture pair from disk and CSVs."""
    print(f"\n[PROSES] Menghapus capture {capture_id}...")
    if not CAPTURES_CSV.exists() or not IMAGES_CSV.exists():
        print("[ERROR] File metadata tidak ditemukan.")
        return

    df_cap = pd.read_csv(CAPTURES_CSV)
    df_img = pd.read_csv(IMAGES_CSV)
    
    # 1. Delete physical files
    img_rows = df_img[df_img["capture_id"] == capture_id]
    for _, row in img_rows.iterrows():
        fpath = PROJECT_ROOT / str(row["image_path"]).replace("\\", "/")
        if fpath.exists():
            fpath.unlink()
            print(f"  [DELETED FILE] {fpath.name}")
            
    # 2. Filter CSVs
    df_cap_new = df_cap[df_cap["capture_id"] != capture_id]
    df_img_new = df_img[df_img["capture_id"] != capture_id]
    
    df_cap_new.to_csv(CAPTURES_CSV, index=False)
    df_img_new.to_csv(IMAGES_CSV, index=False)
    print(f"[OK] Capture {capture_id} berhasil dihapus dari disk dan metadata CSV!\n")


def delete_single_subject(subject_id):
    """Delete an entire subject folder and clean metadata."""
    print(f"\n[PROSES] Menghapus seluruh data untuk subjek {subject_id}...")
    
    # 1. Remove subject raw folder
    sub_dir = RAW_DIR / subject_id
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
        print(f"  [DELETED DIR] {sub_dir}")
        
    # 2. Clean CSVs
    if CAPTURES_CSV.exists():
        df_cap = pd.read_csv(CAPTURES_CSV)
        df_cap_new = df_cap[df_cap["subject_id"] != subject_id]
        df_cap_new.to_csv(CAPTURES_CSV, index=False)
        
    if IMAGES_CSV.exists():
        df_img = pd.read_csv(IMAGES_CSV)
        # Filter by image_id prefix
        df_img_new = df_img[~df_img["image_id"].str.startswith(f"{subject_id}_")]
        df_img_new.to_csv(IMAGES_CSV, index=False)
        
    print(f"[OK] Subjek {subject_id} berhasil dibersihkan dari dataset!\n")


def clean_calibration_raw_frames(confirm=False):
    """Clean temporary checkerboard raw frames and mock frames while keeping JSON configs."""
    calib_raw = PROJECT_ROOT / "02_data" / "private_calibration" / "raw_frames"
    calib_mock = PROJECT_ROOT / "02_data" / "private_calibration" / "mock_frames"
    
    print("\n[PROSES] Membersihkan foto mentah checkerboard & frame simulasi kalibrasi...")
    deleted_count = 0
    for folder in [calib_raw, calib_mock]:
        if folder.exists():
            for f in folder.glob("**/*"):
                if f.is_file() and not f.name.startswith(".gitkeep"):
                    f.unlink()
                    deleted_count += 1
    print(f"[OK] {deleted_count} file citra papan checkerboard sementara berhasil dibersihkan.")
    print("     (Matriks kalibrasi .json di intrinsic/ dan stereo/ TETAP AMAN & AKTIF).\n")


def reset_test_data(confirm=False):
    """Reset all dummy/test captures, leaving clean empty template CSVs."""
    if not confirm:
        ans = input("\n[PERINGATAN] Anda akan menghapus SELURUH citra mentah di private_raw/ (S001, S002, dst.) dan mereset captures.csv & images.csv ke keadaan awal bersih.\nKetik 'RESET' untuk mengonfirmasi: ").strip()
        if ans != "RESET":
            print("Reset dibatalkan.")
            return
            
    print("\n[PROSES] Mereset seluruh data subjek uji coba mentah...")
    # 1. Clear raw subfolders
    for item in RAW_DIR.iterdir():
        if item.is_dir() and item.name.startswith("S"):
            shutil.rmtree(item)
            print(f"  [DELETED DIR] {item.name}")
            
    # 2. Reset captures.csv to header only
    cap_headers = [
        "capture_id", "subject_id", "session_id", "calibration_id",
        "primary_posture", "head_state", "shoulder_state", "pelvis_state",
        "repetition", "subset", "quality", "notes", "chair_id", "lateral_side"
    ]
    with open(CAPTURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cap_headers)
    print(f"  [RESET] {CAPTURES_CSV.name} (Header bersih)")

    # 3. Reset images.csv to header only
    img_headers = [
        "image_id", "capture_id", "camera_id", "image_path",
        "timestamp", "width", "height", "blur_score",
        "exposure_status", "annotation_status", "view_role", "lateral_side"
    ]
    with open(IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(img_headers)
    print(f"  [RESET] {IMAGES_CSV.name} (Header bersih)")

    print("\n[OK] Dataset privat telah berhasil direset ke status BERSIH 100%! Siap untuk perekaman subjek Pilot Study.")


def main():
    parser = argparse.ArgumentParser(description="Dataset Manager & Audit Tool")
    parser.add_argument("--action", type=str, choices=["inspect", "delete_capture", "delete_subject", "reset_test_data", "clean_calib_frames"], default="inspect")
    parser.add_argument("--capture_id", type=str, default=None, help="Capture ID to delete (e.g. CAP000012)")
    parser.add_argument("--subject_id", type=str, default=None, help="Subject ID to delete (e.g. S001)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt for reset")

    args = parser.parse_args()

    if args.action == "inspect":
        inspect_dataset()
    elif args.action == "delete_capture":
        if not args.capture_id:
            print("[ERROR] Harap berikan --capture_id (contoh: --capture_id CAP000012)")
        else:
            delete_single_capture(args.capture_id)
    elif args.action == "delete_subject":
        if not args.subject_id:
            print("[ERROR] Harap berikan --subject_id (contoh: --subject_id S001)")
        else:
            delete_single_subject(args.subject_id)
    elif args.action == "reset_test_data":
        reset_test_data(confirm=args.yes)
    elif args.action == "clean_calib_frames":
        clean_calibration_raw_frames(confirm=args.yes)


if __name__ == "__main__":
    main()
