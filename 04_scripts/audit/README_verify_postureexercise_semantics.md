# Verifikasi Semantik Postureexercise

## Tujuan

Audit teknis sudah selesai. Tahap ini memeriksa:

- urutan tujuh keypoint;
- apakah keypoint hanya mencakup wajah dan bahu;
- sumber klip berdasarkan nama file;
- potensi satu klip tersebar pada split berbeda.

## Simpan skrip

```text
TA\04_scripts\audit\verify_postureexercise_semantics.py
```

## Jalankan

Dari root `TA`:

```powershell
python ".\04_scripts\audit\verify_postureexercise_semantics.py" `
  ".\02_data\raw\Sitting_posture.v17i.yolov8" `
  --output-dir ".\07_results\dataset_audit\postureexercise_semantics"
```

Versi satu baris:

```powershell
python ".\04_scripts\audit\verify_postureexercise_semantics.py" ".\02_data\raw\Sitting_posture.v17i.yolov8" --output-dir ".\07_results\dataset_audit\postureexercise_semantics"
```

## Output yang perlu dikirim

1. `semantic_verification_summary.md`
2. `source_clip_summary.csv`
3. `source_clip_cross_split.csv`
4. Semua gambar di `indexed_keypoint_sheets/`

## Peringatan augmentasi

Jangan mengaktifkan horizontal flip sebelum:

- urutan keypoint kiri–kanan diketahui;
- `flip_idx` diperbaiki;
- kelas arah dipertukarkan:
  - `nga_phai ↔ nga_trai`
  - `nghieng_phai ↔ nghieng_trai`

Ultralytics tidak otomatis mengganti class ID arah saat citra dibalik. Pilihan aman awal adalah `fliplr=0.0`.
