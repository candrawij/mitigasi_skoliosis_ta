# Verifikasi Semantik Postureexercise

- Total citra: **1694**
- Total sumber klip terdeteksi dari nama file: **11**
- Sumber klip yang muncul pada lebih dari satu split: **0**
- Ukuran sumber klip terbesar: **316**
- kpt_shape: **[7, 3]**

## Output

- `image_source_map.csv`
- `source_clip_summary.csv`
- `source_clip_cross_split.csv`
- `indexed_keypoint_sheets/`

## Pemeriksaan manual wajib

1. Tentukan arti indeks keypoint 0–6.
2. Pastikan apakah tujuh titik hanya mencakup wajah dan bahu.
3. Konfirmasi arti kelas `nga_*` dan `nghieng_*`.
4. Periksa apakah satu sumber klip tersebar pada train/valid/test.
5. Jangan memakai horizontal flip sebelum pasangan keypoint dan pertukaran kelas kiri–kanan ditetapkan.
