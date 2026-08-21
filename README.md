# Penelitian Deteksi Postur Duduk

## Fokus
Deteksi dan klasifikasi postur duduk berbasis citra untuk
mendukung mitigasi kebiasaan postur tidak ergonomis.

## Dataset kandidat utama
Project Design 20242025.

## Kelas
- upright
- leaning_forward
- leaning_backward
- leaning_left
- leaning_right

## Status
- SOTA sementara telah ditentukan.
- Audit teknis awal dataset telah selesai.
- Dataset belum layak digunakan dengan split bawaan.
- Tahap berjalan: kurasi dataset dan validasi keypoint.

## Temuan audit awal
- 4.813 citra valid
- 0 citra rusak
- 65.237 pasangan near duplicate
- 10.755 potensi leakage lintas split
- split bawaan tidak memiliki seluruh kelas pada setiap subset