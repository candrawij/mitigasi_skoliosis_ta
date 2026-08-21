# Hasil Clustering dan Metadata Kurasi

- Total gambar: **4813**
- Threshold clustering: **Hamming distance <= 2**
- Edge/pasangan yang dipakai: **23256**
- Jumlah cluster (ukuran >= 2): **371**
- Gambar yang masuk cluster: **4216**
- Gambar singleton: **597**
- Cluster mixed-label: **56**
- Cluster cross-split: **92**
- Contact sheet prioritas dibuat: **97**

## File utama

- `curation_master.csv`: lembar kerja audit manual seluruh gambar.
- `duplicate_clusters.csv`: ringkasan semua cluster.
- `priority_review.csv`: cluster yang harus diperiksa lebih dahulu.
- `representatives.csv`: kandidat gambar perwakilan tiap cluster.
- `contact_sheets_priority/`: visual cluster prioritas.

## Urutan audit manual

1. `P1_mixed_label_cross_split`
2. `P2_cross_split`
3. `P3_mixed_label`
4. `P4_same_label_same_split`

## Keputusan manual

Isi kolom pada `curation_master.csv`:

- `decision`: `KEEP`, `RELABEL`, atau `EXCLUDE`
- `final_label`: diisi jika `KEEP` atau `RELABEL`
- `subject_group`: contoh `S01`
- `session_group`: contoh `S01_SE01`
- `camera_view`: `frontal`, `diagonal`, atau `lateral`
- `torso_visible`, `shoulders_visible`, `hips_visible`: `yes`/`no`
- `label_quality`: `clear`, `ambiguous`, atau `wrong`
- `notes`: alasan keputusan

Skrip tidak mengubah dataset mentah.
