# Master Indeks Metadata Penelitian (03_metadata)

Direktori ini memuat seluruh skema fitur, konvensi koordinat 3D, taksonomi kelas postur, catatan keputusan kurasi dataset, serta partisi resmi *Cross-Validation* (CV).

---

## 1. File Konfigurasi & Konvensi Inti

- **`camera_config.json`**: Parameter konfigurasi perangkat keras stereo dual-kamera (resolusi, FPS, indeks perangkat).
- **`dataset_decisions.md`**: Log keputusan strategis mengenai filtrasi, pembersihan anomali, dan pemisahan subjek.
- **`feature_schema_3d.json`**: Skema resmi fitur geometri 3D (25 fitur untuk *Camera Frame* dan 22 fitur untuk *Canonical Body Frame*).
- **`private_3d_coordinate_convention.md`**: Definisi sumbu koordinat spasial 3D kamera stereo (X: kanan, Y: bawah, Z: kedalaman depan) dan matriks rotasi canonicalization.
- **`private_dataset_conventions.md`**: Konvensi penamaan subjek (`S001` s/d `S024`), sesi (`SE01`), dan penamaan capture stereo (`CAM01`, `CAM02`).
- **`taxonomy_kelas.md`**: Definisi dan justifikasi 6 kelas postur duduk yang dievaluasi.
- **`postureexercise_keypoint_map.md`**: Pemetaan 7-keypoint dataset publik awal.

---

## 2. Subdirektori Metadata

- **`private_final_split/`**:  
  Partisi resmi dataset privat untuk evaluasi ilmiah:
  - `private_stratified_group_5fold.csv`: Partisi 5-Fold Stratified Group K-Fold berdasarkan subjek (tidak ada kebocoran identitas subjek antarfolds).
  - `private_2d3d_intersection_5fold.csv`: Partisi irisan sampel valid (*valid stereo pairs*) yang menjamin *fair comparison* antara model 2D dan model 3D pada subset data yang 100% identik.
- **`private_templates/`**:  
  Tabel relasional master dataset privat:
  - `participants.csv`: Profil demografis 24 partisipan.
  - `captures.csv`: Master data seluruh pengambilan sampel dan label postur.
  - `images.csv`: Daftar file citra per kamera.
  - `calibration_map.csv`: Pemetaan kalibrasi kamera per sesi.
  - `chairs.csv`: Spesifikasi kursi yang digunakan.
  - `qc_audit_log.csv`: Log kontrol kualitas data.
- **`original_audit/`**:  
  Log audit awal terhadap 4 dataset publik (identifikasi duplikasi, kebocoran split, dan inkonsistensi label).
- **`curation/`**:  
  Metadata kurasi perbaikan dataset publik.
- **`final_split/`**:  
  Split dataset publik awal untuk eksperimen *baseline* (Postureexercise, IKORN 4-KP, Project Design).
