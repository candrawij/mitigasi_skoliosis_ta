# Sistem Deteksi dan Klasifikasi Postur Duduk Berisiko Skoliosis

Penelitian Tugas Akhir: **"Deteksi dan Klasifikasi Postur Duduk Berisiko Skoliosis Berbasis Multi-View 2D Keypoint dan Stereo 3D Reconstruction Menggunakan XGBoost"**  
Repositori: `mitigasi_skoliosis_ta`  
Status: **SELESAI & TERKUNCI (FROZEN RESULTS - Tag: `v1.0-freeze-final-four-model`)**

---

## 📌 Ringkasan Eksekutif & Temuan Ilmiah Utama

Penelitian ini membandingkan secara objektif (*fair comparison*) efektivitas representasi geometri **Multi-View 2D Keypoint** versus **Stereo 3D Reconstruction** yang dipadukan dengan algoritma **XGBoost** untuk mendeteksi 6 kelas postur duduk berisiko skoliosis dan gangguan muskuloskeletal pada dataset privat 24 subjek riil menggunakan protokol evaluasi ketat **5-Fold Subject-Stratified Cross-Validation**.

### Tabel Hasil Resmi 4 Model Utama (Single Source of Truth)
*Sumber acuan resmi:* [`07_results/experiments/private_final/FINAL_FOUR_MODEL/final_four_model_comparison.csv`](file:///07_results/experiments/private_final/FINAL_FOUR_MODEL/final_four_model_comparison.csv)

| Model | Representasi & Fitur | Pooled Macro F1 | Fold Std | Akurasi | Rata-rata Latensi | Rekomendasi Deployment |
|---|:---:|:---:|:---:|:---:|:---:|---|
| ⭐ **2D Best Practical** | Multi-View 2D + Telinga (**42 fitur**) | **0.7039** | **0.0701** | **70.61%** | **21.7 ms (~46 FPS)** | **Model Utama Deployment Real-Time** |
| **2D Core Baseline** | Multi-View 2D Inti (36 fitur) | 0.6863 | 0.0846 | 69.11% | 20.4 ms (~49 FPS) | Baseline komparasi 2D |
| **3D Current / Camera** | Stereo 3D Kamera (25 fitur) | 0.6182 | 0.2519 | 63.66% | 48.2 ms (~20 FPS) | Model 3D observasi kamera |
| **3D Canonical / Body** | Stereo 3D Koordinat Tubuh (22 fitur) | 0.3259 | 0.1241 | 36.31% | 51.5 ms (~19 FPS) | Bukti ilmiah limitasi transformasi kanonikal |

### 🔬 Tiga Kontribusi Ilmiah Kunci:
1. **Keunggulan Praktis Multi-View 2D:** Model 2D Best mengungguli 3D Camera Frame sebesar **+8.57% Macro F1** dengan stabilitas antarsubjek jauh lebih konsisten (standar deviasi fold 0.0701 vs 0.2519) dan latensi 2.2x lebih cepat.
2. **Dilema Ruang Kanonikal 3D:** Transformasi rotasi kanonikal ke kerangka acuan tubuh (*body frame*) secara matematis mengeliminasi kemiringan global terhadap gravitasi, menyebabkan 3 fitur torso degenerate (varians mendekati 0), yang meruntuhkan Macro F1 menjadi 0.3259.
3. **Integritas Metodologi Zero-Leakage:** Seluruh metrik dievaluasi menggunakan partisi subjek terisolasi (*subject-stratified*), membuktikan bahwa performa mencerminkan kemampuan generalisasi riil ke individu baru tanpa *data leakage*.

---

## 🗂️ Struktur Direktori Proyek

Proyek ini terstruktur ke dalam 8 modul terorganisir:

```text
mitigasi_skoliosis_ta/
├── 01_literature/           <-- Koleksi 16 artikel referensi ilmiah terkelompok & berindeks
│   ├── articles/            (Reviews, 2D Pose, 3D Sensors, ML/XGBoost, IoT Systems)
│   └── README.md
├── 02_data/                 <-- Repositori data publik & dataset privat 24 subjek
│   ├── private_raw/         (Citra mentah dual-kamera 24 subjek S001-S024)
│   ├── private_calibration/ (Citra kalibrasi papan catur & matriks stereo)
│   ├── private_annotations/ (Anotasi keypoint 2D, 3D, dan person selector)
│   └── private_processed/   (Vektor fitur 2D/3D & manifest irisan valid)
├── 03_metadata/             <-- Konfigurasi perangkat, konvensi 3D, taksonomi & split CV
│   ├── private_final_split/ (5-fold stratified & intersection manifests)
│   ├── private_templates/   (Tabel relasional partisipan, capture, kalibrasi)
│   └── README.md
├── 04_scripts/              <-- Pipeline kode modular dari kalibrasi hingga inferensi
│   ├── analysis/            (final_four_model_comparison.py - generator hasil resmi)
│   ├── calibration/         (Kalibrasi stereo papan catur)
│   ├── capture/             (Perekaman sinkron dual-kamera)
│   ├── processing/          (YOLOv8-Pose & Target Person Selector)
│   ├── preprocessing/       (Ekstraksi fitur 2D, 3D kamera, 3D kanonikal)
│   ├── training/            (Pelatihan XGBoost & export model)
│   ├── evaluation/          (Cross-validation & perbandingan performa)
│   ├── inference/           (Sistem inferensi real-time 2D & Stereo 3D)
│   └── README.md
├── 05_notebooks/            <-- Dokumentasi arsitektur modul Python (README.md)
├── 06_models/               <-- Model serialisasi deployment (.pkl, skema fitur, class map)
│   ├── keypoint_2d/         (Model XGBoost 2D final & augmented 42 fitur)
│   ├── keypoint_3d/         (Model XGBoost 3D final 25 fitur)
│   └── README.md
├── 07_results/              <-- Master indeks seluruh hasil eksperimen, audit & visualisasi
│   ├── experiments/         (Folder FINAL_FOUR_MODEL/ sebagai Single Source of Truth)
│   ├── private_audit/       (Lembar kontak verifikasi 24 subjek & log seleksi)
│   ├── dataset_audit/       (Audit anomali 4 dataset publik awal)
│   ├── live_camera_tests/   (Log pengujian kamera langsung real-time)
│   ├── visualizations/      (Matriks kebingungan & plot komparasi)
│   └── README.md
├── 08_documents/            <-- Arsip dokumentasi akademik, naskah publikasi & panduan
│   ├── 01_laporan_evaluasi_dan_progres/ (Laporan komprehensif, evaluasi stereo 3D)
│   ├── 02_laporan_qc_dataset/           (Audit kualitas dataset 24 subjek riil)
│   ├── 03_panduan_dan_sop/              (SOP protokol akuisisi & pipeline deployment)
│   ├── 04_perencanaan_dan_task/         (Roadmap riset & pelacak tugas)
│   ├── 05_naskah_dan_publikasi/         (Outline 6 bab skripsi & artikel ilmiah)
│   ├── figures/                         (Peta aset visual untuk naskah skripsi)
│   └── README.md
├── run_live_camera.bat      <-- Skrip batch sekali klik untuk inferensi 2D real-time
├── run_live_camera_dual.bat <-- Skrip batch sekali klik untuk inferensi Stereo 3D
├── requirements.txt         <-- Dependensi pustaka Python
└── yolov8n-pose.pt          <-- Bobot pretrained YOLOv8n-pose Ultralytics
```

---

## 🚀 Panduan Menjalankan Sistem (Quickstart)

### 1. Instalasi Dependensi
Pastikan Python 3.10+ telah terpasang:
```bash
pip install -r requirements.txt
```

### 2. Inferensi Kamera Langsung 2D (Model Rekomendasi Deployment)
Jalankan file batch:
```cmd
run_live_camera.bat
```
Atau via terminal Python:
```bash
python 04_scripts/inference/infer_realtime_2d.py --camera-id 0 --model-dir 06_models/keypoint_2d/private_final
```

### 3. Inferensi Kamera Stereo 3D
Jalankan file batch:
```cmd
run_live_camera_dual.bat
```
Atau via terminal Python:
```bash
python 04_scripts/inference/infer_realtime_stereo_3d.py --cam-left 0 --cam-right 1 --calib-file 02_data/private_calibration/stereo_calibration.json
```

### 4. Mereproduksi Evaluasi 4 Model Utama (Single Source of Truth)
Untuk memverifikasi dan mencetak ulang seluruh metrik fold-by-fold resmi:
```bash
python 04_scripts/analysis/final_four_model_comparison.py
```

---

## 📖 Navigasi Naskah & Dokumen Akademik

- **Perjalanan Lengkap Riset (Ide s.d. Selesai):** [`08_documents/01_laporan_evaluasi_dan_progres/LAPORAN_KOMPREHENSIF_AWAL_SAMPAI_AKHIR.md`](08_documents/01_laporan_evaluasi_dan_progres/LAPORAN_KOMPREHENSIF_AWAL_SAMPAI_AKHIR.md)
- **Outline Lengkap Skripsi (6 Bab) & Naskah Publikasi:** [`08_documents/05_naskah_dan_publikasi/OUTLINE_LENGKAP_SKRIPSI_ARTIKEL.md`](08_documents/05_naskah_dan_publikasi/OUTLINE_LENGKAP_SKRIPSI_ARTIKEL.md)
- **Outline Singkat Pipeline & Analisis Kelebihan/Kekurangan:** [`08_documents/05_naskah_dan_publikasi/OUTLINE_SINGKAT.md`](08_documents/05_naskah_dan_publikasi/OUTLINE_SINGKAT.md)
- **Laporan Evaluasi Stereo 3D & Uji Latensi:** [`08_documents/01_laporan_evaluasi_dan_progres/laporan_progress_stereo_3d_dan_evaluasi_final.md`](08_documents/01_laporan_evaluasi_dan_progres/laporan_progress_stereo_3d_dan_evaluasi_final.md)
- **SOP Protokol Akuisisi Data:** [`08_documents/03_panduan_dan_sop/Panduan_Pengumpulan_Dataset_Privat_Postur_Duduk.md`](08_documents/03_panduan_dan_sop/Panduan_Pengumpulan_Dataset_Privat_Postur_Duduk.md)