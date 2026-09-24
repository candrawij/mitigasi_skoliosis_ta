# Master Indeks Hasil Eksperimen & Evaluasi (07_results)

Direktori ini menyimpan seluruh artefak kuantitatif, log eksperimen, metrik performa, matriks kebingungan (*confusion matrices*), serta hasil audit dataset proyek.

---

## 1. Single Source of Truth (Hasil Resmi Terkunci)

> **Catatan Akademik Penting:** Seluruh sitasi angka performa akhir pada naskah skripsi, jurnal ilmiah, dan presentasi wajib merujuk secara eksklusif ke folder:  
> [`07_results/experiments/private_final/FINAL_FOUR_MODEL/`](file:///07_results/experiments/private_final/FINAL_FOUR_MODEL/)  
> File acuan utama: [`final_four_model_comparison.csv`](file:///07_results/experiments/private_final/FINAL_FOUR_MODEL/final_four_model_comparison.csv).

### Tabel Perbandingan Resmi 4 Model Utama (5-Fold Subject-Stratified CV):

| Model | Representasi & Jumlah Fitur | Macro F1 (Pooled) | Fold Std | Akurasi | Status & Rekomendasi |
|---|:---:|:---:|:---:|:---:|---|
| **2D Best Practical** | Multi-View 2D + Telinga (42 fitur) | **0.7039** | **0.0701** | **70.61%** | ⭐ **Model Utama Rekomendasi Deployment Real-Time** |
| **2D Core Baseline** | Multi-View 2D Inti (36 fitur) | 0.6863 | 0.0846 | 69.11% | Baseline komparasi fitur 2D |
| **3D Current / Camera Frame** | Stereo 3D Koordinat Kamera (25 fitur) | 0.6182 | 0.2519 | 63.66% | Model 3D observasi kamera (sensitif sudut) |
| **3D Canonical / Body Frame** | Stereo 3D Koordinat Tubuh (22 fitur) | 0.3259 | 0.1241 | 36.31% | Bukti ilmiah degenerasi fitur sudut |

---

## 2. Struktur Direktori `07_results/`

```text
07_results/
├── experiments/                     <-- Seluruh artefak eksperimen model
│   ├── private_final/               <-- Eksperimen final pada dataset privat
│   │   ├── FINAL_FOUR_MODEL/        <-- ⭐ [SINGLE SOURCE OF TRUTH] 4 Model Utama Terkunci
│   │   ├── 2d/                      <-- Artefak fold CV 2D
│   │   ├── 3d/                      <-- Artefak fold CV 3D
│   │   ├── 3d_canonical/            <-- Artefak fold CV 3D Canonical
│   │   ├── 3d_current/              <-- Uji live stereo pair & latency benchmark
│   │   ├── comparison/              <-- Metrik perbandingan awal
│   │   └── comparison_final/        <-- Ringkasan JSON perbandingan model
│   ├── private_augmented/           <-- Eksperimen augmentasi slouching
│   ├── private_ear_v2/              <-- Eksperimen rekayasa fitur telinga
│   ├── EXP-05-CROSS-DATASET/        <-- Eksperimen generalisasi lintas dataset
│   ├── EXP-IK-KP/                   <-- Eksperimen dataset IKORN 4-keypoint
│   ├── EXP-PE-KP/                   <-- Eksperimen dataset Postureexercise (XGBoost vs MLP)
│   ├── EXP-PD-CNN/                  <-- Eksperimen CNN EfficientNet pada Project Design
│   ├── EXP-SPD-CNN/                 <-- Eksperimen CNN pada Sitting Posture Detection
│   ├── EXP-YOLO-POSE-PROJECT_DESIGN/<-- Evaluasi YOLO-Pose pada Project Design
│   └── EXP-YOLO-POSE-SITTING_POSTURE_DETECTION/ <-- Evaluasi YOLO-Pose pada SPD
├── private_audit/                   <-- Hasil audit & kontrol kualitas dataset privat (24 subjek)
│   ├── contact_sheets/              <-- Lembar kontak visual verifikasi 24 subjek & 7 kelas postur
│   └── selection_logs/              <-- Log disambiguasi pemilihan orang target (CAM01 & CAM02)
├── dataset_audit/                   <-- Audit & verifikasi semantik 4 dataset publik
│   ├── postureexercise_initial/     <-- Audit kualitas dataset Postureexercise
│   ├── postureexercise_semantics/   <-- Verifikasi anomali semantik label kiri/kanan
│   ├── project_design_initial/      <-- Audit duplikasi & kebocoran dataset Project Design
│   ├── sitting_posture_4kp_initial/ <-- Audit dataset IKORN 4-KP
│   └── sitting_posture_detection_initial/ <-- Audit bounding box dataset SPD
├── live_camera_tests/               <-- Log pengujian inferensi kamera langsung (Live Inference)
├── visualizations/                  <-- Gambar matriks kebingungan (confusion matrices) & grafik
└── README.md                        <-- (Dokumen ini) Master Indeks
```

---

## 3. Direktori Eksperimen Penting Lainnya

- **`private_audit/contact_sheets/`**: Berisi lembar verifikasi visual dari 24 subjek penelitian yang menjadi bukti dokumentasi integritas dataset pada bab metodologi.
- **`live_camera_tests/`**: Berisi riwayat log pengujian kamera langsung (dual-camera test) yang membuktikan pipeline dapat berjalan stabil secara real-time dengan latensi rata-rata ~21.7 ms untuk 2D dan ~48.2 ms untuk Stereo 3D.
