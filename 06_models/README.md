# Master Indeks Model Terlatih (06_models)

Direktori ini memuat seluruh berkas model machine learning terlatih (*trained weights*), pipeline preprocessing ter-fit (*scaler* & *imputer*), skema fitur, dan metadata deployment.

---

## 1. Model Deployment Utama

### A. Model 2D Utama (Rekomendasi Terbaik)
📂 [`06_models/keypoint_2d/private_final/`](file:///06_models/keypoint_2d/private_final/)  
atau model varian augmented: [`06_models/keypoint_2d/private_augmented/`](file:///06_models/keypoint_2d/private_augmented/)
- **`model.pkl` / `xgboost_2d_deployment.pkl`**: Model XGBoost terlatih untuk estimasi pose 2D.
- **`scaler.pkl`**: Standarisasi skalar fitur yang fit pada data latih.
- **`pipeline.pkl`**: Pipeline scikit-learn lengkap (scaler + classifier).
- **`feature_schema.json`**: Daftar dan urutan 42 fitur 2D (termasuk fitur telinga dan asimetri bahu).
- **`class_map.json`**: Pemetaan indeks numerik ke label kelas postur (`ideal`, `kyphosis`, `slouching`, `scoliosis_left`, `scoliosis_right`, `lordosis`).
- **`model_metadata.json`**: Catatan hyperparameter, tanggal pelatihan, dan versi paket.

### B. Model 3D Stereo (Pembanding Kedalaman)
📂 [`06_models/keypoint_3d/private_final/`](file:///06_models/keypoint_3d/private_final/)
- **`model.pkl` / `xgboost_3d_deployment.pkl`**: Model XGBoost 3D berbasis koordinat kamera (25 fitur).
- **`coordinate_convention.json`**: Konvensi sumbu ruang 3D sistem kamera stereo.
- **`feature_schema.json`**: Skema 25 fitur spasial 3D.
- **`pipeline.pkl` & `scaler.pkl`**: Pipeline transformasi data 3D.

---

## 2. Model Baseline RGB (Citra Langsung)
📂 [`06_models/rgb_baseline/`](file:///06_models/rgb_baseline/)  
Model baseline citra RGB (EfficientNet-B0) yang dievaluasi pada tahap eksplorasi awal disimpan dan didokumentasikan di bawah direktori eksperimen:
- [`07_results/experiments/EXP-PD-CNN/best_model.pt`](file:///07_results/experiments/EXP-PD-CNN/best_model.pt)
- [`07_results/experiments/EXP-SPD-CNN/best_model.pt`](file:///07_results/experiments/EXP-SPD-CNN/best_model.pt)
Lihat [`06_models/rgb_baseline/README.md`](rgb_baseline/README.md) untuk penjelasan lengkap.
