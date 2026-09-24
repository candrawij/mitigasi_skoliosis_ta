# Master Indeks Skrip Pipeline Penelitian (04_scripts)

Direktori ini memuat seluruh kode program modular untuk akuisisi data, kalibrasi kamera stereo, ekstraksi fitur pose, pelatihan model *machine learning*, evaluasi komparatif, dan sistem inferensi *real-time*.

---

## 1. Alur Eksekusi Pipeline (Workflow Pipeline)

```text
[1. Kalibrasi & Akuisisi]
   04_scripts/calibration/calibrate_stereo.py
   04_scripts/capture/capture_multicam.py
          │
          ▼
[2. Deteksi Pose & Target Selector]
   04_scripts/processing/target_person_selector.py
          │
          ▼
[3. Ekstraksi Fitur Spasial]
   04_scripts/preprocessing/extract_private_2d_features.py (2D Core 36f / Best 42f)
   04_scripts/preprocessing/extract_private_3d_features.py (3D Camera Frame 25f)
   04_scripts/preprocessing/canonicalize_private_3d_pose.py (3D Canonical 22f)
          │
          ▼
[4. Pelatihan Model & Subject-Stratified CV]
   04_scripts/training/train_private_xgboost_2d_augmented.py (2D Best)
   04_scripts/training/train_private_xgboost_3d.py (3D Current)
   04_scripts/training/train_private_xgboost_3d_canonical.py (3D Canonical)
   04_scripts/training/fit_private_deployment_models_augmented.py (Export .pkl)
          │
          ▼
[5. Evaluasi Ilmiah Resmi (Single Source of Truth)]
   04_scripts/analysis/final_four_model_comparison.py
          │
          ▼
[6. Deployment & Inferensi Real-Time]
   04_scripts/inference/infer_realtime_2d.py (atau jalankan run_live_camera.bat)
   04_scripts/inference/infer_realtime_stereo_3d.py (atau jalankan run_live_camera_dual.bat)
```

---

## 2. Struktur Subdirektori Skrip

| Subdirektori | Deskripsi & Peran Utama | Skrip Kunci |
|---|---|---|
| **`analysis/`** | Analisis penutup & evaluasi kuantitatif terkunci (*frozen results*) | [`final_four_model_comparison.py`](file:///04_scripts/analysis/final_four_model_comparison.py) (Menghasilkan tabel final 4 model) |
| **`calibration/`** | Kalibrasi kamera intrinsik & ekstrinsik stereo (papan catur) | `calibrate_stereo.py`, `validate_calibration.py` |
| **`capture/`** | Sinkronisasi penangkapan citra dari kamera stereo dual-USB | `capture_multicam.py`, `dataset_manager.py` |
| **`processing/`** | Deteksi keypoint YOLOv8-Pose dan algoritma penentu orang target | `target_person_selector.py` (Menghindari orang lain di latar belakang) |
| **`preprocessing/`** | Rekayasa fitur sudut/panjang (2D, 3D kamera, dan 3D canonical) | `extract_private_2d_features.py`, `canonicalize_private_3d_pose.py` |
| **`training/`** | Pelatihan model XGBoost, tuning hyperparameter, & export model | `train_private_xgboost_2d_augmented.py`, `fit_private_deployment_models_augmented.py` |
| **`evaluation/`** | Evaluasi 5-fold CV subject-stratified dan evaluasi cross-dataset | `evaluate_private_2d_vs_3d.py`, `compare_2d_vs_3d.py` |
| **`inference/`** | Script inferensi langsung kamera (kamera tunggal & kamera stereo) | `infer_realtime_2d.py`, `infer_realtime_stereo_3d.py` |
| **`audit/`** | Validasi integritas anotasi, korespondensi stereo, dan reprojection | `stereo_person_correspondence_audit.py`, `validate_3d_reprojection_final.py` |
| **`curation/`** | Pembersihan anomali dan standardisasi dataset publik awal | `build_curation_metadata.py`, `prepare_postureexercise.py` |
| **`visualization/`** | Pembuatan plot matriks kebingungan dan grafik perbandingan | `generate_experiment_plots.py` |
