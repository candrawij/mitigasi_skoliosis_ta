# TASK TRACKING IMPLEMENTASI PIPELINE FINAL 2D–3D XGBOOST
## Repositori: `mitigasi_skoliosis_ta`

Dokumen ini memantau progres pelaksanaan panduan implementasi teknis [PANDUAN_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md](file:///e:/TA/mitigasi_skoliosis_ta/08_documents/methodology/PANDUAN_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md).

---

## 📌 Checklist Progres Keseluruhan

- [x] **Fase 0: Freeze Status & Update Metadata Keputusan**
  - [x] Git commit & Tag checkpoint (`private-24subj-before-final-pipeline`)
  - [x] Update `03_metadata/taxonomy_kelas.md` (Definisi 6-class, eksklusi forward_head, reject gate)
  - [x] Update `03_metadata/dataset_decisions.md` (Keputusan final pembimbing)
- [x] **Fase 1: Modul Feature Common & Manifest 6-Class**
  - [x] Buat `04_scripts/preprocessing/private_feature_common.py` (Fungsi normalisasi, canonicalization, fitur 2D & 3D)
  - [x] Buat `04_scripts/preprocessing/build_private_6class_manifest.py`
  - [x] Eksekusi & Audit Manifest: Lolos assertion `N == 727` captures
- [x] **Fase 2: Ekstraksi Fitur 2D Multi-View (36 Fitur) & Audit**
  - [x] Buat `04_scripts/preprocessing/extract_private_2d_features.py`
  - [x] Eksekusi ekstraksi 2D $\rightarrow$ `private_features_2d.csv` (704 usable / 96.84%)
  - [x] Audit fitur 2D $\rightarrow$ `feature_2d_audit.csv` (0 inf/-inf, 0 duplicate, 0.57% NaN nose)
- [x] **Fase 3: Ekstraksi Fitur Stereo 3D (25 Fitur) & Audit**
  - [x] Buat `04_scripts/preprocessing/extract_private_3d_features.py`
  - [x] Eksekusi ekstraksi 3D $\rightarrow$ `private_features_3d.csv` (403 usable / 55.43%)
  - [x] Audit fitur 3D $\rightarrow$ `feature_3d_audit.csv` (0 inf/-inf, 0 duplicate, 7.69% NaN nose)
- [x] **Fase 4: Dataset Intersection & Partisi Grouped 5-Fold**
  - [x] Buat `04_scripts/preprocessing/build_private_intersection.py`
  - [x] Eksekusi pembentukan intersection manifest & feature tables (403 captures / 18 subjects)
  - [x] Buat `04_scripts/evaluation/create_private_subject_folds.py`
  - [x] Eksekusi Stratified Group 5-Fold & validasi zero-leakage subject overlap (Overlap = 0)
- [x] **Fase 5: Benchmark & Evaluasi Ilmiah XGBoost 2D vs 3D**
  - [x] Buat `04_scripts/training/train_private_xgboost_2d.py` (Grouped 5-Fold, OOF predictions)
  - [x] Eksekusi training & evaluasi XGBoost 2D $\rightarrow$ Acc 64.02%, Macro F1 0.6526 (5-Fold: 65.26% ± 8.14%)
  - [x] Buat `04_scripts/training/train_private_xgboost_3d.py` (Grouped 5-Fold, OOF predictions)
  - [x] Eksekusi training & evaluasi XGBoost 3D $\rightarrow$ Acc 59.31%, Macro F1 0.5947 (5-Fold: 60.43% ± 8.03%)
  - [x] Buat `04_scripts/evaluation/evaluate_private_2d_vs_3d.py`
  - [x] Eksekusi komparasi formal 2D vs 3D & failure analysis $\rightarrow$ `comparison_metrics.csv`, `per_class_comparison.csv`, `paired_capture_predictions.csv`, `confusion_matrix_comparison.png`, `interpretation_analysis.md`
- [x] **Fase 6: Serialisasi Model Deployment & Metadata**
  - [x] Buat `04_scripts/training/fit_private_deployment_models.py`
  - [x] Eksekusi fitting model deployment 2D & 3D $\rightarrow$ `06_models/keypoint_2d/private_final/` (704 usable samples, train acc 99.72%, CV F1 0.7310) & `06_models/keypoint_3d/private_final/` (403 usable samples, train acc 99.50%, CV F1 0.6462) lengkap dengan model.pkl, pipeline.pkl, scaler.pkl, feature_schema.json, class_map.json, model_metadata.json, coordinate_convention.json
- [x] **Fase 7: Modul Inferensi Offline & Single Capture**
  - [x] Buat `04_scripts/evaluation/test_private_single_capture.py` (Single capture & subject batch test terverifikasi pada S024 100% dan mode OOF)
  - [x] Buat `04_scripts/inference/private_inference_common.py` (Engine inferensi umum: YOLOv8-pose, TargetPersonSelector, Reject Gate, Stereo Triangulation, Normalisasi)
  - [x] Buat `04_scripts/inference/infer_private_pair_2d.py` (Inference 2D pasang citra baru terverifikasi pada raw Full HD image)
  - [x] Buat `04_scripts/inference/infer_private_pair_3d.py` (Inference 3D + Reject Gate terverifikasi untuk QC Valid dan QC Rejection)
- [ ] **Fase 8: Prototipe Inferensi Real-Time Dual-Kamera**
  - [ ] Buat `04_scripts/inference/infer_realtime_2d.py`
  - [ ] Buat `04_scripts/inference/infer_realtime_stereo_3d.py`
- [ ] **Fase 9: Master Runner & Laporan Akhir Naskah TA**
  - [ ] Buat `04_scripts/run_private_final_pipeline.py`
  - [ ] Update laporan akhir & tabel-tabel hasil evaluasi untuk skripsi
