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
- [ ] **Fase 2: Ekstraksi Fitur 2D Multi-View (36 Fitur) & Audit**
  - [ ] Buat `04_scripts/preprocessing/extract_private_2d_features.py`
  - [ ] Eksekusi ekstraksi 2D $\rightarrow$ `private_features_2d.csv`
  - [ ] Audit fitur 2D $\rightarrow$ `feature_2d_audit.csv`
- [ ] **Fase 3: Ekstraksi Fitur Stereo 3D (25 Fitur) & Audit**
  - [ ] Buat `04_scripts/preprocessing/extract_private_3d_features.py`
  - [ ] Eksekusi ekstraksi 3D $\rightarrow$ `private_features_3d.csv`
  - [ ] Audit fitur 3D $\rightarrow$ `feature_3d_audit.csv`
- [ ] **Fase 4: Dataset Intersection & Partisi Grouped 5-Fold**
  - [ ] Buat `04_scripts/preprocessing/build_private_intersection.py`
  - [ ] Eksekusi pembentukan intersection manifest & feature tables
  - [ ] Buat `04_scripts/evaluation/create_private_subject_folds.py`
  - [ ] Eksekusi Stratified Group 5-Fold & validasi zero-leakage subject overlap
- [ ] **Fase 5: Benchmark & Evaluasi Ilmiah XGBoost 2D vs 3D**
  - [ ] Buat `04_scripts/training/train_private_xgboost_2d.py` (Grouped 5-Fold, OOF predictions)
  - [ ] Eksekusi training & evaluasi XGBoost 2D
  - [ ] Buat `04_scripts/training/train_private_xgboost_3d.py` (Grouped 5-Fold, OOF predictions)
  - [ ] Eksekusi training & evaluasi XGBoost 3D
  - [ ] Buat `04_scripts/evaluation/evaluate_private_2d_vs_3d.py`
  - [ ] Eksekusi komparasi formal 2D vs 3D & failure analysis
- [ ] **Fase 6: Serialisasi Model Deployment & Metadata**
  - [ ] Buat `04_scripts/training/fit_private_deployment_models.py`
  - [ ] Eksekusi fitting model deployment 2D & 3D (pkl, scaler, schemas, metadata)
- [ ] **Fase 7: Modul Inferensi Offline & Single Capture**
  - [ ] Buat `04_scripts/evaluation/test_private_single_capture.py` (Single capture & subject batch test)
  - [ ] Buat `04_scripts/inference/infer_private_pair_2d.py` (Inference 2D pasang citra baru)
  - [ ] Buat `04_scripts/inference/infer_private_pair_3d.py` (Inference 3D + Reject Gate)
- [ ] **Fase 8: Prototipe Inferensi Real-Time Dual-Kamera**
  - [ ] Buat `04_scripts/inference/infer_realtime_2d.py`
  - [ ] Buat `04_scripts/inference/infer_realtime_stereo_3d.py`
- [ ] **Fase 9: Master Runner & Laporan Akhir Naskah TA**
  - [ ] Buat `04_scripts/run_private_final_pipeline.py`
  - [ ] Update laporan akhir & tabel-tabel hasil evaluasi untuk skripsi
