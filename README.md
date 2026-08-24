# Penelitian Deteksi Postur Duduk

## Fokus

Deteksi dan klasifikasi postur duduk berbasis citra untuk
mendukung mitigasi kebiasaan postur tidak ergonomis.

## Pipeline

Dokumen pipeline lengkap: [`08_documents/methodology/pipeline_penelitian_deteksi_postur.md`](08_documents/methodology/pipeline_penelitian_deteksi_postur.md)

## Dataset

| Dataset | Representasi | Pipeline | Peran |
|---|---|---|---|
| Project Design 20242025 | Image | CNN / YOLO Cls | RGB baseline |
| Postureexercise | 7 Keypoint (YOLO-Pose) | MLP / XGBoost / KAN | Pose baseline |
| IKORN / 4-Keypoint | 4 Keypoint (COCO-Pose) | MLP | Minimal-keypoint baseline |
| Sitting Posture Detection | Bounding box (COCO-Det) | YOLO Detection | Detection baseline |
| Dataset privat (TBD) | Multi-view full-body | YOLO-Pose → 2D → 3D | Proposed method |

## Eksperimen

| ID | Dataset | Metode | Status |
|---|---|---|---|
| EXP-PD-01 | Project Design | CNN (EfficientNet-B0) | Planned |
| EXP-PD-02 | Project Design | YOLO Classification | Opsional |
| EXP-PE-01 | Postureexercise | MLP | Planned |
| EXP-PE-02 | Postureexercise | XGBoost | Planned |
| EXP-PE-03 | Postureexercise | KAN | Planned |
| EXP-IK-01 | IKORN 4-KP | MLP | Planned |
| EXP-SPD-01 | Sitting Posture Det | YOLO Detection | Blocked |
| EXP-PR-01–04 | Private | Various | Deferred |

## Struktur Folder

```
01_literature/          ← Referensi, paper, SOTA
02_data/
  ├── raw/              ← Dataset mentah (4 dataset publik)
  ├── interim/          ← Hasil curation / cleaning
  └── processed/        ← Data final siap eksperimen
03_metadata/
  ├── original_audit/   ← Audit mentah
  ├── curation/         ← Metadata curation per dataset
  ├── final_split/      ← Split definitif
  └── subject_groups/   ← Grouping subject
04_scripts/
  ├── audit/            ← Script audit dataset
  ├── curation/         ← Script curation / cleaning
  ├── preprocessing/    ← Script preprocessing (resize, normalize, feature)
  ├── splitting/        ← Script splitting dataset
  ├── pose_estimation/  ← Script pose estimation
  ├── training/         ← Script training model
  └── evaluation/       ← Script evaluasi & perbandingan
05_notebooks/           ← Jupyter notebooks eksplorasi
06_models/
  ├── rgb_baseline/     ← Model image-based
  ├── keypoint_2d/      ← Model keypoint 2D
  └── keypoint_3d/      ← Model keypoint 3D
07_results/
  ├── dataset_audit/    ← Hasil audit per dataset
  ├── pose_feasibility/ ← Feasibility study pose
  ├── rgb_baseline/     ← Hasil eksperimen image baseline
  ├── keypoint_2d/      ← Hasil eksperimen 2D keypoint
  ├── keypoint_3d/      ← Hasil eksperimen 3D keypoint
  ├── experiments/      ← Hasil per eksperimen (EXP-*)
  └── comparison/       ← Perbandingan & ablation study
08_documents/
  ├── methodology/      ← Pipeline & metodologi
  ├── proposal/         ← Proposal penelitian
  └── figures/          ← Gambar / diagram
```

## Status

- ✅ Audit teknis awal 4 dataset publik selesai
- ✅ Curation clustering Project Design dimulai (97 contact sheets)
- ✅ Verifikasi semantik Postureexercise dimulai
- 🔲 Curation manual dataset belum selesai
- 🔲 Preprocessing & training belum dimulai
- 🔒 Sitting Posture Detection menunggu audit contact sheet
- ⏳ Dataset privat belum tersedia