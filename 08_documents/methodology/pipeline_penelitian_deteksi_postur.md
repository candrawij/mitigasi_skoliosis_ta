# Rencana Pipeline Penelitian Deteksi Postur Duduk untuk Mitigasi Risiko Skoliosis

**Status:** Pipeline penelitian versi awal / baseline design

## 1. Ringkasan

Penelitian diarahkan pada deteksi/klasifikasi postur tubuh, khususnya postur saat duduk, sebagai pendekatan computer vision untuk mitigasi risiko postur yang berpotensi tidak baik.

Fokus penelitian bukan diagnosis medis skoliosis. Sistem diarahkan untuk mengenali pola postur dari citra dan representasi pose, lalu membandingkan pendekatan:

1. image-based;
2. bounding-box/object detection;
3. 2D keypoint-based;
4. multi-view 3D keypoint-based.

YOLO digunakan sesuai jenis anotasi:
- **YOLO Detection** untuk dataset dengan bounding box;
- **YOLO-Pose** untuk memperoleh keypoint tubuh pada pipeline utama.

---

## 2. Pembagian fungsi dataset

| Dataset | Representasi | Pipeline | Peran |
|---|---|---|---|
| Project Design 20242025 | Image | Image classification | RGB baseline |
| Postureexercise | Keypoint | 2D keypoint → classifier | Pose baseline |
| IKORN / 4-keypoint | 4 keypoint | 4 keypoint → MLP | Minimal-keypoint baseline |
| Sitting Posture Detection Initial | Bounding box | YOLO Detection | YOLO baseline |
| Dataset privat | Full-body + multi-view | YOLO-Pose → 2D → 3D | Proposed/main experiment |

**Catatan:** `Sitting Posture Detection Initial` masih menunggu pemeriksaan contact sheet sebelum kelayakannya sebagai benchmark dikunci.

---

# 3. Arsitektur umum

```text
                    DATASET PUBLIC
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
        ▼                ▼                 ▼
   Image-based      Keypoint-based    Object Detection
    baseline          baseline           baseline
        │                │                 │
        └────────────────┼─────────────────┘
                         │
                         ▼
                 Perbandingan baseline
                         │
                         ▼
                 DATASET PRIVAT
                         │
                         ▼
                    YOLO-Pose
                         │
                         ▼
                    2D Keypoint
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
            MLP       XGBoost       KAN
             │           │           │
             └───────────┼───────────┘
                         ▼
                   Best 2D Model
                         │
                         ▼
                Multi-view 3D Pose
                         │
                         ▼
                   Best Classifier
                         │
                         ▼
                   Final Evaluation
```

---

# 4. Pipeline A — Project Design 20242025

## Tujuan

Dataset digunakan sebagai **baseline berbasis citra** untuk mengetahui seberapa baik informasi visual langsung dapat digunakan untuk klasifikasi postur.

Audit sebelumnya menunjukkan:
- gambar berasal dari frame;
- terdapat kemiripan antargambar;
- tidak merepresentasikan full-body dengan baik.

Karena itu dataset ini **tidak digunakan sebagai basis utama rekonstruksi 3D**.

## Pipeline

```text
Image
  │
  ▼
Dataset audit / cleaning
  │
  ▼
Resize
  │
  ▼
Normalization
  │
  ▼
Image Classification
  │
  ├── CNN baseline
  │
  └── YOLO Classification (opsional)
  │
  ▼
Posture Class
```

### EXP-PD-01 — CNN baseline

Input RGB image → preprocessing → CNN → posture class.

Tujuan: mendapatkan baseline image-based.

### EXP-PD-02 — YOLO Classification (opsional)

Dilakukan jika diperlukan untuk membandingkan klasifikasi berbasis YOLO dengan CNN.

**Catatan:** YOLO Classification berbeda dari YOLO Detection; target utamanya bukan bounding box.

### Preprocessing

1. validasi gambar;
2. resize sesuai input model;
3. normalisasi;
4. augmentasi hanya pada training;
5. validation/test tanpa augmentasi.

### Evaluasi

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Confusion Matrix

Macro-F1 diprioritaskan jika distribusi kelas tidak seimbang.

---

# 5. Pipeline B — Postureexercise

## Tujuan

Menggunakan informasi **2D keypoint** sebagai representasi tubuh sehingga classifier tidak bekerja langsung pada seluruh piksel.

## Pipeline

```text
Image / Ground-truth Keypoint
          │
          ▼
      Keypoint Audit
          │
          ▼
     Keypoint Cleaning
          │
          ▼
       Normalization
          │
          ▼
      Feature Vector
          │
     ┌────┼─────┐
     ▼    ▼     ▼
    MLP XGBoost KAN
     │    │     │
     └────┼─────┘
          ▼
    Posture Class
```

Jumlah dan nama keypoint **harus diambil dari anotasi aktual**, bukan diasumsikan full-body.

### Eksperimen

- **EXP-PE-01:** keypoint → MLP
- **EXP-PE-02:** keypoint → XGBoost
- **EXP-PE-03:** keypoint → KAN, jika jumlah data dan format memungkinkan

Ketiga classifier harus menggunakan dataset, split, keypoint, dan preprocessing yang sama agar perbandingan adil.

---

# 6. Pipeline C — IKORN / 4-Keypoint

## Karakteristik hasil audit

- 655 citra;
- 2 kelas: Good dan Bad;
- 4 keypoint;
- `bottom`, `shoulder`, `head`, `back`;
- near-duplicate tinggi;
- indikasi leakage antar split.

## Tujuan

Menjadi **minimal-keypoint baseline**:

> Seberapa baik postur dapat diklasifikasikan hanya menggunakan 4 keypoint?

## Pipeline

```text
4 Keypoints
    │
    ▼
Missing / visibility check
    │
    ▼
Coordinate normalization
    │
    ▼
Feature vector
    │
    ▼
MLP
    │
    ▼
Good / Bad
```

Eksperimen:

**EXP-IK-01:** 4 keypoint → normalization → MLP → Good/Bad.

Karena fungsi dataset ini adalah baseline sederhana, tidak perlu banyak classifier.

### Validitas split

Karena audit menunjukkan near-duplicate dan potensi leakage, split bawaan **tidak langsung dianggap sebagai evaluasi independen**.

Prioritas:
- identifikasi near-duplicate;
- hindari pasangan mirip lintas split;
- jika subject/sequence tersedia, gunakan subject/sequence split.

---

# 7. Pipeline D — Sitting Posture Detection Initial

## Fungsi sementara

Dataset digunakan untuk **YOLO Object Detection baseline**.

Audit yang sudah diketahui:
- 490 image;
- 4 kelas;
- bounding box;
- 0 bbox invalid;
- 40 pasangan near-duplicate;
- 17 potensi leakage lintas split;
- 1 image tanpa anotasi.

Contact sheet masih diperlukan.

## Pipeline

```text
RGB Image
    │
    ▼
YOLO Detection
    │
    ├── Bounding Box
    │
    └── Posture Class
          │
          ▼
       Evaluation
```

Kelas:
- `good_posture`
- `leaning_backward`
- `leaning_forward`
- `slouch`

### Model

YOLO Object Detection sebagai baseline.

### Evaluasi

- Precision
- Recall
- mAP@50
- mAP@50–95
- per-class performance
- confusion matrix bila relevan
- inference speed/FPS sebagai tambahan bila real-time dibahas

### Quality gate

Sebelum eksperimen final:
1. periksa contact sheet;
2. periksa image tanpa anotasi;
3. periksa 17 potensi leakage;
4. pahami variasi subject/sequence sejauh metadata memungkinkan.

---

# 8. Pipeline E — Dataset Privat / Proposed Method

Dataset privat dirancang untuk mengatasi keterbatasan utama dataset publik:
- full-body;
- multi-view;
- capture relatif independen;
- keypoint lebih lengkap;
- subject-independent evaluation;
- metadata kondisi kursi;
- kemungkinan rekonstruksi 3D.

## E1. Akuisisi

```text
Subject
   │
   ├── Camera 1
   │
   └── Camera 2
          │
          ▼
      Same capture_id
          │
          ▼
      Paired images
```

Jika tersedia kamera ketiga, kamera juga menggunakan `capture_id` yang sama.

## E2. Quality Control

```text
Raw Images
    │
    ▼
Automatic QC
    ├── corrupt
    ├── resolution
    ├── blur
    ├── dark
    ├── duplicate
    └── near-duplicate
    │
    ▼
Visual QC
    │
    ▼
Clean Dataset
```

## E3. YOLO-Pose

```text
Full-body RGB
      │
      ▼
  YOLO-Pose
      │
      ▼
2D Body Keypoints
```

Tujuan: memperoleh koordinat keypoint tubuh.

Jika pretrained YOLO-Pose belum memadai, fine-tuning menggunakan anotasi pose privat menjadi opsi.

---

# 9. Pipeline F — 2D Keypoint Classification

```text
2D Keypoints
     │
     ▼
Visibility / confidence filtering
     │
     ▼
Coordinate normalization
     │
     ▼
Feature vector
     │
 ┌───┼────┐
 ▼   ▼    ▼
MLP XGBoost KAN
 │   │    │
 └───┼────┘
     ▼
Posture Class
```

Eksperimen:

- **EXP-PR-01:** YOLO-Pose → 2D keypoint → MLP
- **EXP-PR-02:** YOLO-Pose → 2D keypoint → XGBoost
- **EXP-PR-03:** YOLO-Pose → 2D keypoint → KAN

Semua memakai split yang sama.

---

# 10. Pemilihan Best 2D Model

```text
MLP
XGBoost
KAN
  │
  ▼
Comparison
  │
  ├── Macro-F1
  ├── Accuracy
  ├── Precision
  ├── Recall
  └── Confusion Matrix
  │
  ▼
Best 2D Classifier
```

Macro-F1 menjadi kandidat metrik utama.

---

# 11. Pipeline G — Multi-view 3D

Setelah baseline 2D stabil:

```text
Camera 1                     Camera 2
   │                            │
   ▼                            ▼
YOLO-Pose                    YOLO-Pose
   │                            │
   ▼                            ▼
2D Keypoints                2D Keypoints
   │                            │
   └────────────┬───────────────┘
                ▼
        Camera Calibration
                │
                ▼
          Triangulation
                │
                ▼
          3D Keypoints
                │
                ▼
          Normalization
                │
                ▼
        Best 2D Classifier
                │
                ▼
        3D Posture Result
```

**EXP-PR-04:** multi-view 2D keypoint → triangulation → 3D keypoint → best classifier.

Tujuan:

> mengetahui apakah informasi kedalaman meningkatkan performa dibandingkan representasi 2D.

---

# 12. Daftar eksperimen

| ID | Dataset | Input | Metode | Level |
|---|---|---|---|---|
| EXP-PD-01 | Project Design | RGB | CNN | Image |
| EXP-PD-02 | Project Design | RGB | YOLO Classification | Image |
| EXP-PE-01 | Postureexercise | 2D Keypoint | MLP | 2D Pose |
| EXP-PE-02 | Postureexercise | 2D Keypoint | XGBoost | 2D Pose |
| EXP-PE-03 | Postureexercise | 2D Keypoint | KAN | 2D Pose |
| EXP-IK-01 | IKORN | 4 Keypoint | MLP | Minimal Pose |
| EXP-SPD-01 | Sitting Posture | RGB + BBox | YOLO Detection | Detection |
| EXP-PR-01 | Private | 2D Keypoint | MLP | 2D Pose |
| EXP-PR-02 | Private | 2D Keypoint | XGBoost | 2D Pose |
| EXP-PR-03 | Private | 2D Keypoint | KAN | 2D Pose |
| EXP-PR-04 | Private | 3D Keypoint | Best classifier | 3D Pose |

---

# 13. Prinsip split dataset

Prioritas:

```text
Subject
   ↓
Train / Validation / Test
```

Bukan:

```text
Frame
   ↓
Random split
```

Terutama untuk data yang berasal dari sequence/video.

Jika metadata subject tidak tersedia pada dataset publik, keterbatasan tersebut harus dicatat dan tidak diklaim sebagai subject-independent evaluation.

Untuk dataset privat, `subject_id` harus tersedia sejak pengumpulan data.

---

# 14. Preprocessing

## Image

```text
Resize
→ normalization
→ train augmentation
```

## 2D keypoint

```text
coordinate validation
→ visibility/confidence handling
→ translation normalization
→ scale normalization
→ feature vector
```

## 3D keypoint

```text
3D coordinate validation
→ coordinate-system consistency
→ translation normalization
→ scale normalization
→ feature vector
```

Parameter preprocessing tidak boleh ditentukan menggunakan informasi validation/test.

---

# 15. Evaluasi

## Classification

Wajib:
- Accuracy;
- Precision;
- Recall;
- F1-score;
- Macro-F1;
- Confusion Matrix.

## Detection

Wajib:
- Precision;
- Recall;
- mAP@50;
- mAP@50–95.

Tambahan:
- inference time;
- FPS.

## Pose

Jika pose estimator dievaluasi:
- metrik pose yang sesuai framework/dataset;
- keypoint confidence;
- missing/invalid keypoint rate.

Untuk klasifikasi akhir:
- Accuracy;
- Macro-F1;
- Precision;
- Recall;
- F1.

---

# 16. Ablation study

### A1 — Image vs 2D Keypoint

```text
RGB
VS
2D Keypoint
```

Pertanyaan: apakah representasi pose lebih efektif daripada citra langsung?

### A2 — Minimal vs Full Keypoint

```text
4 Keypoint
VS
Full-body Keypoint
```

Pertanyaan: apakah penambahan bagian tubuh meningkatkan klasifikasi?

### A3 — 2D vs 3D

```text
2D Keypoint
VS
3D Keypoint
```

Pertanyaan: apakah informasi kedalaman meningkatkan hasil?

### A4 — Classifier

```text
MLP
VS
XGBoost
VS
KAN
```

Pertanyaan: classifier mana yang paling sesuai untuk representasi keypoint?

---

# 17. Penentuan hasil terbaik

Hasil terbaik tidak ditentukan sebelum eksperimen.

Metrik utama yang disarankan:

> **Macro-F1**

Metrik pendukung:
1. F1 per kelas;
2. Recall per kelas;
3. Precision per kelas;
4. confusion matrix;
5. accuracy.

Untuk real-time:
- FPS;
- latency;
- ukuran model bila relevan.

---

# 18. Urutan implementasi

```text
STEP 1  Audit final dataset publik
   ↓
STEP 2  Project Design baseline
   ↓
STEP 3  Postureexercise keypoint baseline
   ↓
STEP 4  IKORN 4-keypoint baseline
   ↓
STEP 5  Contact sheet Sitting Posture Detection
   ↓
STEP 6  YOLO Detection baseline
   ↓
STEP 7  Finalisasi dataset privat
   ↓
STEP 8  Pengumpulan dataset privat
   ↓
STEP 9  QC + annotation
   ↓
STEP 10 YOLO-Pose / 2D keypoint
   ↓
STEP 11 MLP / XGBoost / KAN
   ↓
STEP 12 Pilih best 2D
   ↓
STEP 13 Multi-view 3D
   ↓
STEP 14 2D vs 3D
   ↓
STEP 15 Final evaluation
   ↓
STEP 16 Real-time prototype
```

---

# 19. Status keputusan saat ini

## Sudah cukup jelas

- dataset publik digunakan sebagai baseline;
- dataset privat digunakan sebagai proposed/main dataset;
- YOLO Detection untuk dataset bounding-box;
- YOLO-Pose untuk memperoleh 2D keypoint;
- MLP sebagai baseline classifier keypoint;
- XGBoost/KAN sebagai pembanding;
- eksperimen 2D vs 3D;
- Macro-F1 sebagai kandidat metrik utama.

## Belum dikunci

- kelayakan final `Sitting Posture Detection Initial`;
- jumlah/nama keypoint Postureexercise;
- jumlah kelas final dataset privat;
- jumlah subject dan capture dataset privat;
- konfigurasi kamera final;
- apakah pretrained YOLO-Pose cukup atau perlu fine-tuning;
- hyperparameter final.

---

# 20. Prinsip metodologi akhir

Penelitian tidak bertujuan membuktikan bahwa **YOLO selalu paling baik**.

YOLO digunakan sesuai fungsi:

```text
YOLO Detection
→ baseline bounding-box

YOLO-Pose
→ memperoleh body keypoint

Classifier
→ mengklasifikasikan postur berdasarkan keypoint

Multi-view triangulation
→ memperoleh representasi 3D

3D classifier
→ menguji manfaat informasi kedalaman
```

Dengan struktur ini, kontribusi penelitian dapat diarahkan pada **perbandingan representasi postur dan pengembangan pipeline multi-view 2D-to-3D untuk klasifikasi postur**, bukan sekadar mengganti-ganti model YOLO.
