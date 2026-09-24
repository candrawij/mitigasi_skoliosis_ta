# RENCANA IMPLEMENTASI FINAL BERDASARKAN PROGRES TERBARU
## Pipeline Multi-View 2D vs Stereo 3D untuk Klasifikasi Postur Duduk

**Basis dokumen:** `progress_report.md`  
**Status saat ini:** Eksperimen utama 2D/3D, tri-model comparison, subject-aware evaluation, dan investigasi Fold 3 sudah selesai. Tahap berikutnya difokuskan pada perbaikan robustness 3D, validasi deployment stereo, dan finalisasi hasil artikel.

---

# 1. Tujuan Dokumen

Dokumen ini digunakan sebagai rencana kerja lanjutan setelah penyelesaian:

- pipeline 2D Multi-View;
- pipeline Stereo 3D;
- evaluasi subject-aware 5-fold;
- perbandingan 2D Core, 2D Best, dan 3D Core;
- investigasi akar masalah Fold 3;
- deployment model 3D offline.

Target akhir:

1. Memastikan hasil 3D tidak bias oleh perbedaan orientasi rig.
2. Menguji apakah canonicalization dapat memperbaiki generalisasi 3D.
3. Memastikan eksperimen 2D vs 3D benar-benar terkunci secara metodologis.
4. Menyelesaikan inference stereo dari raw image hingga prediksi.
5. Menyelesaikan real-time stereo 3D dan pengukuran latency/FPS.
6. Mengisi hasil final ke artikel ilmiah.

---

# 2. Status Eksperimen Saat Ini

## 2.1 Dataset Final

```text
Total 6-class captures : 727
Stereo-QC Usable       : 657 (90.4%)
Model-Ready 3D         : 403 (55.4%)
Excluded Rig           : 70 (9.6%)
Right-Hip Missing      : 254 (34.9%)
```

## 2.2 Dataset Perbandingan Fair

```text
Intersection dataset : 403 capture
Subjects             : 18
Validation           : Subject-Aware 5-Fold
2D dan 3D            : menggunakan capture dan fold yang sama
```

## 2.3 Hasil Tri-Model

| Model | Feature | Pooled Macro F1 | Fold Mean Macro F1 |
|---|---:|---:|---:|
| 2D Core | 36 | 0.6863 | 0.6669 ± 0.0846 |
| 2D Best Practical | 42 | 0.7039 | 0.6875 ± 0.0701 |
| 3D Core Stereo | 25 | 0.6182 | 0.6046 ± 0.2519 |

## 2.4 Temuan Penting

- 2D Best mempunyai performa global tertinggi.
- 2D Core tetap lebih tinggi daripada 3D Core secara pooled Macro F1.
- 3D Core unggul pada beberapa fold ketika rig konsisten.
- Fold 3 mengalami collapse besar pada 3D.
- S003 dan S004 pada CAL_004 menjadi sumber domain shift utama.
- 2D relatif stabil karena tidak bergantung pada transformasi ekstrinsik stereo.
- Single-capture 18/18 hanya digunakan sebagai **smoke test**, bukan bukti generalisasi.

---

# 3. Task yang Sudah Selesai

| Task | Status | Hasil Utama |
|---|---|---|
| 3D-01 Repository Backup | ✅ Selesai | Git checkpoint/tag dibuat |
| 3D-02 Build 6-Class Manifest | ✅ Selesai | 727 capture 6-class |
| 3D-03 Recalculate QC 3D | ✅ Selesai | 657 Stereo-QC usable |
| 3D-04 Coordinate Convention | ✅ Selesai | Frame CAM01 terdokumentasi |
| 3D-05 Lock 3D Feature Schema | ✅ Selesai | 25 fitur terkunci |
| 3D-06 Extract 3D Features | ✅ Selesai | 403 Model-Ready samples |
| 3D-07 Feature Audit | ✅ Selesai | Zero inf/-inf, NaN teridentifikasi |
| 3D-08 Build Intersection | ✅ Selesai | 403 matched captures |
| 3D-09 Common 5-Fold | ✅ Selesai | zero subject leakage |
| 3D-10 Train XGBoost 3D | ✅ Selesai | 5-fold model training |
| 3D-11 Evaluate 3D | ✅ Selesai | pooled F1 0.6182 |
| 3D-12 Tri-Model Comparison | ✅ Selesai | 2D Core vs 2D Best vs 3D Core |
| 3D-13 Deployment Model 3D | ✅ Selesai | `xgboost_3d.pkl` + metadata |
| 3D-14 Single Capture Verification | ✅ Selesai | 18/18 smoke test passed |

---

# 4. Masalah yang Ditemukan

## 4.1 Fold 3 Collapse

Fold 3:

```text
3D Macro F1 : 0.1052
3D Accuracy : 22.2%
```

Subjek:

```text
S003
S004
S022
S024
```

Sebagian besar sampel bermasalah berasal dari:

```text
CAL_004
```

Temuan numerik:

```text
55/55 sampel CAL_004
→ diprediksi leaning_right
```

Distribusi beberapa fitur 3D menunjukkan pergeseran besar pada rig tersebut, terutama:

```text
shoulder_roll_deg
torso_sagittal_lean_deg
```

---

# 5. Kenapa Perlu Perbaikan Lanjutan?

Masalah utama saat ini bukan lagi sekadar "model 3D kurang akurat".

Masalah yang lebih penting adalah:

```text
representasi 3D masih terikat pada camera coordinate frame
```

Jika orientation rig berubah:

```text
same human posture
→ different numeric 3D orientation
→ feature distribution shift
→ classifier failure
```

Karena itu, sebelum hasil 3D dikunci sebagai final, perlu satu eksperimen tambahan untuk memisahkan:

```text
kelemahan representasi 3D
```

dari:

```text
kelemahan coordinate-frame antar rig
```

---

# 6. Rencana Perbaikan

## FIX-01 — Implement 3D Canonicalization

### Tujuan

Mengurangi ketergantungan fitur 3D terhadap orientasi fisik rig / CAM01.

### Implementasi

Buat:

```text
04_scripts/preprocessing/
canonicalize_private_3d_pose.py
```

Pipeline:

```text
triangulated 3D joints
↓
translate to hip center
↓
construct canonical body axes
↓
rotate pose into common reference
↓
normalize scale
↓
extract 3D features
```

### Acceptance Criteria

```text
[ ] CAL_004 tidak lagi memiliki orientation shift ekstrem
[ ] tanda sagittal lean konsisten antar-rig
[ ] left/right semantics tetap benar
[ ] tidak ada NaN baru
[ ] tidak ada perubahan label
```

---

## FIX-02 — Audit Circular Angle Representation

### Tujuan

Mencegah masalah discontinuity angle:

```text
-180° ↔ +180°
```

Audit:

```text
shoulder_roll_deg
hip_roll_deg
torso_lateral_lean_deg
torso_sagittal_lean_deg
```

Simpan:

```text
median
IQR
histogram
circular mean
```

Eksperimen opsional:

```text
angle
vs
sin(angle), cos(angle)
```

Jangan mengganti feature final sebelum hasil ablation diketahui.

---

## FIX-03 — Re-Extract Canonicalized 3D Features

Buat:

```text
04_scripts/preprocessing/
extract_private_3d_features_canonical.py
```

Output:

```text
02_data/private_processed/features/
private_features_3d_canonical.csv
```

Gunakan conceptual feature set yang sama agar perbandingan adil.

---

## FIX-04 — Audit Feature Distribution Per Rig

Output:

```text
02_data/private_processed/audit/
feature_3d_canonical_audit.csv
```

Bandingkan minimal:

```text
CAL_001
CAL_004
CAL_005
CAL_008
CAL_009
CAL_011
```

Periksa:

```text
shoulder_roll
hip_roll
sagittal lean
lateral lean
depth offsets
```

Target:

```text
rig-specific distribution shift berkurang
```

---

## FIX-05 — Retrain Canonicalized 3D

Buat:

```text
04_scripts/training/
train_private_xgboost_3d_canonical.py
```

Gunakan:

```text
same 403 captures
same fold file
same class order
same XGBoost protocol
```

Jangan membuat fold baru.

---

## FIX-06 — Compare Current 3D vs Canonicalized 3D

Tabel evaluasi:

| Metric | 3D Current | 3D Canonical |
|---|---:|---:|
| Accuracy | 61.79% | ? |
| Pooled Macro F1 | 0.6182 | ? |
| Fold Mean Macro F1 | 0.6046 ± 0.2519 | ? |
| Fold 3 Macro F1 | 0.1052 | ? |

Pertanyaan utama:

```text
Apakah canonicalization memperbaiki Fold 3?
```

Interpretasi:

### Jika meningkat besar

```text
Masalah utama 3D terutama berasal dari coordinate-frame / rig domain shift.
```

### Jika tetap rendah

```text
Keterbatasan lebih banyak berasal dari triangulation quality,
feature robustness, dan subject variation.
```

Keduanya tetap merupakan hasil penelitian yang valid.

---

# 7. Eksperimen Final yang Disarankan

Setelah canonicalization:

| Model | Tujuan |
|---|---|
| 2D Core | baseline representation |
| 2D Best | practical optimized system |
| 3D Core Current | stereo camera-frame baseline |
| 3D Canonical | improved stereo representation |

Primary scientific comparison:

```text
2D Core
vs
3D Canonical Core
```

Secondary practical comparison:

```text
2D Best
vs
3D Canonical
```

---

# 8. TASK 3D-15 — Raw Stereo Image Pair Test

**Status:** ⏳ Belum

Script:

```text
04_scripts/inference/
infer_private_pair_3d.py
```

Input:

```text
CAM01 image
CAM02 image
calibration ID
```

Pipeline:

```text
raw images
↓
YOLOv8-Pose
↓
target person selection
↓
keypoint correspondence
↓
triangulation
↓
3D QC
↓
canonicalization
↓
3D feature extraction
↓
XGBoost
↓
posture prediction
```

Output minimal:

```text
Prediction
Confidence
Calibration ID
3D QC status
Latency
```

### Kenapa dilakukan?

Untuk membuktikan bahwa sistem bekerja mulai dari:

```text
pixel
```

hingga:

```text
prediction
```

bukan hanya dari file feature yang sudah jadi.

---

# 9. TASK 3D-16 — Real-Time Stereo 3D

**Status:** ⏳ Belum

Script:

```text
04_scripts/inference/
infer_realtime_stereo_3d.py
```

Catat:

```text
YOLO latency
triangulation latency
feature extraction latency
XGBoost latency
end-to-end latency
FPS
```

UI minimal:

```text
POSTURE
CONFIDENCE
3D QC
CALIBRATION ID
FPS
LATENCY
```

### Kenapa dilakukan?

Scientific evaluation menjawab:

```text
"seberapa baik model menggeneralisasi?"
```

Real-time evaluation menjawab:

```text
"apakah sistem praktis dijalankan?"
```

---

# 10. TASK 3D-17 — Final Article Update

**Status:** ⏳ Belum

Update:

```text
Section 4.5 — Stereo 3D Result
Section 4.6 — 2D vs 3D Comparison
Discussion
Limitations
Conclusion
```

Masukkan:

```text
Stereo-QC vs Model-Ready distinction
Tri-model result
Fold-by-fold result
CAL_004 failure analysis
Canonicalization experiment
Coverage comparison
Latency/FPS
```

---

# 11. Urutan Implementasi Berikutnya

```text
FIX-01
Canonicalize 3D pose
    ↓
FIX-02
Audit circular angles
    ↓
FIX-03
Re-extract canonical 3D features
    ↓
FIX-04
Audit feature distribution per rig
    ↓
FIX-05
Retrain canonicalized 3D
    ↓
FIX-06
Compare current vs canonical 3D
    ↓
LOCK SCIENTIFIC RESULT
    ↓
3D-15
Raw stereo pair inference
    ↓
3D-16
Real-time stereo
    ↓
3D-17
Article update
```

---

# 12. Stop Conditions

Jangan mengunci hasil canonical 3D jika:

```text
[ ] left/right semantics berubah setelah rotation
[ ] coordinate convention tidak terdokumentasi
[ ] fold berubah dari eksperimen sebelumnya
[ ] jumlah intersection berubah tanpa alasan
[ ] label berubah
[ ] NaN geometry diisi dengan 0
[ ] test fold dipakai untuk memilih transformasi
[ ] canonicalization ditentukan berdasarkan test-label error
```

Canonicalization harus ditentukan berdasarkan geometri, bukan berdasarkan hasil prediksi test.

---

# 13. Hasil Akhir yang Diharapkan

## 13.1 Output Data

```text
private_features_3d_canonical.csv
feature_3d_canonical_audit.csv
```

## 13.2 Output Eksperimen

```text
07_results/experiments/private_final/
├── 2d_core/
├── 2d_best/
├── 3d_current/
├── 3d_canonical/
└── comparison_final/
```

## 13.3 Output Model

```text
06_models/keypoint_3d/private_final_canonical/
```

## 13.4 Output Evaluasi

```text
fold_metrics.csv
oof_predictions.csv
classification_report.csv
confusion_matrix.png
per_class_metrics.csv
rig_robustness_report.csv
```

## 13.5 Output Deployment

```text
offline stereo pair prediction
real-time stereo prediction
latency report
FPS report
```

---

# 14. Hasil Ilmiah yang Sudah Terbukti

Saat ini:

```text
2D Core pooled Macro F1 = 0.6863
2D Best pooled Macro F1 = 0.7039
3D Core pooled Macro F1 = 0.6182
```

Temuan:

```text
3D dapat sangat kompetitif pada fold tertentu,
tetapi sensitif terhadap perubahan rig.
```

Pertanyaan berikutnya:

```text
"seberapa besar performa 3D dapat distabilkan
dengan coordinate canonicalization?"
```

---

# 15. Kriteria Penelitian Dianggap Final

## Scientific Experiment Final

```text
[ ] canonical 3D experiment selesai
[ ] Fold 3 sudah diuji ulang
[ ] current vs canonical 3D comparison tersedia
[ ] final 2D vs 3D comparison terkunci
[ ] interpretasi hasil tidak berubah lagi
```

## Deployment Final

```text
[ ] raw stereo pair test berhasil
[ ] real-time stereo berhasil
[ ] FPS tercatat
[ ] latency tercatat
[ ] failure mode terdokumentasi
```

## Artikel Final

```text
[ ] Section 4.5 selesai
[ ] Section 4.6 selesai
[ ] Discussion memasukkan rig-domain issue
[ ] Limitations memasukkan coverage 3D
[ ] Conclusion berdasarkan subject-aware results
```

---

# 16. Prioritas Pengerjaan Sekarang

```text
P1 — FIX-01: Implement 3D canonicalization
P2 — FIX-02: Audit circular angle
P3 — FIX-03/04: Re-extract & audit
P4 — FIX-05/06: Retrain & compare
P5 — 3D-15: Raw stereo-pair inference
P6 — 3D-16: Real-time test
P7 — 3D-17: Final article update
```

---

# 17. Template Hasil Akhir

Isi angka setelah eksperimen canonical selesai:

```text
Pada evaluasi subject-aware yang menggunakan subset capture dan fold
identik, representasi 2D Multi-View menghasilkan performa yang lebih
stabil dibandingkan representasi stereo 3D camera-frame. Analisis
per-fold menunjukkan bahwa penurunan performa stereo 3D berkaitan
dengan sensitivitas terhadap perubahan orientasi rig kalibrasi.

Setelah dilakukan canonicalization koordinat 3D, performa berubah dari:

Current 3D Macro F1   : 0.6182
Canonical 3D Macro F1 : [ISI HASIL]

Fold 3:
Current               : 0.1052
Canonical             : [ISI HASIL]

Hasil tersebut digunakan untuk menentukan apakah keterbatasan utama
stereo 3D berasal dari domain shift koordinat atau dari kualitas
rekonstruksi dan kelengkapan keypoint.
```
