# TASK PENYELESAIAN STEREO 3D FINAL
## Repositori `mitigasi_skoliosis_ta`

**Tujuan utama:** menyelesaikan cabang stereo 3D sampai dapat dilatih, dievaluasi, dibandingkan secara fair dengan model 2D, dan diuji pada inferensi nyata.

**Status saat ini:**
- Dataset privat 24 subjek sudah tersedia.
- Pair CAM01–CAM02 sudah tersedia.
- Kalibrasi stereo sudah tersedia.
- Keypoint 2D dan keypoint 3D sudah tersedia.
- QC 3D sebelumnya sudah tersedia.
- Model 2D multi-view sudah berjalan.
- Bagian hasil stereo 3D dan perbandingan 2D-vs-3D belum final.

---

# 1. Kenapa Stereo 3D Harus Diselesaikan?

Stereo 3D tidak dikerjakan hanya untuk menambah fitur penelitian. Ada beberapa alasan metodologis penting.

## 1.1 Menyelesaikan pertanyaan penelitian

Jika penelitian ingin membahas apakah representasi pose 3D memberikan manfaat dibanding representasi 2D, maka cabang 3D harus sampai pada tahap:

```text
3D keypoints
↓
3D features
↓
XGBoost
↓
subject-aware evaluation
↓
comparison with 2D
```

Tanpa hasil ini, penelitian baru membuktikan bahwa sistem 2D bekerja, tetapi belum menjawab pertanyaan 2D vs 3D.

## 1.2 Menguji manfaat informasi depth

Representasi 2D kehilangan kedalaman. Stereo 3D memberikan komponen ruang yang dapat membawa informasi lateral, vertikal, dan depth.

Informasi depth berpotensi membantu membedakan:

```text
leaning_forward
leaning_backward
slouching
```

Namun manfaat tersebut tidak boleh diasumsikan. Harus dibuktikan melalui eksperimen.

## 1.3 Membuat artikel menjadi lengkap

Pada outline artikel, bagian:

```text
Stereo 3D Result
2D vs 3D Comparison
```

belum terisi. Menyelesaikan cabang ini akan mengisi dua bagian tersebut dengan hasil eksperimen nyata.

## 1.4 Menghindari kesimpulan yang hanya berdasarkan demo

Akurasi live 2D menunjukkan sistem berjalan secara praktis. Namun penelitian ilmiah membutuhkan evaluasi terkontrol:

```text
same subjects
same captures
same folds
same classifier
different representation
```

Karena itu, stereo 3D perlu dievaluasi dengan protokol yang sama.

---

# 2. Prinsip Utama Pengerjaan

Selama menyelesaikan 3D, gunakan aturan berikut:

```text
1. Jangan ulang pengambilan data.
2. Jangan ulang kalibrasi jika rig memang sudah tervalidasi.
3. Jangan ulang YOLOv8-Pose jika keypoint yang dibutuhkan sudah tersedia.
4. Gunakan 6 kelas final.
5. forward_head tidak masuk eksperimen utama.
6. reject tidak menjadi kelas ke-7.
7. Gunakan subject-aware evaluation.
8. Gunakan capture intersection untuk perbandingan 2D-vs-3D.
9. Invalid 3D joint tetap NaN.
10. Jangan mengganti NaN geometry menjadi 0.
11. Jangan memakai test fold untuk tuning.
12. Jangan berasumsi 3D harus lebih baik.
```

---

# 3. Ringkasan Task

| ID | Task | Tujuan | Output |
|---|---|---|---|
| 3D-01 | Backup repository | Mengamankan progres 2D | Git checkpoint |
| 3D-02 | Build manifest 6-class 3D | Memastikan data 3D sesuai taxonomy final | `private_6class_3d_manifest.csv` |
| 3D-03 | Recalculate QC 3D | Mengetahui jumlah usable 3D final | `private_3d_qc_6class_final.csv` |
| 3D-04 | Audit coordinate convention | Memastikan X/Y/Z konsisten | `private_3d_coordinate_convention.md` |
| 3D-05 | Lock 3D feature schema | Menentukan input classifier | `feature_schema_3d.json/md` |
| 3D-06 | Extract 3D features | Mengubah keypoint 3D menjadi tabel model | `private_features_3d.csv` |
| 3D-07 | Audit features | Mendeteksi NaN/inf/geometry abnormal | `feature_3d_audit.csv` |
| 3D-08 | Build 2D–3D intersection | Membuat perbandingan fair | `private_6class_intersection.csv` |
| 3D-09 | Build common 5-fold | Memastikan fold sama | `private_2d3d_intersection_5fold.csv` |
| 3D-10 | Train XGBoost 3D | Mendapatkan model dan OOF prediction | model + OOF |
| 3D-11 | Evaluate 3D | Mendapatkan metrik ilmiah | metrics + confusion matrix |
| 3D-12 | Compare 2D vs 3D | Menjawab pertanyaan penelitian | tabel comparison |
| 3D-13 | Fit deployment model | Menyiapkan model inference | final 3D model |
| 3D-14 | Test single capture | Verifikasi end-to-end | prediksi satu capture |
| 3D-15 | Test new stereo pair | Verifikasi input baru | prediction + QC |
| 3D-16 | Real-time stereo | Uji deployment praktis | live posture + FPS |
| 3D-17 | Update artikel | Memasukkan hasil final | Results 3D + comparison |

---

# 4. TASK 3D-01 — Backup Repository

## Yang dilakukan

```bash
git status
git add .
git commit -m "checkpoint before final stereo 3D experiment"
```

Opsional:

```bash
git tag before-final-stereo-3d
```

## Kenapa harus dilakukan?

Model 2D, feature schema, dan real-time pipeline sudah berjalan. Pengerjaan 3D akan menambah script, manifest, feature, model, dan evaluasi baru. Checkpoint membuat kondisi repository sebelum eksperimen 3D dapat dipulihkan jika terjadi konflik atau perubahan yang tidak diinginkan.

---

# 5. TASK 3D-02 — Build Manifest 6-Class Khusus 3D

## Script

```text
04_scripts/preprocessing/build_private_6class_3d_manifest.py
```

## Input

```text
03_metadata/private_templates/captures.csv
03_metadata/private_templates/calibration_map.csv
07_results/private_audit/private_3d_qc_final.csv
02_data/private_annotations/keypoints_3d/
```

## Kelas

```python
MAIN_CLASSES = [
    "upright",
    "leaning_forward",
    "leaning_backward",
    "leaning_left",
    "leaning_right",
    "slouching",
]
```

Exclude:

```text
forward_head
reject
```

## Expected total sebelum QC 3D

```text
727 captures
```

## Output

```text
02_data/private_processed/manifests/private_6class_3d_manifest.csv
```

## Kenapa harus dilakukan?

QC 3D sebelumnya dibuat ketika dataset masih memiliki label yang sekarang tidak semuanya dipakai. Jika langsung menggunakan file 3D lama untuk training, maka taxonomy, jumlah kelas, dan distribusi datanya tidak lagi identik dengan eksperimen final 2D. Manifest 6-class menjadi sumber data resmi untuk seluruh eksperimen 3D final.

## Stop condition

Jangan lanjut jika:

```text
total != 727
forward_head masih ada
reject masih ada
capture_id duplicate
subject_id kosong
```

---

# 6. TASK 3D-03 — Recalculate QC 3D untuk 6 Kelas

## Script

```text
04_scripts/audit/audit_private_3d_6class.py
```

## Hitung

```text
FULL
MASKING
EXCLUDE
```

Per:

```text
class
subject
calibration rig
```

## Output

```text
07_results/private_audit/private_3d_qc_6class_final.csv
```

Tambahkan summary:

```text
raw 6-class
FULL
MASKING
EXCLUDE
usable 3D
coverage %
```

## Kenapa harus dilakukan?

Angka usable 3D lama berasal dari keseluruhan dataset lama. Untuk eksperimen final, denominator yang benar adalah 727 capture 6-class. Tanpa perhitungan ulang, kita tidak tahu berapa data 3D final, kelas mana yang paling banyak kehilangan sampel, subjek mana yang tidak punya data 3D, atau rig mana yang paling bermasalah.

---

# 7. TASK 3D-04 — Audit Coordinate Convention

## Buat dokumentasi

```text
03_metadata/private_3d_coordinate_convention.md
```

Isi:

```text
X = ?
Y = ?
Z = ?
unit = ?
origin = ?
reference frame = ?
```

Gunakan nilai nyata dari pipeline yang ada, jangan menebak.

## Sanity check

Ambil beberapa capture dari:

```text
upright
leaning_left
leaning_right
leaning_forward
leaning_backward
slouching
```

Periksa perubahan:

```text
torso lateral component
torso depth component
shoulder depth difference
hip depth difference
```

## Kenapa harus dilakukan?

XGBoost dapat belajar pola numerik walaupun tanda sumbu salah. Artinya model mungkin tetap menghasilkan accuracy tertentu, tetapi feature engineering menjadi tidak dapat dijelaskan secara ilmiah. Kalau beberapa rig mempunyai orientasi tanda yang tidak konsisten, classifier berisiko belajar perbedaan rig, bukan posture.

---

# 8. TASK 3D-05 — Lock Feature Schema 3D

Gunakan baseline utama **25 fitur**.

## 8.1 Core joints

```text
nose
left_shoulder
right_shoulder
left_hip
right_hip
```

Derived points:

```text
shoulder_center
hip_center
```

## 8.2 Normalized coordinates — 15 fitur

```text
nose_x
nose_y
nose_z
left_shoulder_x
left_shoulder_y
left_shoulder_z
right_shoulder_x
right_shoulder_y
right_shoulder_z
left_hip_x
left_hip_y
left_hip_z
right_hip_x
right_hip_y
right_hip_z
```

## 8.3 Spatial geometry — 10 fitur

```text
shoulder_roll_deg
hip_roll_deg
torso_lateral_lean_deg
torso_sagittal_lean_deg
torso_3d_inclination_deg
head_torso_angle_3d_deg
head_depth_offset_norm
head_lateral_offset_norm
shoulder_depth_asymmetry_norm
hip_depth_asymmetry_norm
```

Total:

```text
15 + 10 = 25 features
```

## Kenapa harus dilakukan?

Feature schema harus dikunci sebelum melihat hasil test. Jika feature terus diubah setelah melihat test error, ada risiko test-set overfitting dan cherry-picking. Schema yang dikunci sebelumnya membuat eksperimen lebih dapat dipertanggungjawabkan.

---

# 9. TASK 3D-06 — Extract 3D Features

## Script

```text
04_scripts/preprocessing/extract_private_3d_features.py
```

## Pipeline

```text
keypoints_3d JSON
↓
QC
↓
core landmarks
↓
centering
↓
scale normalization
↓
spatial geometry
↓
25-D feature vector
```

## Center

```text
hip_center = (left_hip + right_hip) / 2
```

## Normalization

```text
P_normalized = (P - hip_center) / pose_scale
```

`pose_scale` harus konsisten untuk seluruh dataset dan didokumentasikan.

## Output

```text
02_data/private_processed/features/private_features_3d.csv
```

## Kenapa harus dilakukan?

Koordinat 3D mentah masih membawa informasi jarak subjek ke kamera, posisi absolut dalam ruang, dan variasi setup. Classifier seharusnya lebih banyak belajar shape dan geometry posture, bukan lokasi subjek di depan kamera. Centering dan normalization mengurangi shortcut tersebut.

---

# 10. TASK 3D-07 — Audit NaN dan Geometry

## Output

```text
02_data/private_processed/audit/feature_3d_audit.csv
```

Minimal:

```text
feature_name
count_valid
count_nan
nan_percentage
min
max
median
```

Periksa:

```text
NaN
inf
-inf
sudut di luar range logis
scale mendekati nol
duplicate capture
```

## Aturan missing joint

```text
invalid joint → NaN
```

Contoh jika nose invalid:

```text
nose_x = NaN
nose_y = NaN
nose_z = NaN
head_torso_angle_3d = NaN
head_depth_offset_norm = NaN
head_lateral_offset_norm = NaN
```

Jangan:

```text
invalid joint → 0
```

## Kenapa harus dilakukan?

Nilai `(0,0,0)` adalah koordinat numerik yang valid secara matematis. Jika dipakai untuk joint yang sebenarnya hilang, classifier dapat menganggapnya sebagai pola posture nyata. NaN mempertahankan arti bahwa nilai tidak tersedia.

---

# 11. TASK 3D-08 — Build 2D–3D Intersection

## Script

```text
04_scripts/preprocessing/build_private_2d3d_intersection.py
```

## Definisi

```text
capture usable 2D
AND
capture usable 3D
AND
label termasuk 6 kelas
```

## Output

```text
02_data/private_processed/manifests/private_6class_intersection.csv
```

Print:

```text
2D usable
3D usable
intersection usable
```

Juga distribution per class dan subject.

## Kenapa harus dilakukan?

Tidak fair membandingkan metric 2D dan 3D jika test sample keduanya berbeda. Intersection memastikan satu capture dinilai dalam dua representasi yang berbeda, sehingga faktor yang berubah terutama representation, bukan dataset.

---

# 12. TASK 3D-09 — Generate Common Subject-Aware 5-Fold

## Script

```text
04_scripts/evaluation/create_private_2d3d_subject_folds.py
```

Gunakan:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

```text
target = class_id
group = subject_id
```

## Output

```text
03_metadata/private_final_split/private_2d3d_intersection_5fold.csv
```

## Assertion

Untuk setiap fold:

```text
train_subjects ∩ test_subjects == empty
```

## Kenapa harus dilakukan?

Frame dari orang yang sama sangat mirip. Jika capture satu subjek muncul di training dan test, classifier bisa mengenali karakteristik individu, bukan generalisasi posture. Fold yang sama juga diperlukan agar test data 2D dan 3D benar-benar identik.

---

# 13. TASK 3D-10 — Train XGBoost 3D

## Script

```text
04_scripts/training/train_private_xgboost_3d.py
```

## Model

```python
objective = "multi:softprob"
num_class = 6
tree_method = "hist"
random_state = 42
```

Primary scoring:

```text
Macro F1
```

## Hyperparameter tuning

Tuning hanya pada outer training fold dan gunakan grouped inner CV jika tuning dilakukan.

Jangan gunakan outer test fold untuk memilih parameter.

## Kenapa harus dilakukan?

Tujuan penelitian bukan hanya membuat model yang cocok pada training data. Tujuan utamanya adalah mengukur generalisasi pada subjek yang tidak dilihat sebelumnya. Karena itu test fold harus benar-benar independen.

---

# 14. TASK 3D-11 — Evaluate 3D

## Output

```text
07_results/experiments/private_final/3d/
```

Simpan:

```text
fold_metrics.csv
oof_predictions.csv
best_params_per_fold.json
classification_report.txt
confusion_matrix.png
```

## Metrics

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Per-class Precision
Per-class Recall
Per-class F1
```

## Analisis khusus

Periksa confusion:

```text
slouching ↔ leaning_forward
leaning_forward ↔ leaning_backward
upright ↔ slight lean
```

## Kenapa harus dilakukan?

Accuracy saja dapat menyembunyikan kelas yang performanya buruk. Per-class metrics menunjukkan apakah 3D benar-benar membantu posture tertentu.

---

# 15. TASK 3D-12 — Compare 2D vs 3D

Gunakan OOF prediction dari intersection dan fold yang sama.

## Tabel utama

```text
Metric                2D        3D
Accuracy
Macro Precision
Macro Recall
Macro F1
```

## Per-class

```text
Class                 2D F1     3D F1
upright
leaning_forward
leaning_backward
leaning_left
leaning_right
slouching
```

Tambahkan:

```text
usable coverage
latency
FPS
setup complexity
```

## Kenapa harus dilakukan?

Ini adalah tahap yang menjawab pertanyaan penelitian secara langsung. Tiga hasil berikut semuanya sah:

```text
3D > 2D
2D > 3D
2D ≈ 3D
```

Jangan memaksa hasil 3D harus lebih baik.

---

# 16. TASK 3D-13 — Fit Deployment Model

Setelah scientific evaluation selesai:

```text
04_scripts/training/fit_private_deployment_model_3d.py
```

Simpan:

```text
06_models/keypoint_3d/private_final/
```

Minimal:

```text
xgboost_3d.pkl
feature_schema.json
class_map.json
model_metadata.json
coordinate_convention.json
```

## Kenapa harus dilakukan?

Model fold digunakan untuk evaluasi ilmiah, sedangkan model deployment digunakan untuk aplikasi. Keduanya mempunyai tujuan berbeda dan sebaiknya tidak dicampur.

---

# 17. TASK 3D-14 — Test Single Capture

## Script

```text
04_scripts/evaluation/test_private_single_capture_3d.py
```

Contoh output:

```text
Capture ID   : ...
True Label   : slouching
Prediction   : slouching
Confidence   : 0.87
3D QC        : FULL
Result       : CORRECT
```

## Kenapa harus dilakukan?

Training CSV bisa berhasil sementara pipeline input nyata salah. Single-capture test memastikan urutan JSON 3D → feature extraction → feature order → model → class mapping semuanya konsisten.

---

# 18. TASK 3D-15 — Test New Stereo Image Pair

Input:

```text
CAM01 image
CAM02 image
calibration ID
```

Pipeline:

```text
images
↓
YOLOv8-Pose
↓
target person
↓
correspondence
↓
triangulation
↓
3D QC
↓
feature extraction
↓
XGBoost
```

Jika 3D QC gagal:

```text
INVALID_3D
```

bukan memaksa salah satu dari enam posture.

## Kenapa harus dilakukan?

Ini membuktikan sistem dapat bekerja pada data input baru, bukan hanya membaca file feature yang sudah dibuat sebelumnya.

---

# 19. TASK 3D-16 — Real-Time Stereo 3D

## Script

```text
04_scripts/inference/infer_realtime_stereo_3d.py
```

Overlay minimal:

```text
POSTURE    : leaning_forward
CONFIDENCE : 89.3%
3D QC      : PASS
RIG        : CAL_xxx
FPS        : ...
LATENCY    : ... ms
```

## Kenapa harus dilakukan?

Scientific CV menjawab:

```text
"apakah model dapat menggeneralisasi?"
```

Real-time test menjawab:

```text
"apakah pipeline dapat digunakan sebagai sistem?"
```

Keduanya penting tetapi jangan dicampur.

---

# 20. TASK 3D-17 — Update Artikel

Setelah hasil selesai, isi:

```text
4.5 Stereo 3D Result
4.6 2D vs 3D Comparison
```

Tambahkan ke Discussion:

```text
apakah depth membantu?
kelas apa yang paling terbantu?
berapa kehilangan coverage karena triangulation?
berapa tambahan latency?
apakah keuntungan accuracy sebanding dengan kompleksitas setup?
```

## Kenapa harus dilakukan?

Artikel tidak hanya membutuhkan nilai metric. Artikel harus menjelaskan apa yang berubah, kenapa berubah, apa trade-off-nya, dan apa implikasi praktisnya.

---

# 21. Task Opsional — 3D Ear Feature Ablation

**Jangan dikerjakan sebelum baseline 25 fitur selesai.**

Audit dulu coverage telinga 3D:

```text
left ear
right ear
per class
per rig
```

Jika stabil, buat eksperimen sekunder:

```text
3D baseline
vs
3D + ear features
```

## Kenapa opsional?

Fitur telinga membantu 2D, tetapi triangulasi ear bisa lebih tidak stabil karena self-occlusion, view lateral, correspondence, dan geometry stereo. Karena itu eksperimen utama jangan ditunda demi fitur tambahan.

---

# 22. Urutan Eksekusi di IDE

Kerjakan dalam urutan ini:

```text
[1] 3D-01  Backup
      ↓
[2] 3D-02  Build 6-class manifest
      ↓
[3] 3D-03  Recalculate QC
      ↓
[4] 3D-04  Verify coordinate convention
      ↓
[5] 3D-05  Lock feature schema
      ↓
[6] 3D-06  Extract 3D features
      ↓
[7] 3D-07  Feature audit
      ↓
[8] 3D-08  Build intersection
      ↓
[9] 3D-09  Common subject-aware folds
      ↓
[10] 3D-10 Train XGBoost 3D
      ↓
[11] 3D-11 Evaluate 3D
      ↓
[12] 3D-12 Compare 2D vs 3D
      ↓
[13] 3D-13 Fit deployment model
      ↓
[14] 3D-14 Single-capture test
      ↓
[15] 3D-15 New stereo-pair test
      ↓
[16] 3D-16 Real-time stereo
      ↓
[17] 3D-17 Update article
```

---

# 23. Milestone

## Milestone A — Dataset 3D Final

```text
[ ] 727 6-class raw captures terverifikasi
[ ] FULL/MASK/EXCLUDE 6-class dihitung ulang
[ ] jumlah usable 3D diketahui
[ ] X/Y/Z convention terdokumentasi
```

### Kenapa milestone ini penting?

Sebelum model dibuat, kita harus yakin bahwa input 3D benar.

## Milestone B — Model 3D Scientific Evaluation

```text
[ ] 25-D feature table selesai
[ ] NaN audit selesai
[ ] subject-aware folds selesai
[ ] XGBoost 3D selesai
[ ] OOF predictions selesai
[ ] Macro F1 dan confusion matrix tersedia
```

### Kenapa milestone ini penting?

Pada titik ini kita sudah tahu apakah 3D dapat mengklasifikasikan enam posture pada subjek baru.

## Milestone C — 2D vs 3D Answered

```text
[ ] intersection final
[ ] same capture
[ ] same fold
[ ] 2D metrics
[ ] 3D metrics
[ ] per-class comparison
[ ] coverage comparison
```

### Kenapa milestone ini penting?

Ini merupakan hasil utama untuk menjawab pertanyaan penelitian.

## Milestone D — Deployment

```text
[ ] final 3D deployment model
[ ] single capture inference
[ ] new stereo pair inference
[ ] real-time inference
[ ] latency
[ ] FPS
```

### Kenapa milestone ini penting?

Ini menunjukkan bahwa metode tidak hanya bekerja sebagai eksperimen offline tetapi juga bisa dijalankan sebagai sistem.

---

# 24. Stop Conditions

Jangan lanjut training jika salah satu terjadi:

```text
[ ] total 6-class bukan 727
[ ] forward_head masih masuk target
[ ] reject masih masuk target
[ ] capture_id duplicate
[ ] coordinate convention tidak jelas
[ ] rig yang degenerate ikut dianggap FULL
[ ] NaN diganti menjadi 0
[ ] inf / -inf masih ada
[ ] subject overlap train-test
[ ] fold 2D berbeda dengan fold 3D
[ ] feature order training berbeda dengan inference
```

---

# 25. Definition of Done

Stereo 3D dianggap selesai ketika:

```text
[ ] dataset 6-class 3D final tersedia
[ ] usable 3D count final diketahui
[ ] 3D QC final tersedia
[ ] coordinate convention terdokumentasi
[ ] 25-D feature schema terkunci
[ ] feature table 3D selesai
[ ] feature QC selesai
[ ] intersection 2D–3D selesai
[ ] subject-aware 5-fold selesai
[ ] XGBoost 3D selesai
[ ] OOF prediction selesai
[ ] confusion matrix tersedia
[ ] Macro F1 mean ± SD tersedia
[ ] per-class F1 tersedia
[ ] comparison 2D-vs-3D tersedia
[ ] deployment model tersedia
[ ] single-capture inference berhasil
[ ] new stereo pair inference berhasil
[ ] real-time stereo berhasil atau limitasinya terdokumentasi
[ ] latency/FPS tercatat
[ ] Results 3D dan 2D-vs-3D pada artikel sudah terisi
```

---

# 26. Task Pertama yang Harus Dikerjakan Sekarang

Jangan mulai training terlebih dahulu.

Mulai dari:

```text
3D-02
Build 6-class 3D manifest
```

kemudian:

```text
3D-03
Recalculate FULL / MASKING / EXCLUDE
```

Target pertama adalah mendapatkan angka:

```text
727 raw 6-class captures
↓
berapa FULL?
berapa MASKING?
berapa EXCLUDE?
berapa usable 3D?
berapa intersection dengan usable 2D?
```

Setelah angka tersebut valid, baru lanjut ke feature extraction dan model training.

---

# 27. Ringkasan Alasan Seluruh Tahap

Secara sederhana:

```text
MANIFEST
→ memastikan data yang digunakan benar

QC
→ memastikan 3D yang buruk tidak dianggap data valid

COORDINATE AUDIT
→ memastikan arti X/Y/Z benar

FEATURE ENGINEERING
→ mengubah geometry 3D menjadi input classifier

INTERSECTION
→ memastikan 2D-vs-3D dibandingkan secara fair

SUBJECT-AWARE CV
→ memastikan tidak ada subject leakage

XGBOOST TRAINING
→ membangun classifier

OOF EVALUATION
→ mengukur generalisasi ilmiah

2D-vs-3D COMPARISON
→ menjawab pertanyaan penelitian

DEPLOYMENT TEST
→ membuktikan sistem dapat digunakan

ARTICLE UPDATE
→ mengubah hasil eksperimen menjadi kontribusi ilmiah
```
