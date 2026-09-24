# PANDUAN IMPLEMENTASI PIPELINE FINAL 2D–3D XGBOOST
## Repositori `mitigasi_skoliosis_ta`

**Status dokumen:** Panduan implementasi final untuk melanjutkan pekerjaan di IDE  
**Dataset final:** 24 subjek (`S001`–`S024`)  
**Raw capture:** 885 pasangan stereo / 1.770 citra Full HD  
**Kelas utama:** 6 kelas  
**Classifier utama:** XGBoost  
**Pose estimator:** YOLOv8n-Pose / COCO-17  
**Evaluasi utama:** Subject-aware grouped 5-fold cross-validation  
**Perbandingan utama:** 2D multi-view vs stereo 3D pada capture yang sama  
**Target akhir:** evaluasi model + model deployment + uji inferensi offline dan real-time

---

# 1. Tujuan Dokumen

Dokumen ini digunakan untuk **melanjutkan** repository yang sudah ada. Tidak perlu membuat ulang pipeline dari nol.

Target akhir pengerjaan:

1. Mengubah eksperimen privat dari 7 kelas menjadi **6 kelas utama**.
2. Mengeluarkan `forward_head` dari eksperimen utama tanpa menghapus raw data.
3. Mempertahankan `reject` sebagai **negative / invalid-input gate**, bukan kelas ketujuh.
4. Membuat ulang feature table 2D yang lebih sesuai dengan artikel acuan.
5. Membuat feature table stereo 3D berbasis spatial geometry.
6. Menggunakan **XGBoost sebagai satu-satunya classifier utama**.
7. Melakukan evaluasi **subject-aware grouped 5-fold**.
8. Membandingkan 2D vs 3D secara fair pada capture yang sama.
9. Menyimpan model deployment.
10. Menguji model pada capture dataset privat, pasangan citra baru, dan live camera / real-time prototype.

---

# 2. Keputusan Metodologi yang Sudah LOCKED

## 2.1 Dataset

```text
S001 ... S024
24 subjects
885 stereo captures
1770 images
```

Tidak dilakukan penambahan subjek lagi.

## 2.2 Kelas Utama

Gunakan hanya:

```text
0 upright
1 leaning_forward
2 leaning_backward
3 leaning_left
4 leaning_right
5 slouching
```

Urutan class ID di atas harus dipakai konsisten pada manifest, training, evaluation, inference, confusion matrix, dan model metadata.

## 2.3 `forward_head`

`forward_head` **tidak dihapus dari raw dataset**.

```text
use_for_main_experiment = false
exclusion_reason = class_removed_after_supervisor_review
```

## 2.4 `reject`

`reject` tidak dimasukkan dalam target XGBoost 6-class. Gunakan sebagai kategori invalid input / transition / out-of-frame / wrong person / pose tidak usable.

## 2.5 Pose Estimator

Tetap:

```text
yolov8n-pose.pt
COCO-17
```

Tidak perlu mengganti seluruh pipeline menjadi MoveNet.

## 2.6 Classifier

Hanya:

```text
XGBoost
```

MLP, CNN, dan eksperimen publik lama tetap disimpan sebagai preliminary benchmark.

---

# 3. Bagian Repository yang TIDAK Perlu Dikerjakan Ulang

```text
✓ raw data acquisition
✓ 24-subject collection
✓ CAM01/CAM02 pairing
✓ participants.csv
✓ captures.csv
✓ images.csv
✓ calibration_map.csv
✓ camera calibration
✓ target person selection
✓ YOLOv8-Pose extraction
✓ keypoints_2d JSON
✓ stereo person correspondence
✓ stereo triangulation
✓ keypoints_3d JSON
✓ existing 3D QC
✓ FULL / MASKING / EXCLUDE rules
✓ reprojection validation
```

Pipeline final dimulai dari **metadata + hasil anotasi yang sudah ada**.

---

# 4. Struktur Folder Tambahan yang Disarankan

```text
02_data/
└── private_processed/
    ├── manifests/
    │   ├── private_6class_all.csv
    │   ├── private_6class_2d.csv
    │   ├── private_6class_3d.csv
    │   └── private_6class_intersection.csv
    ├── features/
    │   ├── private_features_2d.csv
    │   ├── private_features_3d.csv
    │   ├── private_features_2d_intersection.csv
    │   └── private_features_3d_intersection.csv
    └── audit/
        ├── feature_2d_audit.csv
        ├── feature_3d_audit.csv
        └── intersection_audit.csv

03_metadata/
├── taxonomy_kelas.md
├── dataset_decisions.md
└── private_final_split/
    └── private_stratified_group_5fold.csv

04_scripts/
├── preprocessing/
│   ├── private_feature_common.py
│   ├── build_private_6class_manifest.py
│   ├── extract_private_2d_features.py
│   ├── extract_private_3d_features.py
│   └── build_private_intersection.py
├── training/
│   ├── train_private_xgboost_2d.py
│   ├── train_private_xgboost_3d.py
│   └── fit_private_deployment_models.py
├── evaluation/
│   ├── create_private_subject_folds.py
│   ├── evaluate_private_2d_vs_3d.py
│   └── test_private_single_capture.py
└── inference/
    ├── private_inference_common.py
    ├── infer_private_pair_2d.py
    ├── infer_private_pair_3d.py
    ├── infer_realtime_2d.py
    └── infer_realtime_stereo_3d.py

06_models/
├── keypoint_2d/private_final/
└── keypoint_3d/private_final/

07_results/
├── experiments/private_final/
│   ├── 2d/
│   ├── 3d/
│   └── comparison/
└── visualizations/private_final/
```

---

# 5. STEP 0 — Backup dan Freeze Kondisi Lama

```bash
git status
git add .
git commit -m "checkpoint before final 6-class XGBoost pipeline"
git tag private-24subj-before-final-pipeline
```

Jangan hapus file hasil pipeline lama.

---

# 6. STEP 1 — Update Metadata Keputusan Penelitian

## 6.1 `03_metadata/taxonomy_kelas.md`

Jangan menghilangkan sejarah 7-class. Tambahkan bagian baru:

```markdown
## Final Taxonomy — Supervisor Revision

Final main classification taxonomy:

1. upright
2. leaning_forward
3. leaning_backward
4. leaning_left
5. leaning_right
6. slouching

Excluded from main classification:
- forward_head

Negative / invalid-input category:
- reject
```

## 6.2 `03_metadata/dataset_decisions.md`

Tambahkan keputusan:

```markdown
### Final Supervisor Decision — 6-Class Private Experiment

- Dataset acquisition stopped at 24 subjects.
- `forward_head` removed from main experiment because the class definition
  was considered insufficiently verified for the final study.
- `reject` retained as negative/QC data and not trained as a seventh posture class.
- XGBoost selected as the sole final classifier.
- Final scientific comparison is 2D pose representation vs stereo 3D representation.
```

---

# 7. STEP 2 — Build Manifest 6-Class

Buat:

```text
04_scripts/preprocessing/build_private_6class_manifest.py
```

## 7.1 Input

Minimal:

```text
03_metadata/private_templates/captures.csv
03_metadata/private_templates/images.csv
03_metadata/private_templates/calibration_map.csv
07_results/private_audit/private_3d_qc_final.csv
```

## 7.2 Filter Kelas

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

`forward_head` dan `reject` tidak masuk `private_6class_all.csv`.

## 7.3 Kolom Minimal Manifest

```text
capture_id
subject_id
session_id
label
class_id
cam01_path
cam02_path
calibration_id
lateral_side
subset
status_2d
status_3d
qc_3d_status
```

Boleh menambahkan:

```text
selected_person_valid_cam01
selected_person_valid_cam02
core_2d_valid_cam01
core_2d_valid_cam02
reprojection_error
```

## 7.4 Acceptance Check

Script HARUS gagal (`raise`) jika:

```text
forward_head ada di label utama
reject ada di label utama
class_id bukan 0..5
subject_id kosong
capture_id duplikat
CAM01/CAM02 pairing tidak lengkap
```

## 7.5 Expected Count

Sebelum branch-specific QC:

```text
upright           126
leaning_forward   124
leaning_backward  118
leaning_left      119
leaning_right     120
slouching         120
---------------------
TOTAL             727
```

Gunakan assertion:

```python
assert len(df) == 727
```

Jika hasil bukan 727, **jangan lanjut training**.

Output:

```text
02_data/private_processed/manifests/private_6class_all.csv
```

---

# 8. STEP 3 — Definisi Core Keypoints

YOLO tetap menghasilkan 17 keypoint, tetapi feature utama menggunakan:

```text
nose             COCO 0
left_shoulder    COCO 5
right_shoulder   COCO 6
left_hip         COCO 11
right_hip        COCO 12
```

Derived points:

```text
shoulder_center = (left_shoulder + right_shoulder) / 2
hip_center      = (left_hip + right_hip) / 2
```

### Mandatory untuk normalisasi

```text
left_shoulder
right_shoulder
left_hip
right_hip
```

`nose` boleh NaN jika tidak valid, tetapi semua feature yang bergantung pada nose juga harus NaN.

---

# 9. STEP 4 — Canonicalization CAM02

Dataset memiliki lateral view kiri dan kanan. Jangan membiarkan arah lateral menjadi shortcut model.

Target:

```text
semua CAM02 diperlakukan seolah-olah memiliki orientasi lateral yang sama
```

Implementasi:

1. Lakukan centering terlebih dahulu.
2. Untuk capture dengan `lateral_side == "left"`, balik tanda koordinat horizontal CAM02:

```python
x_centered = -x_centered
```

3. Terapkan aturan yang sama pada feature CAM02 yang memiliki tanda horizontal/sagittal.
4. **Jangan** menukar label `leaning_left` menjadi `leaning_right`.

Buat unit test beberapa sample S001–S004 dan subject right-lateral untuk memastikan forward/backward mempunyai arah tanda yang konsisten.

---

# 10. STEP 5 — Ekstraksi Feature 2D

Buat:

```text
04_scripts/preprocessing/extract_private_2d_features.py
```

Input:

```text
private_6class_all.csv
02_data/private_annotations/keypoints_2d/
02_data/private_annotations/selected_person/
```

## 10.1 Centering

```text
C = hip_center
P_centered = P - C
```

## 10.2 Scale Normalization

Gunakan ukuran pose yang berasal dari geometry pose:

```text
S = max distance dari hip_center ke seluruh core landmark valid
P_normalized = P_centered / S
```

Syarat:

```text
S > epsilon
```

Jika `S` tidak valid, tandai `status_2d = invalid`.

## 10.3 Feature per View

Setiap kamera menghasilkan 18 feature.

### Normalized coordinates — 10

```text
nose_x
nose_y
left_shoulder_x
left_shoulder_y
right_shoulder_x
right_shoulder_y
left_hip_x
left_hip_y
right_hip_x
right_hip_y
```

### Engineered geometry — 8

```text
shoulder_slope_deg
hip_slope_deg
torso_inclination_deg
head_torso_angle_deg
head_to_shoulder_norm
torso_length_norm
head_horizontal_offset_norm
torso_horizontal_offset_norm
```

Prefix:

```text
cam01_
cam02_
```

Total:

```text
18 CAM01 + 18 CAM02 = 36 features
```

## 10.4 Formula Inti

```text
shoulder_center = mean(left_shoulder, right_shoulder)
hip_center      = mean(left_hip, right_hip)
torso           = shoulder_center - hip_center
head            = nose - shoulder_center
```

Shoulder slope:

```text
atan2(y_right_shoulder - y_left_shoulder,
      x_right_shoulder - x_left_shoulder)
```

Hip slope:

```text
atan2(y_right_hip - y_left_hip,
      x_right_hip - x_left_hip)
```

Torso inclination:

```text
angle(torso, image vertical axis)
```

Head-to-torso angle:

```text
angle(head, torso)
```

Offsets:

```text
head_horizontal_offset =
    (nose.x - shoulder_center.x) / S

torso_horizontal_offset =
    (shoulder_center.x - hip_center.x) / S
```

## 10.5 Missing Values

Jika nose invalid:

```text
nose_x                         = NaN
nose_y                         = NaN
head_torso_angle_deg           = NaN
head_to_shoulder_norm          = NaN
head_horizontal_offset_norm    = NaN
```

Jangan isi dengan `0`.

Output:

```text
02_data/private_processed/features/private_features_2d.csv
```

---

# 11. STEP 6 — Audit Feature 2D

Script harus print:

```text
Total 6-class captures
2D valid
2D invalid
NaN rate per feature
count per class
count per subject
```

Simpan:

```text
02_data/private_processed/audit/feature_2d_audit.csv
```

Acceptance:

```text
[ ] tidak ada forward_head
[ ] tidak ada reject
[ ] class_id hanya 0..5
[ ] tidak ada inf / -inf
[ ] normalized coordinate masuk akal
[ ] missing value hanya berasal dari keypoint invalid
[ ] tidak ada capture_id duplicate
```

---

# 12. STEP 7 — Ekstraksi Feature 3D

Buat:

```text
04_scripts/preprocessing/extract_private_3d_features.py
```

Input:

```text
private_6class_all.csv
02_data/private_annotations/keypoints_3d/
07_results/private_audit/private_3d_qc_final.csv
```

## 12.1 QC Rule

Gunakan:

```text
INCLUDE_3D_FULL
INCLUDE_3D_WITH_MASKING
```

Exclude:

```text
EXCLUDE_3D
```

Jangan memasukkan sample yang memang sudah ditandai unusable oleh QC stereo final.

## 12.2 Core 3D Points

```text
nose
left_shoulder
right_shoulder
left_hip
right_hip
```

## 12.3 Center dan Scale

```text
C3 = (left_hip + right_hip) / 2
S3 = max Euclidean distance dari hip_center ke core landmark valid
P3_normalized = (P3 - C3) / S3
```

Jangan menggunakan absolute camera distance sebagai feature utama.

## 12.4 Coordinate Convention Check

Sebelum feature generation, dokumentasikan:

```text
X = lateral
Y = vertical
Z = depth
origin/reference = ...
```

Lakukan sanity check `leaning_left`, `leaning_right`, `leaning_forward`, dan `leaning_backward` pada beberapa rig.

## 12.5 Feature 3D — 25 total

### Normalized coordinates — 15

```text
nose_x nose_y nose_z
left_shoulder_x left_shoulder_y left_shoulder_z
right_shoulder_x right_shoulder_y right_shoulder_z
left_hip_x left_hip_y left_hip_z
right_hip_x right_hip_y right_hip_z
```

### Spatial geometry — 10

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

Vector utama:

```text
shoulder_center = mean(left_shoulder, right_shoulder)
hip_center      = mean(left_hip, right_hip)
torso_vector    = shoulder_center - hip_center
head_vector     = nose - shoulder_center
```

Lateral lean: projection torso ke plane `(X,Y)` terhadap vertical axis.  
Sagittal lean: projection torso ke plane `(Z,Y)` terhadap vertical axis.  
3D inclination: angle antara `torso_vector` dan vertical 3D axis.

Depth asymmetry:

```text
shoulder_depth_asymmetry =
    (right_shoulder.z - left_shoulder.z) / S3

hip_depth_asymmetry =
    (right_hip.z - left_hip.z) / S3
```

## 12.6 Jangan Gunakan

Jangan gunakan feature bernama:

```text
cobb_angle
cobb_angle_approx
scoliosis_angle
```

Gunakan istilah:

```text
torso inclination
lateral lean
sagittal lean
shoulder tilt
pelvis tilt
spatial joint geometry
```

## 12.7 Missing 3D

```text
invalid joint → NaN
```

Tidak boleh:

```text
invalid joint → (0, 0, 0)
```

Output:

```text
02_data/private_processed/features/private_features_3d.csv
```

---

# 13. STEP 8 — Build Manifest 2D / 3D / Intersection

Buat:

```text
04_scripts/preprocessing/build_private_intersection.py
```

Output:

```text
private_6class_2d.csv
private_6class_3d.csv
private_6class_intersection.csv
```

### 2D manifest

Capture 6-class yang feature 2D-nya usable.

### 3D manifest

Capture 6-class yang QC 3D-nya FULL atau MASKING dan core 3D feature bisa dibentuk.

### Intersection

```text
capture_id terdapat pada 2D usable
AND
capture_id terdapat pada 3D usable
```

Inilah dataset **primary comparison**.

Print:

```text
6-class raw total
2D usable
3D usable
intersection usable
```

Lalu distribusi class dan subject.

Simpan audit:

```text
02_data/private_processed/audit/intersection_audit.csv
```

---

# 14. STEP 9 — Subject-Aware 5-Fold

Buat:

```text
04_scripts/evaluation/create_private_subject_folds.py
```

Gunakan:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Input utama:

```text
private_6class_intersection.csv
```

Group:

```text
subject_id
```

Target:

```text
class_id
```

Output:

```text
03_metadata/private_final_split/private_stratified_group_5fold.csv
```

Minimal columns:

```text
capture_id
subject_id
class_id
label
fold_id
```

Assertion anti-leakage:

```text
train_subjects ∩ test_subjects == empty
```

Jangan gunakan random split per image.

---

# 15. STEP 10 — Training XGBoost 2D

Buat:

```text
04_scripts/training/train_private_xgboost_2d.py
```

Input:

```text
private_features_2d_intersection.csv
private_stratified_group_5fold.csv
```

Jangan masukkan metadata sebagai feature:

```text
capture_id
subject_id
label
class_id
session_id
calibration_id
lateral_side
subset
```

Pipeline outer fold:

```text
TRAIN SUBJECTS
      │
      ├── fit scaler hanya pada training fold
      ├── hyperparameter tuning hanya pada training fold
      └── fit XGBoost
             │
             ▼
       TEST SUBJECTS
```

XGBoost:

```python
objective = "multi:softprob"
num_class = 6
eval_metric = "mlogloss"
random_state = 42
tree_method = "hist"
```

Jika XGBoost GPU pada environment saat ini stabil, `device="cuda"` boleh digunakan dan harus dicatat di log eksperimen.

Hyperparameter search awal:

```text
n_estimators        100, 150, 200, 300
max_depth           3, 5, 7, 9
learning_rate       0.03, 0.05, 0.07, 0.10
subsample           0.6, 0.8, 1.0
colsample_bytree    0.6, 0.8, 1.0
min_child_weight    1, 3, 5
gamma               0, 1, 5
reg_alpha           0, 0.001, 0.01, 0.1
reg_lambda          0.01, 0.1, 1.0
```

Practical search:

```text
RandomizedSearchCV
n_iter ≈ 25–40
inner grouped CV ≈ 3 folds
scoring = f1_macro
```

Inner CV juga menggunakan `subject_id` sebagai groups.

---

# 16. STEP 11 — Training XGBoost 3D

Buat:

```text
04_scripts/training/train_private_xgboost_3d.py
```

Protocol harus sama dengan 2D:

```text
same capture IDs
same labels
same fold assignment
same class order
same XGBoost search protocol
same scoring
```

Perbedaan utama hanya feature representation:

```text
X_2D = 36 features
X_3D = 25 features
```

---

# 17. STEP 12 — Output Evaluasi Per Fold

Simpan untuk masing-masing model:

```text
fold_metrics.csv
oof_predictions.csv
best_params_per_fold.json
classification_report.txt
confusion_matrix.png
```

OOF predictions minimal:

```text
capture_id
subject_id
fold_id
y_true
y_pred
prob_upright
prob_leaning_forward
prob_leaning_backward
prob_leaning_left
prob_leaning_right
prob_slouching
```

---

# 18. STEP 13 — Metric Utama

Laporkan:

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Per-Class Precision
Per-Class Recall
Per-Class F1
Confusion Matrix
Mean ± Standard Deviation across 5 folds
```

Primary metric:

```text
Macro F1
```

---

# 19. STEP 14 — Compare 2D vs 3D

Buat:

```text
04_scripts/evaluation/evaluate_private_2d_vs_3d.py
```

Script membaca OOF predictions dari dataset intersection yang sama.

Output:

```text
07_results/experiments/private_final/comparison/
```

Minimal:

```text
comparison_metrics.csv
comparison_metrics.md
confusion_matrix_2d.png
confusion_matrix_3d.png
per_class_comparison.csv
paired_capture_predictions.csv
```

Tabel utama:

```text
Metric              2D             3D
Accuracy
Macro Precision
Macro Recall
Macro F1
```

Tambahkan coverage sekunder:

```text
2D usable / 727
3D usable / 727
intersection / 727
```

Jangan mencampur coverage dengan accuracy.

---

# 20. STEP 15 — Interpretasi Fair

Jangan menganggap 3D harus lebih baik.

Analisis:

```text
kelas mana paling diuntungkan 3D?
kelas mana cukup dengan 2D?
apakah leaning_forward/backward mendapat manfaat depth?
apakah leaning_left/right sudah sangat baik pada frontal 2D?
apakah slouching masih tertukar dengan leaning_forward?
```

---

# 21. STEP 16 — Training Model Deployment

Evaluasi ilmiah menggunakan grouped CV dan OOF predictions.

Setelah evaluasi disimpan, buat:

```text
04_scripts/training/fit_private_deployment_models.py
```

Simpan:

```text
06_models/keypoint_2d/private_final/xgboost_2d_deployment.pkl
06_models/keypoint_3d/private_final/xgboost_3d_deployment.pkl
```

Simpan juga:

```text
feature_schema.json
class_map.json
scaler.pkl
model_metadata.json
```

Contoh metadata:

```json
{
  "model": "XGBoost",
  "representation": "2D_multi_view",
  "classes": [
    "upright",
    "leaning_forward",
    "leaning_backward",
    "leaning_left",
    "leaning_right",
    "slouching"
  ],
  "pose_estimator": "yolov8n-pose.pt",
  "n_features": 36
}
```

Untuk 3D, `n_features = 25`.

---

# 22. STEP 17 — Uji Model pada Capture Dataset

Buat:

```text
04_scripts/evaluation/test_private_single_capture.py
```

Contoh:

```bash
python 04_scripts/evaluation/test_private_single_capture.py \
    --capture-id <CAPTURE_ID> \
    --mode 2d
```

Output:

```text
Capture ID   : ...
Subject      : ...
Ground Truth : leaning_forward
Prediction   : leaning_forward
Confidence   : 0.93
Status       : CORRECT
```

Mode:

```text
--mode 2d
--mode 3d
```

Untuk demonstration yang tidak bias, gunakan capture yang prediksinya berasal dari OOF model saat membahas hasil evaluasi.

---

# 23. STEP 18 — Batch Functional Test

Tambahkan mode seperti:

```text
--subject S024
```

Output contoh:

```text
S024
--------------------------------
upright            ... / ...
leaning_forward    ... / ...
leaning_backward   ... / ...
leaning_left       ... / ...
leaning_right      ... / ...
slouching          ... / ...
```

---

# 24. STEP 19 — Inference 2D pada Pasangan Image Baru

Buat:

```text
04_scripts/inference/infer_private_pair_2d.py
```

Input:

```text
--cam01 path/to/front.jpg
--cam02 path/to/side.jpg
--lateral-side right
```

Pipeline:

```text
image pair
↓
YOLOv8-Pose
↓
target person selection
↓
core keypoints
↓
CAM02 canonicalization
↓
2D normalization
↓
36 features
↓
deployment scaler
↓
XGBoost
↓
posture + probability
```

Output:

```text
Prediction : leaning_left
Confidence : 0.914
```

---

# 25. STEP 20 — Inference Stereo 3D pada Pasangan Baru

Buat:

```text
04_scripts/inference/infer_private_pair_3d.py
```

Input:

```text
--cam01 front.jpg
--cam02 side.jpg
--calibration CAL_009
```

Pipeline:

```text
CAM01 + CAM02
↓
YOLOv8-Pose
↓
target person
↓
keypoint correspondence
↓
triangulation
↓
reprojection / core QC
↓
3D feature extraction
↓
XGBoost
```

Jika QC gagal:

```text
Prediction : REJECT / INVALID_3D
```

Jangan paksa classifier memberi salah satu dari enam kelas.

---

# 26. STEP 21 — Reject / Invalid-Input Gate

XGBoost hanya mengenali enam kelas.

Sebelum classifier:

```text
person detected?
↓ yes
target person valid?
↓ yes
shoulders valid?
↓ yes
hips valid?
↓ yes
pose normalization valid?
↓ yes
feature vector usable?
↓ yes
XGBoost
```

Jika gagal:

```text
REJECT
```

Untuk 3D tambahkan:

```text
triangulation core valid?
reprojection acceptable?
```

Gunakan threshold yang sudah dipakai pada QC repository. Jangan menciptakan threshold baru tanpa dokumentasi.

---

# 27. STEP 22 — Real-Time 2D Prototype

Buat:

```text
04_scripts/inference/infer_realtime_2d.py
```

Gunakan dua kamera bila target sistem adalah multi-view 2D.

Overlay minimal:

```text
POSTURE    : upright
CONFIDENCE : 94.2%
FPS        : 23.8
MODE       : 2D XGBoost
```

Functional test manual:

```text
upright
leaning_left
leaning_right
leaning_forward
leaning_backward
slouching
```

---

# 28. STEP 23 — Real-Time Stereo 3D Prototype

Buat:

```text
04_scripts/inference/infer_realtime_stereo_3d.py
```

Syarat:

```text
dual camera
kamera tersinkronisasi
calibration ID valid
setup fisik sama dengan calibration
```

Overlay:

```text
POSTURE    : leaning_backward
CONFIDENCE : 88.7%
FPS        : ...
MODE       : Stereo 3D XGBoost
3D QC      : PASS
```

---

# 29. Functional Demo ≠ Scientific Evaluation

## Scientific Evaluation

```text
grouped 5-fold
unseen subjects
OOF predictions
Macro F1
confusion matrix
```

Ini yang masuk hasil utama penelitian.

## Functional / Prototype Test

```text
coba duduk di depan kamera
ubah posture
lihat output
cek latency/FPS
```

Ini membuktikan pipeline dapat digunakan secara praktis. Laporkan terpisah.

---

# 30. STEP 24 — Update Runner

Jangan langsung mengubah `run_all_experiments.py` sebelum semua script final lolos satu per satu.

Setelah stabil, lebih aman membuat:

```text
04_scripts/run_private_final_pipeline.py
```

Urutan:

```text
1. build 6-class manifest
2. extract 2D features
3. extract 3D features
4. build intersection
5. create group folds
6. train/evaluate 2D
7. train/evaluate 3D
8. compare 2D vs 3D
9. fit deployment models
```

---

# 31. Urutan Eksekusi Final di IDE

```text
STEP 1  build_private_6class_manifest.py
STEP 2  extract_private_2d_features.py
STEP 3  extract_private_3d_features.py
STEP 4  build_private_intersection.py
STEP 5  create_private_subject_folds.py
STEP 6  train_private_xgboost_2d.py
STEP 7  train_private_xgboost_3d.py
STEP 8  evaluate_private_2d_vs_3d.py
STEP 9  fit_private_deployment_models.py
STEP 10 test_private_single_capture.py
STEP 11 infer_private_pair_2d.py
STEP 12 infer_private_pair_3d.py
STEP 13 infer_realtime_2d.py
STEP 14 infer_realtime_stereo_3d.py
```

Jangan lompat ke STEP 6 sebelum audit STEP 1–5 selesai.

---

# 32. Stop Conditions

STOP dan jangan training jika:

```text
[ ] total 6-class != 727
[ ] forward_head masih masuk target
[ ] reject masih masuk target XGBoost
[ ] capture_id duplicate
[ ] subject leakage antar fold
[ ] CAM01/CAM02 mismatch
[ ] coordinate convention 3D tidak konsisten
[ ] banyak inf / -inf pada features
[ ] feature schema train != inference
[ ] class_map berbeda antar script
```

---

# 33. Reusable Feature Functions

Buat:

```text
04_scripts/preprocessing/private_feature_common.py
```

Berisi fungsi reusable:

```text
normalize_pose_2d()
canonicalize_lateral_view()
extract_2d_feature_vector()
normalize_pose_3d()
extract_3d_feature_vector()
angle_between_vectors()
validate_core_keypoints()
```

Training dan inference harus import fungsi yang sama agar tidak terjadi training-serving skew.

---

# 34. Model Artifact Wajib

Untuk setiap deployment model:

```text
model.pkl
scaler.pkl
feature_schema.json
class_map.json
model_metadata.json
```

Untuk 3D tambahkan:

```text
coordinate_convention.json
```

Calibration tetap dibaca dari:

```text
02_data/private_calibration/
```

---

# 35. Expected Final Research Outputs

```text
07_results/experiments/private_final/
├── 2d/
│   ├── fold_metrics.csv
│   ├── oof_predictions.csv
│   ├── summary_metrics.json
│   └── confusion_matrix.png
├── 3d/
│   ├── fold_metrics.csv
│   ├── oof_predictions.csv
│   ├── summary_metrics.json
│   └── confusion_matrix.png
└── comparison/
    ├── comparison_metrics.csv
    ├── comparison_metrics.md
    ├── per_class_comparison.csv
    └── paired_capture_predictions.csv
```

---

# 36. Minimum Final Tables untuk Skripsi

## Tabel Dataset Final

```text
Class               Raw 6-Class   2D Usable   3D Usable   Intersection
upright
leaning_forward
leaning_backward
leaning_left
leaning_right
slouching
TOTAL
```

## Tabel 5-Fold 2D / 3D

```text
Fold   Accuracy   Macro Precision   Macro Recall   Macro F1
1
2
3
4
5
Mean
STD
```

## Tabel Final Comparison

```text
Metric              XGBoost 2D      XGBoost 3D
Accuracy
Macro Precision
Macro Recall
Macro F1
```

## Tabel Per-Class F1

```text
Class               2D F1           3D F1
upright
leaning_forward
leaning_backward
leaning_left
leaning_right
slouching
```

---

# 37. Failure Analysis

Untuk sample salah klasifikasi, simpan:

```text
capture_id
subject_id
true_label
predicted_label
confidence
2D/3D QC status
calibration_id
```

Kategori yang perlu dianalisis:

```text
leaning_forward ↔ slouching
upright ↔ slight lean
leaning_left ↔ upright
leaning_right ↔ upright
forward ↔ backward
pose estimator failure
stereo reconstruction failure
```

Jangan mengubah test label setelah melihat prediction tanpa mencatat perubahan label secara formal.

---

# 38. Keputusan `forward_head`

Tidak ikut:

```text
training 6-class
5-fold evaluation
2D-vs-3D comparison
confusion matrix utama
```

Boleh disimpan untuk future work / out-of-taxonomy test.

---

# 39. Keputusan `reject`

Raw reject tetap disimpan dan boleh digunakan untuk functional gate testing, tetapi:

```text
num_class = 6
```

---

# 40. Definition of Done

```text
[ ] 24 subjects final
[ ] 727 raw 6-class captures terverifikasi
[ ] forward_head excluded
[ ] reject excluded dari classifier
[ ] 2D feature table selesai
[ ] 3D feature table selesai
[ ] intersection dataset selesai
[ ] subject-aware 5-fold selesai
[ ] zero subject overlap
[ ] XGBoost 2D selesai
[ ] XGBoost 3D selesai
[ ] OOF predictions tersedia
[ ] confusion matrix tersedia
[ ] Macro F1 mean ± SD tersedia
[ ] per-class metrics tersedia
[ ] 2D-vs-3D comparison tersedia
[ ] deployment 2D model tersedia
[ ] deployment 3D model tersedia
[ ] single-capture inference berhasil
[ ] new-pair inference berhasil
[ ] real-time 2D prototype berhasil
[ ] real-time stereo 3D prototype berhasil atau keterbatasannya terdokumentasi
[ ] latency/FPS tercatat
```

---

# 41. Prioritas Saat Membuka IDE Berikutnya

Kerjakan lima hal ini terlebih dahulu:

```text
1. Update taxonomy_kelas.md
2. Update dataset_decisions.md
3. Implement build_private_6class_manifest.py
4. Jalankan dan pastikan TOTAL = 727
5. Implement extract_private_2d_features.py
```

Setelah lima langkah ini lolos audit, lanjut ke 3D.

---

# 42. Rumusan Pipeline Final

Pipeline final bukan lagi:

```text
mencari algoritma classifier terbaik
```

melainkan:

```text
membandingkan representasi pose 2D multi-view dan stereo 3D
menggunakan classifier XGBoost yang sama,
pada enam kelas postur,
dengan evaluasi subject-independent.
```

Kontrol utama:

```text
classifier sama
class taxonomy sama
capture sama pada primary comparison
fold sama
subject split sama
evaluation metric sama
```

---

# 43. Pipeline Final Ringkas

```text
24 Subjects
    │
    ▼
885 Stereo Captures
    │
    ▼
6-Class Filtering
    │
    ├── forward_head → excluded
    └── reject       → invalid-input/QC gate
    │
    ▼
YOLOv8-Pose COCO-17
    │
    ├───────────────────────────────┐
    ▼                               ▼
2D Multi-View                   Stereo 3D
CAM01 + CAM02                   Triangulation
    │                               │
Center + Scale                  Center + Scale
Normalization                   Normalization
    │                               │
2D Geometry                     Spatial Geometry
36 Features                     25 Features
    │                               │
XGBoost                         XGBoost
    │                               │
    └───────────────┬───────────────┘
                    ▼
            Same Capture IDs
                    ▼
       Subject-Aware Grouped 5-Fold
                    ▼
     Accuracy / Precision / Recall
     Macro F1 / Per-Class / CM
                    ▼
              2D vs 3D
                    ▼
         Deployment Models
                    ▼
    Offline + Real-Time Inference
```

---

**Dokumen ini menjadi panduan implementasi teknis utama untuk fase final repository `mitigasi_skoliosis_ta`.**
