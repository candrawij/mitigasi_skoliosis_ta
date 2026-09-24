# Outline Lengkap — Skripsi / Artikel Ilmiah
## Sistem Deteksi Postur Duduk Multi-Modalitas Berbasis Multi-View 2D Keypoint dan Stereo 3D Reconstruction Menggunakan XGBoost

> **Versi:** Final — Berdasarkan eksperimen frozen `v1.0-freeze-final-four-model` (2026-09-24)  
> **Tipe:** Outline lengkap untuk skripsi (5–6 bab) atau artikel jurnal (IMRaD)

---

## BAB I — PENDAHULUAN

### 1.1 Latar Belakang
- Prevalensi masalah muskuloskeletal akibat postur duduk buruk
  - Skoliosis, lordosis, nyeri punggung bawah kronis
  - Populasi rentan: pelajar dan pekerja kantoran (6–10 jam/hari)
  - Data epidemiologis: prevalensi dan dampak ekonomi (referensi WHO/jurnal)
- Keterbatasan solusi yang ada saat ini
  - Sensor wearable: tidak nyaman, kalibrasi individual, high dropout
  - Pressure mat: mahal, hanya berat badan, tidak menangkap postur atas
  - Kamera tunggal: ambigu pada kondisi postur lateral vs frontal
  - Dataset publik: kondisi lab tidak merepresentasikan variasi rig nyata
- Peluang: sistem stereo dual-kamera sudah tersedia, belum dioptimalkan untuk monitoring postur

### 1.2 Rumusan Masalah
1. Apakah representasi stereo 3D memberikan peningkatan akurasi yang konsisten dibandingkan multi-view 2D untuk klasifikasi postur duduk pada dataset multi-subjek multi-rig?
2. Apakah *body-aligned canonicalization* mampu mengatasi domain shift akibat heterogenitas rig kalibrasi tanpa mengorbankan akurasi klasifikasi?
3. Sejauh mana fitur sagital telinga (*ear features*) berkontribusi terhadap peningkatan deteksi postur *slouching* dan *leaning forward*?

### 1.3 Tujuan Penelitian
1. Merancang dan mengimplementasikan pipeline klasifikasi postur duduk berbasis stereo dual-kamera yang berjalan end-to-end
2. Mengumpulkan dataset privat multi-subjek (24 subjek, 6 kelas postur, multi-rig stereo)
3. Membandingkan secara kuantitatif 4 sistem representasi pose pada dataset yang sama dan protokol evaluasi yang identik
4. Menganalisis penyebab kegagalan model dan memberikan implikasi untuk desain sistem yang lebih robust

### 1.4 Kontribusi Penelitian
1. **Dataset privat terstandarisasi** — 24 subjek, 727 tangkapan, 6 rig stereo, 6 kelas postur, dengan protokol QC berlapis
2. **Protokol evaluasi *subject-aware*** — Subject-Aware 5-Fold CV dengan zero subject leakage
3. **Analisis komparatif 4 representasi** — 2D Core, 2D+Ear, 3D Camera, 3D Canonical pada data identik
4. **Investigasi kuantitatif Fold 3 Collapse** — rig-specific domain shift (CAL_004) sebagai sumber kegagalan model 3D
5. **Temuan canonicalization** — bukti bahwa sinyal diskriminatif 3D Camera Frame bukan dari geometri pose murni
6. **Bukti deployment CPU** — pipeline end-to-end 191.5ms (5.2 FPS); bottleneck terisolasi ke YOLO

### 1.5 Batasan Penelitian
- Dataset privat (tidak tersedia publik); 24 subjek
- Kondisi kamera statis, pencahayaan terkontrol
- Tidak ada validasi eksternal di luar 24 subjek
- Coverage 3D terbatas: 55.4% (403/727) akibat right_hip NaN

### 1.6 Sistematika Penulisan
*(deskripsi singkat isi tiap bab)*

---

## BAB II — TINJAUAN PUSTAKA

### 2.1 Postur Duduk dan Masalah Tulang Belakang
- Definisi postur normal dan deviasi klinis
- Mekanisme injury akibat postur buruk berkepanjangan
- Sistem penilaian postur yang ada (RULA, REBA, klinisi)

### 2.2 Human Pose Estimation
- **2D HPE:** OpenPose, HRNet, ViTPose; representasi heatmap vs keypoint regression
- **3D HPE:** Pendekatan monocular (lifting from 2D) vs multi-view; keterbatasan depth ambiguity
- **YOLO-Pose (YOLOv8n-Pose):** Arsitektur, keypoint head, kecepatan vs akurasi tradeoff
- Pilihan YOLOv8n-Pose untuk penelitian ini: real-time capable, single-stage, tidak butuh GPU training

### 2.3 Stereo Vision untuk Rekonstruksi 3D
- Kalibrasi kamera: model pinhole, distorsi radial/tangensial
- Kalibrasi stereo: rotasi R dan translasi T ekstrinsik
- Triangulasi DLT: prinsip matematis, sensitivitas terhadap reprojection error
- Keterbatasan triangulasi pasif: oklusi, korespondensi keypoint, baseline trade-off

### 2.4 Klasifikasi Postur Duduk — State of the Art
- Pendekatan berbasis sensor vs berbasis visi
- Pendekatan berbasis fitur geometri (keypoint-based): SVM, Random Forest, XGBoost
- Pendekatan deep learning (CNN, ViT): data-hungry, kurang interpretable
- Dataset yang ada: SitRight, IntelliCare, dataset akademis lainnya; keterbatasan

### 2.5 XGBoost untuk Klasifikasi Pose
- Prinsip gradient boosting; `multi:softprob` untuk multi-class
- Keunggulan untuk data tabular keypoint: interpretable, tidak butuh GPU, robust terhadap fitur redundan
- Penggunaan XGBoost pada pose recognition: literatur terkait

### 2.6 Evaluasi Model yang Adil pada Dataset Kecil
- Masalah *subject leakage* dalam klasifikasi biometrik
- Subject-Aware Group K-Fold: motivasi dan implementasi
- Macro F1 vs Accuracy pada dataset imbalanced multi-class
- Perbedaan *pooled OOF F1* vs *fold mean F1*: kapan masing-masing digunakan

### 2.7 Domain Adaptation dan Canonicalization Pose 3D
- Body-aligned canonical frame: konsep dan formulasi (Gram-Schmidt orthogonalization)
- Penggunaan canonical frame dalam pose recognition dan motion capture
- Potensi dan limitasi pendekatan geometric canonicalization

---

## BAB III — METODOLOGI

### 3.1 Desain Sistem Keseluruhan
- Diagram arsitektur sistem (flowchart end-to-end)
- Dua jalur paralel: jalur 2D dan jalur 3D
- Prinsip desain: interpretable, CPU-feasible, modular

### 3.2 Pengumpulan Dataset Privat

#### 3.2.1 Desain Eksperimental
- Definisi operasional 6 kelas postur
- Kriteria subjek inklusif/ekslusif
- Standarisasi kursi, jarak kamera, instruksi kepada subjek

#### 3.2.2 Sistem Kamera Stereo
- Hardware: kamera, mounting bracket, sinkronisasi
- Kalibrasi intrinsik (metode checkerboard, OpenCV)
- Kalibrasi ekstrinsik stereo (11 rig; CAL_001–CAL_011)
- Validasi kalibrasi: reprojection error < 1.5 px (target)

#### 3.2.3 Statistik Dataset
- Total: 24 subjek, 727 tangkapan
- Distribusi per kelas postur
- Distribusi per rig kalibrasi

### 3.3 Quality Control Pipeline

#### 3.3.1 Layer 1 — Stereo-QC
- YOLO detection check (kedua kamera)
- Bounding box overlap / identity verification
- Reprojection error threshold (50px)
- Hasil: 657/727 (90.4%) Stereo-QC Usable

#### 3.3.2 Layer 2 — Feature-QC (3D Model-Ready)
- Right_hip mandatory check
- Alasan right_hip mandatory: normalisasi panggul dan vektor torso
- Alasan tidak imputed: zero imputation merusak topologi koordinat 3D
- Hasil: 403/727 (55.4%) Model-Ready 3D; 254 gugur (right_hip NaN)

### 3.4 Deteksi Pose — YOLOv8n-Pose
- Konfigurasi inferensi: resolusi, confidence threshold, NMS
- Handling multi-person detection (ambil bounding box terbesar)
- Keypoint filtering: visibility threshold

### 3.5 Feature Engineering

#### 3.5.1 Representasi 2D Core (36 fitur)
- Normalisasi koordinat terhadap bounding box torso
- Daftar fitur lengkap per kamera (CAM01 + CAM02)
- Geometri turunan: slope, inclination, offset

#### 3.5.2 Representasi 2D Best (42 fitur)
- Ear sagittal features: ear_shoulder_horizontal/vertical_norm, ear_neck_angle_deg
- Motivasi klinis: forward head posture, slouching
- Jumlah total: 36 + 6 = 42 fitur

#### 3.5.3 Representasi 3D Camera Frame (25 fitur)
- Triangulasi DLT: persamaan matematis
- Normalisasi: pelvis-centered, scale=max_distance
- Daftar 25 fitur: koordinat + geometri spasial
- NaN handling: SimpleImputer (median) untuk fitur nose dan head

#### 3.5.4 Representasi 3D Canonical Body Frame (22 fitur)
- Formulasi canonical frame (matematis)
  - Origin = hip_center
  - Y_can = torso direction (Gram-Schmidt)
  - X_can = hip direction (orthogonalized)
  - Z_can = X_can × Y_can
- 3 fitur degenerate yang dihapus dan alasannya
- Skema 22 fitur final
- Verifikasi: tabel shoulder_roll_deg sebelum dan sesudah canonicalization per rig

### 3.6 Protokol Evaluasi

#### 3.6.1 Dataset dan Fold Split
- 403 sampel (irisan 2D ∩ 3D)
- private_2d3d_intersection_5fold.csv (identik untuk semua 4 model)
- Subject-Aware Stratified 5-Fold CV
- Distribusi subjek per fold

#### 3.6.2 Classifier dan Hyperparameter
- XGBoost: multi:softprob, 6 kelas
- Hyperparameter: n_estimators=300, max_depth=6, lr=0.1, subsample=0.8, seed=42
- Preprocessing: SimpleImputer (median)
- Tidak ada tuning per model (hyperparameter identik)

#### 3.6.3 Metrik Evaluasi
- **Primer:** Pooled OOF Macro F1
- **Sekunder:** Fold Mean Macro F1 ± Std (stabilitas), per-class F1
- Penjelasan perbedaan pooled F1 vs fold mean F1

---

## BAB IV — HASIL DAN PEMBAHASAN

### 4.1 Perbandingan Global 4 Model (FROZEN Results)

Tabel perbandingan:

| Model | Fitur | Pooled Acc | Pooled Macro F1 | Fold Mean F1 | Fold Std |
|---|:---:|:---:|:---:|:---:|:---:|
| 2D Core Baseline | 36f | 67.99% | 0.6863 | 0.6669 | 0.0846 |
| **2D Best Practical** | **42f** | **69.73%** | **0.7039** | **0.6875** | **0.0701** |
| 3D Current / Camera | 25f | 61.79% | 0.6182 | 0.6046 | 0.2519 |
| 3D Canonical / Body | 22f | 33.00% | 0.3259 | 0.3237 | 0.1241 |

**Pembahasan:**
- 2D Best mengungguli 3D Current sebesar 0.0857 Macro F1 (+13.9% relatif)
- Namun standar deviasi fold 3D Current sangat tinggi (0.2519 vs 0.0701) — bukan karena 3D selalu lebih buruk, melainkan karena 1 fold collapsed
- Perbandingan lebih murni modalitas: 2D Core (36f) vs 3D Current (25f) = 0.6863 vs 0.6182 (+0.0681)

### 4.2 Analisis Per-Kelas

Tabel lengkap 6 kelas × 4 model (F1, Precision, Recall).

**Pola kunci:**
- **leaning_left / leaning_right:** 2D sangat dominan (F1 0.98 vs 0.69) — lateral slope lebih terlihat di 2D
- **leaning_forward / leaning_backward:** 3D sedikit unggul — kedalaman sagital memberikan sinyal tambahan
- **slouching:** fitur ear krusial — 2D Core 0.50 → 2D Best 0.59 (+8.7%)
- **upright:** semua model moderat (0.58–0.66) — kelas baseline, sulit dibedakan dari condong ringan

### 4.3 Analisis Per-Fold dan Fold 3 Collapse

Tabel fold-by-fold (5 fold × 4 model + test subjects).

**Pembahasan:**
- Fold 0 & 4: 3D Current mengungguli 2D Best (0.7648 vs 0.7480; 0.7583 vs 0.5930) — potensi superior 3D pada rig homogen
- Fold 3: 3D Current 0.1052 (collapse) vs 2D Best 0.6187 (stabil)

**Investigasi Fold 3 — Bukti Empiris:**
- Distribusi prediksi: 72.2% sampel Fold 3 diprediksi sebagai leaning_right
- Subjek S003 & S004 (rig CAL_004): 100% salah diprediksi
- shoulder_roll_deg: Fold 3 = 126.48° ± 79.73° vs non-Fold3 = 31.99° ± 109.51°
- torso_sagittal_lean_deg: Fold 3 = −19.07° vs non-Fold3 = +8.07°
- Mekanisme: CAL_004 mengalami rotational shift pada matriks ekstrinsik; orientasi "tegak" terbaca sebagai >90° roll; model tidak melihat pola ini saat training

### 4.4 Analisis Canonicalization

**Berhasil:** Koreksi domain shift CAL_004 (shoulder_roll: 125° → 1°)  
**Gagal:** Akurasi turun dari 0.6182 → 0.3259 (−29.2%)

**Analisis distribusi fitur canonical:**
- Koordinat joint (15f) menjadi hampir tidak diskriminatif antar kelas (contoh: left_shoulder_z: mean 0.012–0.016 untuk semua kelas)
- head_depth_offset_norm: −0.165 s/d −0.194 (tumpang tindih antar kelas)
- Interpretasi: sinyal 3D Camera Frame berasal dari interaksi geometri pose × orientasi frame kamera — bukan dari pose tubuh murni

**Implikasi:** 3D Camera Frame adalah representasi hybrid; canonical frame menghilangkan "artefak" yang justru berguna

### 4.5 Kontribusi Fitur Telinga (Ear Features) — Analisis Terisolasi

**Perbandingan 2D Core vs 2D Best (fold split identik, hanya beda 6 fitur):**
- Macro F1: 0.6863 → 0.7039 (+0.0176, +2.6%)
- Stabilitas: Fold Std 0.0846 → 0.0701 (−16% lebih stabil)
- slouching: F1 0.5000 → 0.5872 (+8.7%)
- leaning_forward: F1 0.4783 → 0.5211 (+4.3%)
- leaning_left: F1 0.8376 → 0.8205 (−2.0%, minor regression)

**Kesimpulan:** Ear features signifikan untuk kelas sagital; tradeoff minor pada kelas lateral

### 4.6 Uji Deployment

**3D-15 (Raw Pair Test):**
- 1/6 benar; pipeline tanpa exception ✅
- Feature schema konsisten training vs inference ✅
- Reproj error tinggi (20–42px) → penyebab rendahnya akurasi, bukan code bug

**3D-16 (Latency Benchmark):**
- End-to-end: 191.5ms (5.2 FPS CPU)
- YOLO: 188ms (98%); triangulasi + fitur + XGBoost: 3.4ms (2%)
- Bottleneck: YOLO inference → solusi: GPU (TensorRT)

---

## BAB V — KESIMPULAN DAN SARAN

### 5.1 Kesimpulan
1. **2D Best Practical (42f) adalah sistem terbaik** pada kondisi ini: Macro F1 0.7039, Fold Std 0.0701
2. **3D Camera Frame kompetitif pada rig homogen** (Fold 0: 0.7648 > 2D Best 0.7480) tetapi sangat fragil pada rig heterogen (Fold 3: 0.1052)
3. **Canonicalization 3D belum berhasil** memisahkan domain shift dari sinyal postur — F1 turun 29.2%
4. **Ear features berkontribusi signifikan** terhadap kelas sagital (slouching, leaning_forward)
5. **Pipeline feasible di CPU** dengan 5.2 FPS; komponen 3D sendiri hanya 3.4ms (sangat efisien)
6. **Temuan kegagalan CAL_004** memberikan panduan konkret: model 3D membutuhkan mekanisme online calibration adaptation atau setidaknya representasi yang tidak sensitif terhadap orientasi frame kamera absolut

### 5.2 Saran Penelitian Lanjutan
1. **Adaptive calibration normalization** — normalisasi online berbasis orientasi kamera yang terdeteksi dari data
2. **Data-driven canonical frame** — PCA-based atau learned alignment daripada geometric Gram-Schmidt
3. **Dataset lebih besar** — menambah subjek untuk mengurangi variance fold dan meningkatkan F1 kelas minority
4. **Cross-dataset validation** — validasi pada subjek atau rig yang tidak pernah dilihat saat training
5. **GPU inference** — TensorRT/ONNX Runtime untuk YOLO: potensi meningkatkan ke 30+ FPS
6. **Sensor fusion** — kombinasi 2D keypoint + depth sensor dedicated (ToF camera) sebagai alternatif stereo pasif

---

## BAB VI — REFERENSI

*(Daftar referensi — perlu dilengkapi)*

**Wajib dikutip:**
- Jocher, G. et al. (2023). YOLOv8 / Ultralytics
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016
- OpenCV Documentation: Stereo Calibration and Triangulation
- COCO Keypoint Challenge — 17-keypoint skeleton definition
- Literature: Subject-Aware CV / Group K-Fold (scikit-learn documentation)
- Literature: Sitting posture monitoring (SitRight dataset, IntelliCare)
- Literature: Body-aligned canonical frame for 3D pose

---

## LAMPIRAN

### A. Daftar Fitur Lengkap per Model
*(reproduksi dari feature_schema.json)*

### B. Detail Fold Split
*(distribusi subjek, tangkapan, kelas per fold)*

### C. Confusion Matrix per Model
*(4 confusion matrix: 2D Core, 2D Best, 3D Current, 3D Canonical)*

### D. Kode Utama
- `04_scripts/analysis/final_four_model_comparison.py` — eksperimen final
- `04_scripts/preprocessing/canonicalize_private_3d_pose.py` — canonicalization
- `04_scripts/training/train_private_xgboost_3d_canonical.py` — training canonical

### E. Reproducibility
- Dataset: privat (tidak tersedia publik; tersedia atas permintaan)
- Fold split: `03_metadata/private_final_split/private_2d3d_intersection_5fold.csv`
- Seed: 42; XGBoost params terdokumentasi di `feature_schema.json`
- Git tag: `v1.0-freeze-final-four-model` (commit `a7f2821b`)

---

*Outline ini dibuat berdasarkan eksperimen frozen 2026-09-24.*  
*Angka mengacu pada `FINAL_FOUR_MODEL/final_four_model_comparison.csv` sebagai single source of truth.*
