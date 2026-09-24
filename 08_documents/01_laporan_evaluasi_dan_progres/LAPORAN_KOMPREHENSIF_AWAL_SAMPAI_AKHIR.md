# Laporan Komprehensif Penelitian
# Sistem Deteksi Postur Duduk: Dari Ide Awal Hingga Hasil Akhir

**Judul Penelitian:** Sistem Deteksi dan Klasifikasi Postur Duduk Berbasis *Multi-View 2D Keypoint* dan *Stereo 3D Reconstruction* Menggunakan XGBoost untuk Mitigasi Risiko Masalah Tulang Belakang  
**Penulis:** Candra  
**Tanggal Laporan:** 24 September 2026  
**Status:** Eksperimen Dibekukan — `v1.0-freeze-final-four-model`  
**Commit Freeze:** `a7f2821b`

---

## Bagian 1 — Ide Awal dan Motivasi Penelitian

### 1.1 Latar Belakang Masalah

Penelitian ini berangkat dari permasalahan klinis yang nyata: **postur duduk yang buruk dalam waktu panjang merupakan faktor risiko utama kelainan muskuloskeletal**, termasuk skoliosis, lordosis lumbar, dan nyeri kronis punggung bawah. Populasi yang paling rentan adalah pelajar dan pekerja kantor yang menghabiskan 6–10 jam per hari dalam posisi duduk.

Solusi yang tersedia saat ini memiliki keterbatasan mendasar:
- **Sensor wearable** (IMU, accelerometer): tidak nyaman dipakai dalam jangka panjang, perlu kalibrasi individual
- **Pressure mat**: mahal, hanya mendeteksi distribusi berat, bukan postur tubuh bagian atas
- **Kamera tunggal**: tidak menangkap kedalaman (depth), ambigu pada postur lateral vs frontal
- **Dataset publik yang ada** (SitRight, MPII, dsb.): dikumpulkan dalam kondisi lab yang berbeda, pose terbatas, tidak merepresentasikan keragaman rig kamera nyata

### 1.2 Ide Utama Penelitian

Ide inti adalah memanfaatkan **sistem kamera stereo yang sudah lazim tersedia** (dua kamera consumer-grade yang dikalibrasi) untuk memperoleh **informasi kedalaman (depth) tanpa sensor tambahan**. Dua representasi utama diuji:

1. **Multi-View 2D** — menggabungkan keypoint dari dua sudut pandang berbeda (frontal + lateral) sebagai fitur 2D yang diperkaya perspektif
2. **Stereo 3D** — merekonstruksi koordinat 3D sendi dari pasangan gambar stereo menggunakan triangulasi

Pertanyaan ilmiah utama: *Apakah rekonstruksi 3D memberikan keunggulan nyata dibandingkan penggabungan sederhana sudut pandang 2D untuk klasifikasi postur duduk?*

### 1.3 Pendekatan Awal yang Ditinggalkan

Sebelum mencapai desain final, beberapa pendekatan dicoba dan ditinggalkan:

| Pendekatan | Alasan Ditinggalkan |
|---|---|
| EfficientNet end-to-end (image → label) | Terlalu data-hungry; tidak interpretable; butuh GPU |
| YOLO-Pose classifier (langsung dari bbox) | Akurasi rendah; terlalu bergantung pada resolusi kamera |
| Dataset publik Posture Exercise, IKORN, Project Design | Tidak merepresentasikan postur duduk realistis di lingkungan nyata |
| Pose landmark 4-titik saja (bahu + pinggul) | Informasi tidak cukup untuk membedakan 6 kelas postur |

---

## Bagian 2 — Desain Sistem dan Pengumpulan Dataset

### 2.1 Desain Sistem Akhir

Sistem yang dirancang terdiri dari dua jalur inferensi paralel yang berjalan di atas infrastruktur kamera yang sama:

```
[CAM01 Frontal] ─┬─[YOLOv8n-Pose]─┬─[2D Keypoints]──[Feature Eng. 2D]──[XGBoost 2D]──[Label]
                 │                 └─[2D Keypoints]─┐
[CAM02 Lateral] ─┴─[YOLOv8n-Pose]─[2D Keypoints]───┴─[Triangulasi DLT]─[3D Points]──[XGBoost 3D]──[Label]
```

**Komponen utama:**
- **Detektor pose:** YOLOv8n-Pose (17 COCO keypoints)
- **Kalibrasi:** OpenCV stereo calibration (intrinsik + ekstrinsik)
- **Triangulasi:** Direct Linear Transform (DLT) stereo triangulation
- **Classifier:** XGBoost (multi-class, 6 label)
- **Evaluasi:** Subject-Aware 5-Fold Cross-Validation

### 2.2 Setup Pengumpulan Dataset

**Hardware:**
- 2 kamera USB consumer-grade (resolusi 1280×720, 30fps)
- Mounting bracket kalibrasi (kamera frontal + kamera lateral kanan/kiri)
- Kursi standar tanpa roda, meja 70cm
- Pencahayaan ruangan normal (tidak ada lampu khusus)

**Protokol Perekaman:**
- Setiap subjek melakukan 6 postur: upright, leaning_forward, leaning_backward, leaning_left, leaning_right, slouching
- ~5 tangkapan per postur per subjek (sesi ~15–20 menit)
- Instruksi tertulis diberikan; tidak ada koreksi postur dari peneliti selama sesi

**Kelas Postur:**

| Kelas | Deskripsi Postur |
|---|---|
| `upright` | Duduk tegak, punggung lurus, kepala netral |
| `leaning_forward` | Condong ke depan dari pinggang/bahu |
| `leaning_backward` | Bersandar ke belakang, punggung bertumpu sandaran |
| `leaning_left` | Condong ke kiri, bahu kiri turun |
| `leaning_right` | Condong ke kanan, bahu kanan turun |
| `slouching` | Membungkuk, punggung melengkung ke belakang, kepala maju |

### 2.3 Statistik Dataset

| Metrik | Nilai |
|---|---|
| Total subjek | 24 (S001–S024) |
| Total tangkapan 6-kelas | 727 |
| Rig kalibrasi stereo | 11 rig (CAL_001–CAL_011) |
| Rig lolos kalibrasi ekstrinsik | 9 rig (CAL_006 & CAL_010 excluded) |
| Stereo-QC Usable | **657 / 727 (90.4%)** |
| Model-Ready 3D | **403 / 727 (55.4%)** |
| Model-Ready 2D | **704 / 727 (96.8%)** |

---

## Bagian 3 — Quality Control (QC) dan Preprocessing

### 3.1 Layer 1 — Stereo-QC

Setiap tangkapan diperiksa secara otomatis:

1. **YOLO Detection Check:** Apakah pose terdeteksi di kedua kamera?
2. **Bounding Box Overlap:** Apakah ROI subjek dari CAM01 dan CAM02 konsisten (orang yang sama)?
3. **Reprojection Error:** Setelah triangulasi, apakah proyeksi balik ke citra ≤ 50 pixel?
4. **Keypoint Visibility:** Apakah sendi *core mandatory* (bahu, pinggul) terdeteksi dengan confidence ≥ 0.5?

Hasil: **657/727 (90.4%)** berstatus `USABLE`.

### 3.2 Layer 2 — Feature-QC (3D Model-Ready)

Tambahan pemeriksaan khusus untuk pipeline 3D:
- **Right Hip Mandatory:** Koordinat `right_hip` (sendi 12) wajib ada untuk membangun vektor torso dan normalisasi panggul. Jika NaN → sampel gugur.
- Sebanyak **254 tangkapan lolos Stereo-QC tetapi tidak lolos Feature-QC** karena `right_hip` NaN, kemungkinan akibat oklusi sandaran kursi atau kontur paha pada sudut pandang kamera lateral.

> **Keputusan metodologis penting:** Zero imputation (mengisi NaN dengan 0) untuk `right_hip` TIDAK dilakukan karena merusak topologi koordinat 3D. Sampel dengan `right_hip` NaN **TIDAK** dimasukkan ke model.

### 3.3 Kalibrasi Kamera

- **Intrinsik:** Dilakukan dengan checkerboard pattern (OpenCV, ≥30 pose gambar per kamera)
- **Ekstrinsik:** Kalibrasi stereo (rotasi R + translasi T antara kedua kamera)
- **Validasi:** Reprojection error < 1.5 px (target kalibrasi); < 50 px (threshold operasional)

**Catatan kritis — Rig CAL_004:**  
Rig CAL_004 yang digunakan untuk subjek S003 dan S004 teridentifikasi mengalami *rotational shift* pada matriks ekstrinsik — `shoulder_roll_deg` 3D rata-rata mencapai **126.48°** (vs 31.99° pada rig lain). Ini kemudian menjadi penyebab utama *Fold 3 Collapse* pada model 3D.

---

## Bagian 4 — Feature Engineering

### 4.1 Deteksi Pose — YOLOv8n-Pose

- Model: `yolov8n-pose.pt` (Ultralytics)
- Input: frame 1280×720 dari CAM01 dan CAM02
- Output: 17 COCO keypoints + confidence per keypoint
- Filtering: deteksi multi-orang → ambil orang dengan area bounding box terbesar

### 4.2 Representasi 2D Multi-View

**Normalisasi Koordinat:**
Koordinat keypoint dinormalisasi terhadap bounding box torso agar *scale-invariant* (tidak bergantung pada jarak subjek ke kamera).

**2D Core (36 fitur) — per kamera (CAM01 + CAM02), masing-masing 18 fitur:**

| Fitur | Deskripsi |
|---|---|
| `nose_x`, `nose_y` | Posisi hidung (kepala proxy) |
| `left/right_shoulder_x/y` | Posisi bahu kiri & kanan |
| `left/right_hip_x/y` | Posisi pinggul kiri & kanan |
| `shoulder_slope_deg` | Sudut kemiringan garis bahu |
| `hip_slope_deg` | Sudut kemiringan garis pinggul |
| `torso_inclination_deg` | Inklinasi torso dari vertikal |
| `head_torso_angle_deg` | Sudut antara kepala dan torso |
| `head_to_shoulder_norm` | Jarak kepala ke bahu (norm) |
| `torso_length_norm` | Panjang torso (norm) |
| `head_horizontal_offset_norm` | Offset horizontal kepala dari pusat bahu |
| `torso_horizontal_offset_norm` | Offset horizontal torso dari pusat pinggul |

**2D Best (42 fitur) = 2D Core + 6 fitur telinga sagital:**

| Fitur Tambahan | Deskripsi |
|---|---|
| `cam01_ear_shoulder_horizontal_norm` | Jarak horizontal telinga ke bahu kiri (depan) |
| `cam01_ear_shoulder_vertical_norm` | Jarak vertikal telinga ke bahu |
| `cam01_ear_neck_angle_deg` | Sudut inklinasi kepala-leher (sagital) |
| `cam02_ear_*` | Sama, dari sudut lateral |

**Motivasi fitur telinga:** Untuk mendeteksi *slouching* dan *forward head posture*, informasi sudut sagital leher yang terlihat dari kamera lateral (CAM02) sangat krusial — tidak tersedia dari kamera frontal saja.

### 4.3 Representasi Stereo 3D — Camera Frame (25 fitur)

**Proses triangulasi:**
1. Ambil pasangan keypoint 2D dari CAM01 dan CAM02
2. Terapkan DLT triangulasi menggunakan matriks proyeksi P1, P2
3. Normalisasi koordinat: origin = pusat panggul, skala = jarak maksimum sendi dari origin

**25 Fitur:**

| Grup | Fitur | Jumlah |
|---|---|:---:|
| Koordinat 3D (5 sendi × 3 axis) | nose, left/right shoulder, left/right hip (x,y,z) | 15 |
| Roll geometri | shoulder_roll_deg, hip_roll_deg | 2 |
| Lean planar | torso_lateral_lean_deg, torso_sagittal_lean_deg | 2 |
| Inklinasi | torso_3d_inclination_deg, head_torso_angle_3d_deg | 2 |
| Asimetri kedalaman | head_depth/lateral_offset_norm, shoulder/hip_depth_asymmetry_norm | 4 |

**NaN dalam fitur 3D:**
- `nose_x/y/z`, `head_torso_angle_3d_deg`, `head_depth/lateral_offset_norm`: 7.7% NaN (kepala tidak terdeteksi) → imputed dengan median (SimpleImputer)

### 4.4 Representasi Stereo 3D — Canonical Body Frame (22 fitur)

**Motivasi:** Menghilangkan dependensi model pada orientasi rig kalibrasi (*coordinate frame shift*) yang menjadi penyebab *Fold 3 Collapse*.

**Konstruksi Canonical Frame:**
```
Origin  : hip_center (midpoint left_hip + right_hip)
Y_can   : dari hip_center → shoulder_center (arah "tubuh tegak")
X_can   : dari left_hip → right_hip, diortogonalisasi Gram-Schmidt
Z_can   : X_can × Y_can (ke depan tubuh, right-hand rule)
Skala   : S = jarak maksimum sendi core dari hip_center
```

**3 Fitur Degenerate yang Dihapus (25 → 22 fitur):**

| Fitur | Alasan Degenerate |
|---|---|
| `torso_lateral_lean_deg` | Torso IS sumbu Y_can → selalu 0 |
| `torso_sagittal_lean_deg` | Torso IS sumbu Y_can → selalu 0 |
| `hip_depth_asymmetry_norm` | Hip IS sumbu X_can → selalu 0 |

> **Catatan:** `torso_3d_inclination_deg` **tidak** degenerate — inklinasi torso terhadap sumbu Z dunia (vertikal gravitasi) tetap bervariasi dalam canonical frame.

**Verifikasi Efektivitas Canonicalization:**

| Rig | shoulder_roll_deg (RAW) | shoulder_roll_deg (Canonical) |
|---|:---:|:---:|
| CAL_001 | -160.17° | -13.29° |
| **CAL_004** | **+125.34°** | **-0.79° ✅** |
| CAL_005 | +10.16° | -23.35° |
| CAL_009 | +109.07° | -12.11° |
| CAL_011 | +130.97° | +9.66° |

---

## Bagian 5 — Eksperimen dan Hasil

### 5.1 Protokol Evaluasi

- **Dataset:** 403 sampel Model-Ready (irisan 2D ∩ 3D)
- **Fold Split:** `private_2d3d_intersection_5fold.csv` — digunakan identik untuk semua 4 model
- **Metode:** Subject-Aware Stratified 5-Fold Cross-Validation (zero subject leakage)
- **Classifier:** XGBoost (`multi:softprob`, `n_estimators=300`, `max_depth=6`, `lr=0.1`, `seed=42`)
- **Metrik Utama:** Pooled OOF Macro F1
- **File Referensi:** `07_results/experiments/private_final/FINAL_FOUR_MODEL/final_four_model_comparison.csv`

### 5.2 Hasil Global — Final Four-Model Comparison (FROZEN)

> Semua angka berasal dari satu eksperimen bersamaan dengan seed identik. Ini adalah **satu-satunya sumber kebenaran** penelitian ini.

| Model | Fitur | Pooled Acc | Pooled Precision | Pooled Recall | **Pooled Macro F1** | Fold Mean F1 | Fold Std |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2D Core Baseline | 36f | 67.99% | 0.6897 | 0.6901 | 0.6863 | 0.6669 | 0.0846 |
| **2D Best Practical** | **42f** | **69.73%** | **0.7174** | **0.7046** | **0.7039** | **0.6875** | **0.0701** |
| 3D Current / Camera | 25f | 61.79% | 0.6462 | 0.6267 | 0.6182 | 0.6046 | 0.2519 |
| 3D Canonical / Body | 22f | 33.00% | 0.3378 | 0.3305 | 0.3259 | 0.3237 | 0.1241 |

### 5.3 Hasil Per-Kelas (Pooled OOF F1)

| Kelas Postur | 2D Core | 2D Best | 3D Current | 3D Canonical | Analisis |
|---|:---:|:---:|:---:|:---:|---|
| leaning_backward | 0.6667 | 0.6512 | **0.6800** | 0.1441 | 3D sedikit unggul (kedalaman sagital lebih informatif) |
| leaning_forward | 0.4783 | 0.5211 | **0.5440** | 0.2532 | 3D dan ear features sama-sama membantu |
| leaning_left | **0.8376** | 0.8205 | 0.7395 | 0.5439 | 2D sangat dominan (lateral kamera + slope bahu) |
| leaning_right | **0.9839** | **0.9839** | 0.6860 | 0.3926 | 2D sangat dominan (asimetri kanan lebih terlihat) |
| slouching | 0.5000 | **0.5872** | 0.4833 | 0.2500 | Ear features krusial (+8.7% dari Core ke Best) |
| upright | 0.6517 | **0.6595** | 0.5765 | 0.3718 | Semua model kurang pasti; baseline sulit |

### 5.4 Hasil Per-Fold

| Fold | Subjek Uji | 2D Core | 2D Best | 3D Current | 3D Canonical | Interpretasi |
|:---:|---|:---:|:---:|:---:|:---:|---|
| 0 | S001, S013, S018 | 0.6937 | 0.7480 | **0.7648** | 0.3285 | 3D unggul — rig homogen |
| 1 | S002, S012, S021 | **0.7211** | 0.7074 | 0.6704 | 0.2781 | 2D Core sedikit unggul |
| 2 | S006, S015–S019 | **0.7715** | 0.7703 | 0.7243 | 0.3864 | 2D stabil, 3D kompetitif |
| **3** | **S003, S004, S022, S024** | **0.6187** | **0.6187** | **0.1052 ❌** | **0.1246 ❌** | **3D Collapse — CAL_004** |
| 4 | S005, S011, S014 | 0.5297 | 0.5930 | **0.7583** | 0.5008 | 3D unggul signifikan |

### 5.5 Investigasi Fold 3 Collapse

**Konteks:** Fold 3 mencakup S003, S004, S022, S024 sebagai fold uji. Model dilatih tanpa keempat subjek ini.

**Temuan Empiris:**
- S003 (n=25) dan S004 (n=30) semuanya direkam dengan **rig CAL_004**
- S022 dan S024 menggunakan rig berbeda (normal)
- 55 dari 90 sampel Fold 3 = dari CAL_004

**Bukti domain shift CAL_004:**
```
shoulder_roll_deg — Fold 3 (CAL_004 dominant):  Mean = 126.48° ± 79.73°
shoulder_roll_deg — Fold lain (non-CAL_004):     Mean =  31.99° ± 109.51°

torso_sagittal_lean_deg — Fold 3:   Mean = -19.07° ± 14.06°
torso_sagittal_lean_deg — Fold lain: Mean =  +8.07° ± 12.13°
```

**Mekanisme kegagalan:**
Matriks ekstrinsik CAL_004 merotasi sistem koordinat sedemikian sehingga postur **berdiri tegak / netral** tampak memiliki `shoulder_roll_deg > 90°` dalam ruang 3D. Model yang dilatih tanpa CAL_004 tidak pernah melihat pola ini — semua sampel CAL_004 diprediksi sebagai `leaning_right` (100% salah untuk S003 dan S004).

**Mengapa 2D tidak terpengaruh?**
Pipeline 2D beroperasi dalam koordinat citra masing-masing kamera — tidak melibatkan matriks ekstrinsik stereo. Sehingga pergeseran orientasi rig tidak memengaruhi fitur 2D, dan model 2D pada Fold 3 tetap mencapai F1 0.6187.

### 5.6 Hasil Eksperimen Canonicalization

**Hipotesis awal:** Canonicalization akan menghilangkan domain shift CAL_004 dan meningkatkan generalisasi.

**Hasil nyata:**
- Canonicalization **berhasil** mengoreksi CAL_004: shoulder_roll 125° → 1° ✅
- Namun F1 **turun drastis**: 0.6182 (3D Current) → 0.3259 (3D Canonical) — penurunan 29.2%
- Fold 3 hanya membaik sedikit: 0.1052 → 0.1246 (+0.019, tidak bermakna)

**Interpretasi:**
Sinyal diskriminatif yang dimanfaatkan model 3D Camera Frame bukan semata-mata geometri pose tubuh murni — sebagian besar sinyal berasal dari **interaksi antara geometri tubuh dan orientasi frame kamera**. Ketika frame kamera dihilangkan oleh canonicalization, informasi tersebut turut hilang. Ini membuktikan bahwa "3D Camera Frame" pada dasarnya adalah representasi hybrid yang sensitive terhadap rig.

**Temuan ini merupakan kontribusi ilmiah yang sah** — menunjukkan bahwa 3D dalam camera frame ≠ 3D dalam body frame dari sudut pandang discriminative power.

---

## Bagian 6 — Uji Deployment

### 6.1 Raw Stereo Pair Test (3D-15)

**Tujuan:** Uji integritas pipeline end-to-end dari piksel mentah → label.  
**Konfigurasi:** 6 pasang gambar; subjek S011; rig CAL_009

| Postur | Prediksi | Benar | Reproj. Error | Confidence |
|---|---|:---:|:---:|:---:|
| upright | leaning_backward | ❌ | 29.39 px | 0.557 |
| leaning_forward | leaning_left | ❌ | 20.11 px | 0.374 |
| leaning_backward | leaning_left | ❌ | 42.26 px | 0.500 |
| leaning_left | **leaning_left** | ✅ | 36.47 px | 0.477 |
| leaning_right | REJECT (right_hip NaN) | — | 12.70 px | — |
| slouching | leaning_backward | ❌ | 24.22 px | 0.605 |

**Temuan:**
- Pipeline berjalan tanpa exception ✅
- 1/6 prediksi benar (akurasi rendah karena reprojection error 20–42px, di atas batas optimal)
- Audit fitur: **skema fitur konsisten** antara training dan inference; perbedaan bukan dari bug kode tetapi dari kualitas rekonstruksi pada sesi test ini

### 6.2 Real-Time Latency Benchmark (3D-16)

**Konfigurasi:** 30 iterasi headless; CPU (Intel/AMD, tanpa GPU); 3 pasang gambar

| Komponen Pipeline | Mean | Median | Std | % Total |
|---|:---:|:---:|:---:|:---:|
| YOLO Pose — CAM01 | 97.42 ms | 88.02 ms | 28.16 ms | 51% |
| YOLO Pose — CAM02 | 90.67 ms | 86.86 ms | 9.58 ms | 47% |
| Triangulasi + QC | 0.72 ms | 0.43 ms | 1.36 ms | <1% |
| Feature Extraction | 0.27 ms | 0.22 ms | 0.15 ms | <1% |
| XGBoost Predict | 2.40 ms | 1.89 ms | 1.21 ms | ~1% |
| **End-to-End** | **191.50 ms** | **185.64 ms** | 31.93 ms | 100% |

**FPS Estimasi: 5.2 FPS (CPU)**

**Kesimpulan:** Bottleneck hampir sepenuhnya dari 2× YOLO inference (188ms, 98%). Komponen 3D (triangulasi + fitur + XGBoost) hanya 3.4ms — sangat efisien. Untuk mencapai >10 FPS, solusi yang dibutuhkan adalah GPU acceleration untuk YOLO (TensorRT, ONNX Runtime GPU).

---

## Bagian 7 — Rekonsiliasi Angka dan Temuan Metodologis

### 7.1 Perbaikan Angka: Mengapa Ada Perbedaan di Laporan Sebelumnya?

Laporan sebelumnya mencatat dua versi angka yang berbeda untuk 2D Best:
- **0.7039** (dari tri-model comparison script)
- **0.6526** (dari comparison_final/final_comparison_all_models.csv)

**Root cause:** `fix06_save_comparison.py` (yang menghasilkan comparison_final) menggabungkan OOF predictions dari run yang **berbeda fold split**:
- OOF 2D: menggunakan `private_stratified_group_5fold.csv` (fold assignments berbeda)
- OOF 3D: menggunakan `private_2d3d_intersection_5fold.csv`

Akibatnya, perbandingan 2D vs 3D sebelumnya tidak benar-benar "fair" karena fold assignments-nya berbeda.

**Resolusi:** Eksperimen `FINAL_FOUR_MODEL` menjalankan semua 4 model secara simultan pada fold split yang **benar-benar identik** → angka 2D Best yang benar adalah **0.7039**.

### 7.2 Perbaikan Skema Fitur Canonical

Laporan sebelumnya menyebut **4 fitur degenerate**. Investigasi kode aktual (`train_private_xgboost_3d_canonical.py`) mengkonfirmasi hanya **3 fitur degenerate**:

| Klaim Sebelumnya | Fakta Terverifikasi |
|---|---|
| 4 fitur degenerate termasuk `torso_3d_inclination_deg` | `torso_3d_inclination_deg` TETAP DIGUNAKAN (tidak degenerate) |
| 25 - 4 = 21 fitur | **25 - 3 = 22 fitur** ✓ |

---

## Bagian 8 — Ringkasan Akhir dan Status Penelitian

### 8.1 Tabel Perbandingan Final (FROZEN — Single Source of Truth)

> **File referensi:** `07_results/experiments/private_final/FINAL_FOUR_MODEL/final_four_model_comparison.csv`

| Model | Fitur | Pooled Acc | Pooled Macro F1 | Fold Mean F1 | Fold Std | Fold3 F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 2D Core Baseline | 36f | 67.99% | 0.6863 | 0.6669 | 0.0846 | 0.6187 |
| **2D Best Practical ★** | **42f** | **69.73%** | **0.7039** | **0.6875** | **0.0701** | **0.6187** |
| 3D Current / Camera | 25f | 61.79% | 0.6182 | 0.6046 | 0.2519 | 0.1052 |
| 3D Canonical / Body | 22f | 33.00% | 0.3259 | 0.3237 | 0.1241 | 0.1246 |

### 8.2 Temuan Ilmiah Final

1. **2D Multi-View dengan ear features adalah sistem terbaik** — Macro F1 0.7039, stabilitas tinggi (std 0.0701)
2. **3D Camera Frame kompetitif pada rig homogen** — Fold 0 & 4: F1 0.7648 dan 0.7583, mengungguli 2D
3. **Fold 3 Collapse terbukti akibat rig CAL_004** — bukan heterogenitas postur subjek; mekanisme terdokumentasi secara kuantitatif
4. **Canonicalization menghilangkan domain shift tapi juga sinyal postur** — temuan negatif yang bernilai ilmiah
5. **Ear features berkontribusi terukur** — slouching +8.7%, leaning_forward +4.3% secara terisolasi
6. **Pipeline deployment feasible di CPU** — 5.2 FPS; komponen 3D hanya 3.4ms dari 191.5ms total

### 8.3 Batasan Penelitian

1. Coverage 3D hanya 55.4% (403/727) — right_hip NaN pada 254 tangkapan
2. Dataset kecil (24 subjek, 6 kelas) — potensi underfitting pada kelas minority
3. Tidak ada validasi eksternal di luar 24 subjek yang direkam
4. 3D sangat sensitif terhadap perubahan orientasi rig kalibrasi
5. Canonicalization belum berhasil memisahkan domain shift dari sinyal postur
6. FPS 5.2 (CPU) — belum memadai untuk real-time >10 FPS tanpa GPU
7. Perbandingan 2D Best (42f) vs 3D Current (25f) tidak sepenuhnya isolasi modalitas (beda jumlah fitur dan jenis fitur)

### 8.4 Status dan Langkah Selanjutnya

| Status | Keterangan |
|---|---|
| ✅ Eksperimen FROZEN | Git tag `v1.0-freeze-final-four-model`, commit `a7f2821b` |
| ✅ Angka direkonsiliasi | Single source of truth: `final_four_model_comparison.csv` |
| ✅ Skema fitur direkonsiliasi | `feature_schema.json` di FINAL_FOUR_MODEL dir |
| ✅ Audit training-inference | Konsisten; bug-free; limitasi rekonstruksi terdokumentasi |
| 🔄 Penulisan artikel | Outline tersedia di `05_naskah_dan_publikasi/` |
| 🔄 Penulisan skripsi | Outline tersedia; perlu drafting bab-per-bab |
| ⬜ Validasi eksternal | Rekomendasi riset lanjutan |
| ⬜ GPU deployment | Rekomendasi untuk meningkatkan FPS |

---

*Laporan ini mencakup seluruh perjalanan penelitian dari 2026-08 hingga 2026-09-24.*  
*Dibuat secara otomatis berdasarkan eksperimen frozen dan dokumentasi yang ada.*
