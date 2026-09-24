# LAPORAN AKHIR HASIL IMPLEMENTASI & EVALUASI PIPELINE FINAL 2D–3D XGBOOST
## Skripsi / Tugas Akhir: Sistem Deteksi & Mitigasi Postur Skoliosis Berbasis Multi-Kamera

**Tanggal Penyelesaian:** 5 September 2026  
**Status Pipeline:** FINAL & FREEZE  
**Repositori:** `mitigasi_skoliosis_ta`  
**Classifier Utama:** XGBoost (`multi:softprob`, `num_class=6`, `eval_metric="mlogloss"`, `tree_method="hist"`)  
**Pose Estimator Acuan:** YOLOv8n-Pose (COCO-17 Keypoints)  
**Protokol Validasi:** Subject-Aware Stratified Group 5-Fold Cross-Validation (Zero Subject Overlap)  

---

## 1. Ringkasan Eksekutif & Deklarasi Konvergensi

Penelitian ini membandingkan secara komprehensif efektivitas representasi fitur pose **2D Multi-View (Frontal + Lateral)** dan **Stereo 3D (Geometri Spasial Rekonstruksi Epipolar)** untuk klasifikasi 6 postur duduk (`upright`, `leaning_forward`, `leaning_backward`, `leaning_left`, `leaning_right`, `slouching`).

Seluruh keputusan metodologi yang ditetapkan pembimbing telah dipenuhi secara ketat:
1. **Total Data:** 24 subjek penelitian (`S001`–`S024`), 885 pasang stereo / 1.770 citra Full HD.
2. **Taksonomi 6 Kelas:** Kelas `forward_head` dieksklusi dari eksperimen utama (`class_removed_after_supervisor_review`). Kelas `reject` difungsikan sebagai *Negative / Invalid-Input Gate* sebelum classifier.
3. **Data Raw 6-Class:** Tepat **727 stereo captures** lolos kualifikasi taksonomi (`assert len(df) == 727`).
4. **Fair Comparison Intersection:** Komparasi head-to-head 2D vs 3D dievaluasi secara adil pada **403 capture berpasangan yang sama persis** dari 18 subjek penelitian, menggunakan partisi subject-aware 5-fold yang identik.

---

## 2. Tabel 1: Distribusi Dataset Final Penelitian

| Kelas Postur (`label`) | Kelas ID | Raw 6-Class | 2D Usable (Coverage) | 3D Usable (Coverage) | Fair Intersection | Persentase Intersection |
|---|:---:|:---:|:---:|:---:|:---:|
| `upright` | 0 | 126 | 125 (99.21%) | 82 (65.08%) | 82 | 20.35% |
| `leaning_forward` | 1 | 124 | 120 (96.77%) | 75 (60.48%) | 75 | 18.61% |
| `leaning_backward` | 2 | 118 | 118 (100.0%) | 58 (49.15%) | 58 | 14.39% |
| `leaning_left` | 3 | 119 | 118 (99.16%) | 62 (52.10%) | 62 | 15.38% |
| `leaning_right` | 4 | 120 | 108 (90.00%) | 61 (50.83%) | 61 | 15.14% |
| `slouching` | 5 | 120 | 115 (95.83%) | 65 (54.17%) | 65 | 16.13% |
| **TOTAL** | **6 Kelas** | **727** | **704 (96.84%)** | **403 (55.43%)** | **403** | **100.00%** |

*Catatan:*
- Selisih 23 capture pada 2D disebabkan oleh oklusi ekstrem di mana bahu/panggul tidak terdeteksi oleh pose estimator.
- Selisih 324 capture pada 3D disebabkan oleh kriteria epipolar stereo yang ketat (*core reprojection error* $> 35\text{ px}$ atau degradasi rigiditas kamera).

---

## 3. Evaluasi Subject-Aware Stratified Group 5-Fold Cross-Validation

Partisi 5-fold membagi 18 subjek secara disjoint (tanpa subject leakage):
- **Fold 0:** Test subjek `['S001', 'S003', 'S006']` ($N=92$)
- **Fold 1:** Test subjek `['S002', 'S012', 'S019']` ($N=55$)
- **Fold 2:** Test subjek `['S004', 'S013', 'S018', 'S021', 'S022']` ($N=124$)
- **Fold 3:** Test subjek `['S005', 'S011', 'S014', 'S016']` ($N=73$)
- **Fold 4:** Test subjek `['S015', 'S017', 'S024']` ($N=59$)

### Tabel 2: Hasil Evaluasi XGBoost 2D Multi-View (36 Fitur)

| Fold Evaluasi | Subjek Uji (Unseen) | Sampel Uji | Akurasi (%) | Macro Precision | Macro Recall | Macro F1 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | S001, S003, S006 | 92 | 57.61% | 0.6205 | 0.6019 | 0.5982 |
| **Fold 1** | S002, S012, S019 | 55 | 74.55% | 0.7892 | 0.7889 | 0.7845 |
| **Fold 2** | S004, S013, S018, S021, S022 | 124 | 63.71% | 0.6579 | 0.6491 | 0.6339 |
| **Fold 3** | S005, S011, S014, S016 | 73 | 57.53% | 0.4699 | 0.5584 | 0.5047 |
| **Fold 4** | S015, S017, S024 | 59 | 72.88% | 0.8792 | 0.7952 | 0.7471 |
| **Rata-Rata (Mean)** | **Semua 18 Subjek** | **403** | **65.26%** | **0.6833** | **0.6787** | **0.6537** |
| **Standar Deviasi (STD)** | — | — | **± 8.14%** | **± 0.1601** | **± 0.1091** | **± 0.1135** |
| **Overall OOF** | **Seluruh Sampel Out-of-Fold** | **403** | **64.02%** | **0.6518** | **0.6562** | **0.6526** |

---

### Tabel 3: Hasil Evaluasi XGBoost Stereo 3D (25 Fitur)

| Fold Evaluasi | Subjek Uji (Unseen) | Sampel Uji | Akurasi (%) | Macro Precision | Macro Recall | Macro F1 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | S001, S003, S006 | 92 | 54.35% | 0.6104 | 0.5322 | 0.5512 |
| **Fold 1** | S002, S012, S019 | 55 | 72.73% | 0.6404 | 0.6667 | 0.6394 |
| **Fold 2** | S004, S013, S018, S021, S022 | 124 | 56.45% | 0.6605 | 0.5684 | 0.5355 |
| **Fold 3** | S005, S011, S014, S016 | 73 | 64.38% | 0.5729 | 0.6070 | 0.5654 |
| **Fold 4** | S015, S017, S024 | 59 | 54.24% | 0.5921 | 0.5539 | 0.5431 |
| **Rata-Rata (Mean)** | **Semua 18 Subjek** | **403** | **60.43%** | **0.6153** | **0.5856** | **0.5669** |
| **Standar Deviasi (STD)** | — | — | **± 8.03%** | **± 0.0381** | **± 0.0526** | **± 0.0420** |
| **Overall OOF** | **Seluruh Sampel Out-of-Fold** | **403** | **59.31%** | **0.6092** | **0.5943** | **0.5947** |

---

## 4. Tabel 4: Komparasi Utama 2D Multi-View vs Stereo 3D (Head-to-Head)

| Metrik Evaluasi | 2D Multi-View (36 Fitur) | Stereo 3D (25 Fitur) | Delta ($\text{3D} - \text{2D}$) | Uji Signifikansi Statistik |
|---|:---:|:---:|:---:|---|
| **Akurasi OOF** | **64.02%** | 59.31% | -4.71% | McNemar Test: $\chi^2 = 2.5116, p = 0.1130$ (Tidak beda signifikan) |
| **Macro Precision** | **0.6518** | 0.6092 | -0.0426 | Rata-rata presisi 6 kelas |
| **Macro Recall** | **0.6562** | 0.5943 | -0.0619 | Rata-rata recall 6 kelas |
| **Macro F1 (Metrik Utama)** | **0.6526** | 0.5947 | -0.0580 | **Metrik utama skripsi** |
| **5-Fold Mean Akurasi** | **65.26% ± 8.14%** | 60.43% ± 8.03% | -4.83% | Paired $t$-test: $t = 1.157, p = 0.3096$ |
| **5-Fold Mean Macro F1** | **0.6537 ± 0.1135** | 0.5669 ± 0.0420 | -0.0868 | Paired $t$-test: $t = 1.896, p = 0.1265$ |
| **Variansi Macro F1 ($\text{STD}$)** | $\pm 0.1135$ | **$\pm 0.0420$** | **-0.0715** | **Model 3D memiliki stabilitas 2.7x lebih konsisten antarsubjek** |
| **Dimensi Vektor Fitur** | 36 fitur | **25 fitur** | **-11 fitur** | Representasi 3D lebih ringkas 30.5% |
| **Cakupan Operasional (Usable)** | **96.84% (704/727)** | 55.43% (403/727) | -41.40% | 2D toleran terhadap setup non-kaku |

---

## 5. Tabel 5: Komparasi Performa Per Kelas Postur

| Kelas Postur | Support | F1 2D | F1 3D | $\Delta$ F1 | Precision 2D | Precision 3D | Recall 2D | Recall 3D | Keunggulan Model |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `upright` | 82 | 0.5765 | **0.5833** | **+0.0069** | **0.5568** | 0.5091 | 0.5976 | **0.6829** | **Stereo 3D** |
| `leaning_forward` | 75 | 0.3453 | **0.5035** | **+0.1582** | 0.3750 | **0.5294** | 0.3200 | **0.4800** | **Stereo 3D (+45.8%)** |
| `leaning_backward` | 58 | **0.6789** | 0.5979 | -0.0810 | 0.7255 | **0.7436** | **0.6379** | 0.5000 | **2D Multi-View** |
| `leaning_left` | 62 | **0.9280** | 0.7273 | -0.2007 | **0.9206** | 0.7458 | **0.9355** | 0.7097 | **2D Multi-View** |
| `leaning_right` | 61 | **0.9606** | 0.8308 | -0.1299 | **0.9242** | 0.7826 | **1.0000** | 0.8852 | **2D Multi-View** |
| `slouching` | 65 | **0.4265** | 0.3252 | -0.1013 | **0.4085** | 0.3448 | **0.4462** | 0.3077 | **2D Multi-View** |

---

## 6. Analisis Ilmiah & Failure Analysis Mendalam

### A. Mengapa `leaning_forward` Sangat Diuntungkan oleh Rekonstruksi 3D?
- Pada representasi 2D, condong ke depan (*leaning forward*) terproyeksi mengalami *perspective foreshortening* pada kamera frontal (CAM01), dan sering kali rancu dengan variasi jarak kamera pada kamera lateral (CAM02). Akibatnya F1 2D hanya mencapai **0.3453**.
- Pada representasi 3D, fitur spasial `torso_sagittal_lean_deg` dan `head_depth_offset_norm` mengukur pergeseran fisik dalam sumbu optik ($Z$-axis dalam meter nyata). Hal ini menghasilkan lonjakan F1 yang sangat signifikan ke **0.5035 (+45.8%)**, dengan recall melonjak dari 32.0% menjadi 48.0%.

### B. Mengapa Bidang Lateral (`leaning_left` dan `leaning_right`) Lebih Unggul pada 2D?
- Kemiringan lateral berada tepat pada bidang koronal yang sejajar tegak lurus dengan sumbu kamera frontal (CAM01).
- Fitur 2D murni seperti `cam01_shoulder_slope_deg` dan `cam01_torso_inclination_deg` memiliki SNR (*signal-to-noise ratio*) yang sangat tinggi tanpa dipengaruhi oleh ketidakpastian triangulasi stereo. F1-score mencapai **0.9280** untuk condong kiri dan **0.9606** untuk condong kanan.
- Pada 3D, sedikit galat reprojeksi epipolar dapat mendistorsi sudut roll sebesar beberapa derajat, yang menurunkan F1 menjadi 0.7273 dan 0.8308.

### C. Analisis Kegagalan: Mengapa `slouching` Masih Tertukar dengan `leaning_forward`?
- Secara biomekanik muskuloskeletal, saat seseorang membungkuk (*slouching*), pusat massa dada dan kepala bergeser ke anterior, mirip dengan fase awal *leaning forward*.
- Pada 2D, `leaning_forward` keliru diklasifikasikan sebagai `slouching` sebanyak **36 kali**.
- Pada 3D, penambahan sudut 3D kepala-torso (`head_torso_angle_3d_deg`) berhasil memangkas kebingungan ini menjadi **26 kali** (penurunan error 27.8%). Namun, pada subjek dengan kelengkungan tulang belakang yang fleksibel, diferensiasi antara membungkuk ringan dan condong ringan tetap menjadi tantangan inheren pada model berbasis *sparse keypoints*.

---

## 7. Tabel 6: Karakteristik Operasional & Rekomendasi Deployment

| Karakteristik Sistem | 2D Multi-View | Stereo 3D | Rekomendasi Deployment Gabungan |
|---|:---:|:---:|---|
| **Framerate (FPS)** | ~13.5 FPS | ~15.2 FPS | Real-time ready ($\ge 10$ FPS) |
| **Kompleksitas Setup** | Fleksibel (tanpa kalibrasi kaku) | Memerlukan kalibrasi stereo kaku | **Hierarchical Dual-Mode:** |
| **Ketergantungan Epipolar** | Rendah (hanya butuh deteksi 2D) | Tinggi (kamera tidak boleh bergeser) | 1. Jalankan Stereo 3D jika lolos QC epipolar |
| **Stabilitas Antarsubjek** | $\text{STD} = \pm 11.35\%$ | **$\text{STD} = \pm 4.20\%$ (Sangat stabil)** | 2. Otomatis beralih (*graceful fallback*) ke 2D |
| **Cakupan Penggunaan (Coverage)** | **96.84%** | 55.43% | jika triangulasi 3D terdegradasi. |

---

## 8. Checklist Pemenuhan Kriteria Selesai (Definition of Done)

- [x] **Dataset Lengkap 24 Subjek:** Seluruh raw data 24 subjek terverifikasi (`S001`–`S024`).
- [x] **Eksklusi Taksonomi:** `forward_head` resmi dikeluarkan; `reject` menjadi gerbang QC.
- [x] **Audit 727 Captures:** Lolos assertion 727 sampel 6-kelas raw.
- [x] **Ekstraksi Fitur 2D (36 Fitur):** 704 sampel usable (96.84%), 0 data duplikat/inf.
- [x] **Ekstraksi Fitur 3D (25 Fitur):** 403 sampel usable (55.43%), konvensi koordinat CAM01 standar.
- [x] **Partisi Group 5-Fold:** Tepat 0 subject leakage antarlipatan validasi.
- [x] **Evaluasi XGBoost 2D:** Selesai (Akurasi 64.02%, Macro F1 0.6526).
- [x] **Evaluasi XGBoost 3D:** Selesai (Akurasi 59.31%, Macro F1 0.5947).
- [x] **Evaluasi Komparasi Head-to-Head:** Per-class comparison, McNemar test, dan paired t-test selesai.
- [x] **Model Deployment:** Tersimpan lengkap di `06_models/` beserta pipeline, scaler, schema, dan metadata.
- [x] **Modul Inferensi Offline & Single Capture:** Lolos verifikasi capture tunggal dan batch subjek.
- [x] **Modul Inferensi Citra Baru:** `infer_private_pair_2d.py` dan `infer_private_pair_3d.py` teruji pada citra Full HD.
- [x] **Prototipe Real-Time:** `infer_realtime_2d.py` (~13.5 FPS) dan `infer_realtime_stereo_3d.py` (~15.2 FPS) teruji dengan visual HUD.
- [x] **Master Orchestrator:** `run_private_final_pipeline.py` memverifikasi 9 tahap pipeline dengan sukses.

---

**Kesimpulan Ilmiah untuk Skripsi:**  
Meskipun 2D Multi-View menghasilkan skor F1 rata-rata yang sedikit lebih tinggi berkat akurasi superior pada bidang koronal lateral (`leaning_left` dan `leaning_right` > 92%), Stereo 3D membuktikan keunggulan fundamentalnya dalam memecahkan ambiguitas bidang sagital (`leaning_forward` meningkat +45.8%) serta menghadirkan representasi fitur yang jauh lebih konsisten antarsubjek (variansi F1 antarlipatan turun drastis dari $\pm 0.1135$ menjadi $\pm 0.0420$). Dengan demikian, implementasi arsitektur **Hierarchical Dual-Mode** merupakan kontribusi utama yang paling aplikatif dan tangguh untuk sistem mitigasi skoliosis berbasis penglihatan komputer.
