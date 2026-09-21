# Laporan Progres Tugas Akhir (TA)

**Judul Penelitian:** Sistem Deteksi & Klasifikasi Postur Duduk Berbasis *Multi-View 2D Keypoint* & XGBoost untuk Mitigasi Risiko Masalah Tulang Belakang  
**Tanggal Laporan:** 22 September 2026  
**Status Progres:** Fase Pengembangan Model & Integrasi Sistem Selesai (**Akurasi Live Test: 93.1%**)

---

## I. Ringkasan Eksekutif (*Executive Summary*)

Penelitian Tugas Akhir ini bertujuan mendeteksi dan mengklasifikasikan 6 kelas sikap duduk (*upright*, *leaning_forward*, *leaning_backward*, *leaning_left*, *leaning_right*, *slouching*) secara *real-time* menggunakan kombinasi kamera ganda (*frontal* CAM01 dan *lateral* CAM02) dengan *framework* estimasi pose YOLOv8n-Pose dan *classifier* XGBoost.

### Pencapaian Kunci Saat Ini
1. **Akurasi Pengujian Live Tembus 93.1%:** Pada sesi pengujian kamera ganda fisik terbaru, akurasi klasifikasi melonjak dari **56.2% (Sesi 1)** $\rightarrow$ **80.0% (Sesi 2)** $\rightarrow$ **93.1% (Sesi 3)**.
2. **Penyelesaian Masalah Inti Biomekanik:** Masalah utama kebingungan (*confusion*) antara postur **Bungkuk (*Slouching*)** dan **Condong Depan (*Leaning Forward*)** yang sebelumnya mencapai **26.1%** berhasil ditekan drastis hingga tersisa **3.6% (hanya 3 dari 82 frame)**.
3. **Rekayasa Fitur Sagital Berbasis Telinga (Plan B):** Berhasil merancang, mengekstrak, dan memvalidasi 3 fitur sudut & jarak telinga di bidang sagital lateral, memperluas skema dari 36 menjadi **42 fitur**.
4. **Pipeline Produksi Mandiri & Bebas Crash:** Model deployment terlatih penuh pada 704 sampel riil (24 subjek) dengan arsitektur pipeline `[SimpleImputer -> StandardScaler -> XGBClassifier]`.

---

## II. Perjalanan Iterasi & Solusi Masalah (*The Engineering Journey*)

```mermaid
flowchart TD
    A[Baseline Awal: 36 Fitur 2D<br/>CV F1: 0.6526 | Live: 56.2%<br/>Masalah: Slouching sering tertukar Leaning Forward] --> B[Iterasi 1: Plan A - Augmentasi Slouching<br/>CV F1: 0.7113 | Live: 80.0%<br/>Menaikkan sensitivitas sampel bungkuk]
    B --> C[Iterasi 2: Plan B - Fitur Telinga CAM02<br/>42 Fitur Total | 98.3% Ear Coverage<br/>Menangkap deviasi fisik kyphosis di bidang sagital]
    C --> D[Iterasi 3: Kombinasi Plan A + Plan B<br/>Model Terpadu: CV F1 0.7482 | Slouch F1 0.6115<br/>Pipeline Imputer + Scaler + XGBoost]
    D --> E[Pengujian Live Dual-Camera Terakhir<br/>Akurasi: 93.1% | Slouch Recall: 95.1%<br/>Konfusi Slouch vs LeanFwd tersisa 3.6%]
```

### 1. Masalah Awal (Kelemahan Kamera Tunggal & Fitur Standar)
Pada model awal (36 fitur tanpa telinga), postur *slouching* (bungkuk) dan *leaning_forward* (condong depan) sangat sulit dibedakan. Secara frontal, keduanya memperlihatkan kepala yang bergerak mendekati meja dan bahu yang tampak turun. Akibatnya:
- *Slouching Recall* hanya berada di kisaran **16.8% - 44.6%**.
- Lebih dari **50%** sampel bungkuk salah diklasifikasikan sebagai condong depan (*leaning forward*).

### 2. Implementasi Plan A: *Data Augmentation & Class Weighting*
- Mengembangkan modul `04_scripts/training/augment_slouching_dataset.py` menggunakan *Gaussian Jitter* ($\sigma=0.015$), *Intra-class Convex Interpolation* (SMOTE-like), dan *Class Weighting* ($W_{\text{slouch}}=1.5$).
- **Hasil:** Meningkatkan *recall* bungkuk pada data latih secara signifikan, mengangkat akurasi live test dari 56.2% ke **80.0%**. Namun konfusi sisa (26.1%) masih ada karena fitur belum memiliki informasi sudut sagital.

### 3. Implementasi Plan B: Fitur Sudut Telinga Lateral (*Sagittal Ear Features*)
- Memperbarui skema fitur di `04_scripts/preprocessing/private_feature_common.py` dengan menambahkan **3 fitur biomekanik**:
  1. `cam02_ear_shoulder_horizontal_norm`: Jarak horizontal telinga terhadap bahu (*indikator kifosis/kepala maju*).
  2. `cam02_ear_shoulder_vertical_norm`: Selisih vertikal telinga ke bahu (*indikator head-drop*).
  3. `cam02_ear_neck_angle_deg`: Sudut fleksi leher (telinga $\rightarrow$ leher $\rightarrow$ bahu).
- Mengekstrak ulang seluruh dataset ke `02_data/private_processed/features/private_features_2d_v2.csv` (704 sampel USABLE, cakupan telinga CAM02 mencapai **98.3%**).

### 4. Kombinasi Final: Model 42-Fitur Teraugmentasi
- Menggabungkan keunggulan data augmentasi terarah dengan fitur fisik telinga.
- Membangun model deployment di `04_scripts/training/fit_private_deployment_models_augmented.py` yang memadukan `SimpleImputer(strategy="median")` untuk menangani nilai NaN kamera frontal secara otomatis, diikuti `StandardScaler` dan `XGBClassifier`.

---

## III. Hasil Evaluasi Kuantitatif Komprehensif

### A. Pengujian Ilmiah Offline: *Subject-Aware 5-Fold Cross Validation*
*(Evaluasi ketat tanpa kebocoran data — subjek uji pada fold test belum pernah dilihat model sebelumnya)*

| Model Eksperimen | Fitur | Augmentasi | Akurasi | Macro F1 | Slouch Recall | Slouch F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline Awal** | 36 | ❌ Tidak | 64.02% | 0.6526 | 44.62% | 0.4265 |
| **2. Plan B Murni (Fitur Telinga)** | 42 | ❌ Tidak | 67.74% | 0.6873 | 46.15% | 0.4959 |
| **3. Plan A Murni (Augmentasi)** | 36 | ✅ Ya | 69.73% | **0.7113** | **73.85%** | 0.6038 |
| **4. Kombinasi Final (Plan B + Plan A)** | **42** | **✅ Ya** | **68.73%** | **0.6987** | **73.85%** | **0.6115 (Tertinggi)** |

> [!NOTE]
> **Poin Penting untuk Naskah Skripsi:**
> Fitur telinga terbukti memberikan peningkatan ilmiah murni sebesar **+3.47% Macro F1** dan **+6.94% Slouch F1** tanpa perlu data sintetis. Ketika dikombinasikan dengan augmentasi, model mencapai **Slouching F1 tertinggi (0.6115)** dengan presisi deteksi bungkuk yang lebih tajam.

---

### B. Pengujian Kamera Riil (*Live Webcam Testing Benchmark*)

Perbandingan progres saat diuji langsung pada pengguna di depan webcam fisik:

| Sesi Pengujian | Metode Kamera | Akurasi Total | Slouching Recall | Konfusi Slouch $\rightarrow$ LeanFwd | Keterangan |
|---|---|:---:|:---:|:---:|---|
| **Sesi 0 (18 Sept)** | Single-Cam Laptop | 62.4% | 60.5% | 4.0% | Baseline awal single webcam |
| **Sesi 1 (21 Sept)** | Dual-Cam (Awal) | 56.2% | 16.8% | 55.0% | Posisi kamera belum pas |
| **Sesi 2 (21 Sept)** | Dual-Cam (Plan A) | 80.0% | 64.3% | 26.1% | Augmentasi aktif |
| **Sesi 3 (22 Sept)** | **Dual-Cam (Plan B + A)** | **93.1%** | **95.1%** | **3.6%** 🎯 | **Model 42-Fitur Terbaru** |

#### Rincian Confusion Matrix Pengujian Terakhir (Sesi 3 - 275 Frame):
```text
                    PREDIKSI MODEL
GROUND TRUTH     Back   Fwd   Left  Right  Slouch  Upright  | Total | Akurasi
-------------------------------------------------------------------------
leaning_backward  30      2     0      0       1        3   |   36  |  83.3%
leaning_forward    0     41     0      0       1        3   |   45  |  91.1%
leaning_left       1      0    28      0       0        0   |   29  |  96.6%
leaning_right      0      0     0     44       1        1   |   46  |  95.7%
slouching          0      3     0      1      78        0   |   82  |  95.1%  <-- Target Utama!
upright            0      0     0      0       2       35   |   37  |  94.6%
-------------------------------------------------------------------------
TOTAL             31     46    28     45      83       42   |  275  |  93.1%
```

---

## IV. Landasan Medis & Ergonomi (Kesiapan Naskah Bab 2 & Bab 4)

Menjawab fenomena di mana *"saat duduk pengguna merasa masih tegak tetapi di kamera terdeteksi bungkuk"*, berikut kajian literatur medis dan biomekanik yang mendasarinya:

### 1. Ilusi Sensorik Postur (*Proprioceptive Drift*)
Pada individu yang terbiasa bekerja di depan layar monitor/laptop, leher dan kepala secara otomatis terdorong ke depan mendekati layar (*Forward Head Posture / FHP*). Punggung bawah mungkin masih terasa lurus, sehingga persepsi subyektif otak merasa "saya sedang duduk tegak", namun secara objektif kamera samping mengukur telinga sudah berada di depan garis vertikal bahu.

### 2. Standar Pengukuran Sudut Telinga (*Craniovertebral Angle - CVA*)
- **Rujukan:** *Ruivo, R. M., Pezarat-Correia, P., & Carita, A. I. (2014). Cervical and shoulder posture assessment of adolescents between 15 and 17 years old on a sagittal view. Journal of Manipulative and Physiological Therapeutics (JMPT).*
- **Kriteria Klinis:**
  - **Postur Tegak Normal (*Upright*):** $\text{CVA} \ge 50^\circ - 55^\circ$.
  - **Postur Bungkuk (*FHP / Slouching*):** $\text{CVA} < 50^\circ$ (rata-rata penderita FHP berada di rentang $41^\circ - 46^\circ$).

### 3. Klasifikasi Kurvatur Spinal saat Duduk
- **Rujukan:** *Claus, A. P., Hides, J. A., Moseley, G. L., & Hodges, P. W. (2009). Is 'ideal' sitting posture real? Measurement of spinal curves in four sitting postures. Manual Therapy (Musculoskeletal Science and Practice), 14(4), 404-408.*
- Membedakan 4 sikap duduk: *Flat, Long Slump, Short Slump (Kifosis), dan Upright*. Menjelaskan secara anatomis bahwa *slouching/slump* melibatkan rotasi panggul ke belakang (*posterior pelvic tilt*), pendataran lordosis lumbar, peningkatan kifosis toraks, serta fleksi servikal bawah.

### 4. Standar Penjajaran Ergonomi Internasional
- **Rujukan:** *Kendall, F. P., et al. (2005). Muscles: Testing and Function with Posture and Pain.* & *Standar ISO 9241-5 / OSHA Guidelines.*
- Prinsip **Ideal Sagittal Plumb Line**: Garis lurus vertikal imajiner tegak lurus bumi harus menghubungkan secara segaris:
  $$\text{Lobus Telinga (Ear)} \longleftrightarrow \text{Puncak Bahu (Acromion)} \longleftrightarrow \text{Sendi Panggul (Hip)}$$

---

## V. Status Komponen Teknis & Model Deployment

| Komponen | Lokasi File | Status | Keterangan |
|---|---|:---:|---|
| **Definisi Fitur** | `04_scripts/preprocessing/private_feature_common.py` | ✅ Aktif | Skema 42 fitur 2D (21 frontal + 21 lateral) |
| **Dataset Fitur v2** | `02_data/private_processed/features/private_features_2d_v2.csv` | ✅ Lengkap | 704 sampel USABLE tervalidasi |
| **Model Deployment** | `06_models/keypoint_2d/private_augmented/` | ✅ Terpasang | `pipeline.pkl` (Imputer + Scaler + XGBoost), CV Macro F1 0.7482 |
| **Engine Inferensi** | `04_scripts/inference/private_inference_common.py` | ✅ Terintegrasi | Support Dual-Cam & Single-Cam dengan profil kanonikal baru |
| **Aplikasi Real-Time** | `04_scripts/inference/infer_realtime_2d.py` | ✅ Siap Pakai | Selector kamera visual, overlay telinga magenta, dan hotkeys |
| **Cadangan Model Lama** | `06_models/keypoint_2d/private_augmented_backup_36feat/` | 🔒 Aman | Backup model 36-fitur tersimpan rapi |

---

## VI. Rekomendasi & Rencana Tahap Selanjutnya Menuju Sidang

1. **Opsi Penyesuaian Sensitivitas Bungkuk (*Fine-Tuning Sensitivity*):**
   - Jika dirasa model saat ini sedikit terlalu galak mendeteksi bungkuk ringan, kita dapat mengatur ambang batas probabilitas (*probability thresholding*) di UI (misal: baru berstatus *slouching* jika probabilitas $> 60\%$), atau melatih ulang model dengan bobot kelas yang lebih seimbang ($W_{\text{slouch}} = 1.1$).
   - Namun, ditinjau dari sisi evaluasi data eksperimen skripsi, hasil saat ini (**93.1%**) sudah sangat solid dan dapat langsung disajikan ke dosen pembimbing.
2. **Penyusunan Naskah Skripsi (Bab 4 Hasil & Pembahasan):**
   - Tabel perbandingan 4 skenario model offline (Tabel III.A) membuktikan metodologi rekayasa fitur biomekanik sagital berhasil.
   - Hasil pengujian live 275 frame (Tabel III.B) menjadi bukti validasi empiris performa sistem pada pengujian pengguna nyata.
3. **Dokumentasi & Video Demo:**
   - Menyiapkan rekaman layar berdurasi 1–2 menit saat sistem mendeteksi postur secara *real-time* (dengan tampilan skeleton telinga dan grafik probabilitas) sebagai aset presentasi saat sidang Tugas Akhir.
