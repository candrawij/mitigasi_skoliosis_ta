# Outline Singkat — Penelitian Deteksi Postur Duduk

**Judul:**  
Sistem Deteksi dan Klasifikasi Postur Duduk Berbasis *Multi-View 2D Keypoint* dan *Stereo 3D Reconstruction* Menggunakan XGBoost

**Status:** Eksperimen dibekukan — hasil final tersedia

---

## Judul dan Tujuan

Mengembangkan sistem monitoring postur duduk yang memanfaatkan **dua kamera stereo** (frontal + lateral) tanpa sensor tambahan, membandingkan apakah rekonstruksi **3D** lebih unggul dari penggabungan **2D multi-view** untuk klasifikasi 6 postur duduk.

---

## Pipeline Penelitian

### 1. Pengumpulan Dataset
- 24 subjek, 6 kelas postur duduk (upright, leaning forward/backward/left/right, slouching)
- Setup: 2 kamera sinkron (frontal + lateral), kalibrasi stereo 11 rig
- Total: 727 tangkapan
- QC berlapis → **403 tangkapan model-ready** (55.4%); 704 tersedia untuk 2D (96.8%)
- Penyebab utama gugur 3D: koordinat sendi panggul kanan NaN (254 tangkapan)

### 2. Deteksi Pose — YOLOv8n-Pose
- Inferensi di kedua kamera → 17 keypoint COCO per gambar
- Filtering: ambil orang dengan bounding box terbesar
- Output: koordinat (x, y) + confidence per sendi

### 3. Feature Engineering (4 skema fitur diuji)

| Skema | Jumlah Fitur | Deskripsi |
|---|:---:|---|
| 2D Core | 36f | Koordinat + geometri dari 2 kamera; tanpa fitur telinga |
| 2D Best | 42f | 2D Core + 6 fitur sagital telinga (slouchin detector) |
| 3D Current | 25f | Koordinat 3D + geometri spasial; frame kamera asli |
| 3D Canonical | 22f | 3D yang ditransformasi ke frame tubuh; 3 fitur degenerate dihapus |

### 4. Triangulasi Stereo (khusus 3D)
- Rekonstruksi koordinat 3D dari pasangan keypoint 2D menggunakan DLT
- Normalisasi: origin = pusat panggul; skala = jarak max sendi dari pusat

### 5. Canonicalization Body Frame (khusus 3D Canonical)
- Transformasi ke sistem koordinat yang berpusat dan berorientasi pada tubuh
- Tujuan: menghilangkan ketergantungan pada orientasi kamera
- Hasilnya mengejutkan — akurasi malah turun drastis (lihat Hasil)

### 6. Klasifikasi — XGBoost
- Protokol: Subject-Aware 5-Fold Cross-Validation (24 subjek, zero leakage)
- Dataset: 403 sampel identik, fold split identik, seed=42 untuk semua 4 model
- Metrik utama: Pooled OOF Macro F1

### 7. Evaluasi dan Deployment
- Perbandingan 4 model pada kondisi identik
- Uji pipeline end-to-end dari gambar mentah → prediksi
- Benchmark latency CPU

---

## Hasil Utama (FROZEN)

| Model | Pooled Macro F1 | Fold Std | Keterangan |
|---|:---:|:---:|---|
| 2D Core Baseline (36f) | 0.6863 | 0.0846 | Baseline solid |
| **2D Best Practical (42f)** | **0.7039** | **0.0701** | **Sistem terbaik** |
| 3D Current / Camera (25f) | 0.6182 | 0.2519 | Baik tapi sangat tidak stabil |
| 3D Canonical / Body (22f) | 0.3259 | 0.1241 | Sangat buruk — canonicalization backfire |

**Temuan penting:**
- 3D ungguli 2D pada fold tanpa masalah rig (Fold 0: F1 0.7648 vs 0.7480)
- Fold 3 collapse: 3D turun ke 0.1052 akibat satu rig kalibrasi (CAL_004) yang mengalami rotational shift ekstrim — 2D tidak terpengaruh (tetap 0.6187)
- Canonicalization berhasil koreksi rig shift, tapi menghilangkan sinyal postur → F1 −29.2%
- Fitur telinga meningkatkan slouching +8.7% dan leaning_forward +4.3%

**Latency (CPU):** 191.5ms end-to-end ≈ 5.2 FPS; bottleneck: 2× YOLO inference (98%)

---

## Kelebihan Penelitian

| # | Kelebihan |
|---|---|
| 1 | Perbandingan fair: 4 model, data identik, fold identik, seed identik |
| 2 | Dataset multi-subjek asli dengan variasi rig nyata (6 rig stereo berbeda) |
| 3 | Zero-subject-leakage cross-validation — estimasi generalisasi yang jujur |
| 4 | Pipeline end-to-end terbukti berjalan di CPU (tanpa GPU) |
| 5 | Fitur telinga sagital terbukti berguna secara terisolasi |
| 6 | Temuan canonicalization bernilai ilmiah — menjelaskan sifat sinyal 3D Camera |
| 7 | Fold 3 collapse teranalisis tuntas — akar masalah CAL_004 teridentifikasi |

---

## Kekurangan Penelitian

| # | Kekurangan | Dampak |
|---|---|---|
| 1 | Coverage 3D hanya 55.4% (right_hip NaN) | Dataset 3D lebih kecil |
| 2 | 3D sangat sensitif terhadap perubahan rig | Generalisasi ke kamera baru tidak pasti |
| 3 | Canonicalization belum berhasil | Belum ada solusi untuk rig heterogen |
| 4 | FPS rendah: 5.2 FPS di CPU | Butuh GPU untuk real-time >10 FPS |
| 5 | Dataset kecil: 24 subjek | Potensi underfitting kelas minority |
| 6 | Tidak ada validasi eksternal | Risiko model terlalu spesifik ke 24 subjek |
| 7 | Perbandingan 2D Best (42f) vs 3D (25f) tidak murni modalitas | Beda jumlah dan jenis fitur |

---

## Kesimpulan

**2D Multi-View dengan ear features adalah sistem terbaik** untuk kondisi ini — lebih akurat, lebih stabil, dan coverage data jauh lebih tinggi. Stereo 3D memiliki **potensi superior pada rig yang homogen**, tetapi membutuhkan solusi yang lebih robust terhadap perubahan kalibrasi kamera sebelum siap untuk deployment nyata.

---

*Berdasarkan eksperimen frozen `v1.0-freeze-final-four-model` (2026-09-24)*  
*Sumber data: `07_results/experiments/private_final/FINAL_FOUR_MODEL/final_four_model_comparison.csv`*
