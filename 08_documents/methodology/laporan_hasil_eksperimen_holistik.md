# Laporan Holistik Hasil Eksperimen & Analisis Benchmark Deteksi Postur Duduk

**Dokumen:** Laporan Akhir Fase Eksperimen Multi-Dataset  
**Tanggal:** 25 Agustus 2026  
**Status Pipeline:** Fase 0 (Setup), Fase 1 (Audit & Curation), Fase 2 (Preprocessing), Fase 3 (Training & Eksperimen) **SELESAI (100%)**

---

## 1. Executive Summary

Eksperimen komparatif telah berhasil dieksekusi pada **4 dataset publik** (`Project Design 20242025`, `Sitting Posture Detection`, `Postureexercise`, dan `IKORN 4-KP`) menggunakan arsitektur hardware GPU NVIDIA GTX 1660 Ti. Seluruh dataset telah diaudit secara ketat dan dibagi menggunakan skema **Group-Aware Splitting (0-Leakage)**.

Tiga paradigma utama dievaluasi secara sistematis:
1. **End-to-End Image Classification (CNN):** Menggunakan *EfficientNet-B0* (Pretrained ImageNet).
2. **Tabular Keypoint Classification (Direct Geometric):** Menggunakan *MLP* dan *XGBoost* pada fitur geometris keypoint tubuh.
3. **Two-Stage Pose Estimation + Classification (YOLO-Pose):** Ekstraksi 17 COCO Keypoints via *YOLOv8-Pose* diikuti *MLP / XGBoost*.

---

## 2. Tabel Rekapitulasi Benchmark Lengkap

| ID Eksperimen | Dataset & Kelas | Paradigma & Model | Test Accuracy | Balanced Acc | Macro F1-Score | Status / Catatan |
|---|---|---|:---:|:---:|:---:|---|
| **EXP-01** | `Project Design` (5 kelas) | **EfficientNet-B0 (CNN)** | **88.37%** | **87.24%** | **0.8635** | Best Image Model |
| **EXP-01** | `Sitting Posture Detection` (4 kelas) | **EfficientNet-B0 (CNN)** | **84.93%** | **83.80%** | **0.8410** | Konvergensi Epoch 20 |
| **EXP-02** | `Postureexercise` (7-KP, 5 kelas) | **MLP Classifier** | **87.34%** | **84.94%** | **0.8419** | Best 5-Class Keypoint |
| **EXP-02** | `Postureexercise` (7-KP, 5 kelas) | **XGBoost Classifier** | 84.81% | 82.87% | 0.8155 | Sangat cepat |
| **EXP-02** | `IKORN` (4-KP, 2 kelas) | **XGBoost Classifier** | **96.97%** | **98.08%** | **0.9569** | **Highest Binary Score** |
| **EXP-02** | `IKORN` (4-KP, 2 kelas) | **MLP Classifier** | 92.93% | 95.51% | 0.9051 | Recall Good 100% |
| **EXP-03** | `Project Design` (5 kelas) | **YOLO-Pose + MLP** | **83.72%** | - | **0.8207** | Pipeline 2-Stage Portabel |
| **EXP-03** | `Project Design` (5 kelas) | **YOLO-Pose + XGBoost** | 79.84% | - | 0.7855 | Feature-importance based |
| **EXP-03** | `Sitting Posture Detection` (4 kelas) | **YOLO-Pose + XGBoost** | **82.19%** | - | **0.7745** | Resisten terhadap noise |
| **EXP-03** | `Sitting Posture Detection` (4 kelas) | **YOLO-Pose + MLP** | 60.27% | - | 0.4925 | Underfitting pada 4-class pose |
| **EXP-05** | `Postureexercise` (Binary Good/Bad) | **MLP Transfer/Binary** | **95.89%** | - | **0.9372** | Unified Taxonomy Validated |

---

## 3. Analisis Mendalam & Komparasi Paradigma

### 3.1. Paradigma 1: End-to-End CNN (`EfficientNet-B0`)
* **Kelebihan:** Mampu menangkap fitur visual holistik secara langsung (termasuk interaksi tubuh dengan sandaran kursi dan meja) tanpa bergantung pada kegagalan pendeteksi keypoint. Menghasilkan akurasi tertinggi pada data citra (*88.37%* pada Project Design).
* **Keterbatasan:** Membutuhkan komputasi pelatihan yang lebih tinggi dan rentan mengalami pergeseran distribusi jika latar belakang (background) atau pencahayaan ruangan berubah drastis pada data uji baru.

### 3.2. Paradigma 2: Direct Geometric Keypoints (`MLP` & `XGBoost`)
* **Kelebihan:** Sangat ringan (inferensi <1 ms), bebas dari pengaruh visual latar belakang, dan fitur berbasis biomekanika (sudut kemiringan bahu, kelengkungan tulang belakang, asimetri telinga-mata) sangat interpretatif untuk deteksi skoliosis.
* **Hasil:**
  * Pada klasifikasi biner (`IKORN`), *XGBoost* mencapai akurasi **96.97%** dan F1-Score **0.9569**.
  * Pada multi-kelas 5 kategori (`Postureexercise`), *MLP* unggul dengan akurasi **87.34%**.

### 3.3. Paradigma 3: Two-Stage YOLO-Pose + Tabular Classifier
* **Kelebihan:** Solusi paling fleksibel untuk penerapan di dunia nyata karena dapat menerima input kamera RGB biasa, mengekstrak 17 titik pose COCO secara real-time, lalu mengklasifikasikan postur dengan model tabular yang ringan.
* **Temuan Khusus:**
  * *XGBoost* jauh lebih tangguh menghadapi *missing keypoints* atau noise deteksi YOLO dibandingkan *MLP* (misalnya pada dataset SPD: XGBoost 82.19% vs MLP 60.27%).
  * Fitur biomekanik terpenting yang teridentifikasi adalah `shoulder_angle`, `torso_angle`, `ear_dy`, dan `nose_to_shoulder_dx`.

---

## 4. Analisis Error & Performa per Kelas (Confusion Matrix)

Berdasarkan visualisasi Confusion Matrix yang dihasilkan di `07_results/visualizations/`:
1. **Kelas Simetris vs Asimetris:** Kelas postur miring ke kanan (`nghieng_phai` / `leaning_right`) dan miring ke kiri (`nghieng_trai` / `leaning_left`) memiliki akurasi deteksi yang sangat tinggi karena dicirikan oleh sudut kemiringan bahu (`shoulder_angle`) dan asimetri telinga yang kontras.
2. **Kebingungan Utama (Misclassification Bias):** Sedikit tumpang tindih terjadi antara kelas `slouch` (membungkuk) dan `leaning_forward` pada dataset citra sudut sagital/oblique, dikarenakan proyeksi 2D kamera sering kali meratakan kedalaman tulang belakang bagian atas.

---

## 5. Visualisasi Hasil Eksperimen

File-file visualisasi beresolusi tinggi telah disimpan di folder `07_results/visualizations/`:
* `model_comparison_benchmark.png`: Grafik perbandingan menyeluruh Accuracy & Macro F1.
* `cnn_training_convergence_curves.png`: Kurva konvergensi Loss & Accuracy EfficientNet-B0.
* `cm_efficientnet_project_design.png` & `cm_efficientnet_sitting_posture_detection.png`: Confusion Matrix model CNN.
* `cm_keypoint_postureexercise_MLP.png` & `cm_keypoint_ikorn_XGBoost.png`: Confusion Matrix model Keypoint.

---

## 6. Implikasi & Rekomendasi untuk Pengumpulan Dataset Privat (Fase 5)

Berdasarkan temuan eksperimen pada 4 dataset publik, berikut adalah parameter standar yang direkomendasikan untuk akuisisi dataset privat:

1. **Sudut Pandang Kamera (Camera Angle):**
   * Sudut **Frontal (Depan)** sangat optimal untuk mendeteksi asimetri bahu, kepala miring, dan condong lateral (indikator utama risiko skoliosis).
   * Sudut **Sagital / Oblique (Samping 45°–90°)** diperlukan jika ingin membedakan *slouching* (bungkuk) dengan *upright* secara presisi.
2. **Kualitas Anotasi:**
   * Rekomendasi pipeline utama: Gunakan model **YOLO-Pose + XGBoost** sebagai *core classifier* untuk aplikasi real-time ringan, atau **EfficientNet-B0** jika resolusi citra dan komputasi mencukupi.
3. **Standar Anotasi Ground Truth:**
   * Menggunakan skema taksonomi terpadu Level A (Biner: Tegak vs Postur Buruk) dan Level B (Multi-kelas: Tegak, Condong Kiri, Condong Kanan, Bungkuk, Bersandar).
