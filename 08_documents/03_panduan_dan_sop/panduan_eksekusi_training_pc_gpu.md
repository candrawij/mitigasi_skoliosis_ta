# Panduan Lengkap Eksekusi Eksperimen & Training di PC (GPU/Lokal)

Dokumen ini adalah panduan teknis operasional untuk menjalankan seluruh rangkaian eksperimen klasifikasi postur duduk di PC/Laptop target (terutama dengan GPU).

---

## 1. Daftar Eksperimen & Peran

| ID | Nama Eksperimen | Arsitektur / Pipeline | Dataset Target | Output & Metrik Utama |
|---|---|---|---|---|
| **EXP-01** | **Image Baseline (CNN)** | EfficientNet-B0 (Pretrained ImageNet) | 1. `project_design` (5 kelas)<br>2. `sitting_posture_detection` (4 kelas) | `results.json`, Confusion Matrix, Macro F1, Akurasi |
| **EXP-02** | **Keypoint Baseline** | Fitur Geometri + MLP & XGBoost | 1. `postureexercise` (7 keypoint, 5 kelas)<br>2. `ikorn_4kp` (4 keypoint, 2 kelas) | `comparison.csv`, Akurasi, Balanced Acc, Macro F1 |
| **EXP-03** | **YOLO Pose + Classifier** | YOLOv8 Pose (17 KP) + Feature Engineering + MLP/XGBoost | 1. `project_design` (5 kelas)<br>2. `sitting_posture_detection` (4 kelas) | `results.json`, Macro F1, Akurasi |
| **EXP-05** | **Cross-Dataset Generalization** | Evaluasi model binary Good vs Bad lintas dataset | Postureexercise, IKORN, SPD, Project Design | `cross_dataset_results.csv`, Transfer Matrix |

---

## 2. Persiapan di PC Target (Setup Environment)

### A. Clone Repository & Masuk ke Folder Project
```bash
git clone <URL_REPOSITORY_ANDA>
cd <NAMA_FOLDER_PROJECT>
git pull origin main
```

### B. Buat Virtual Environment (Sangat Disarankan)
```powershell
# Di Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### C. Install Dependencies
```powershell
# 1. Install PyTorch dengan Akselerasi GPU NVIDIA (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# (Catatan: Jika PC target hanya CPU, gunakan: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu)

# 2. Install semua library pendukung:
pip install -r requirements.txt
```

---

## 3. Cara Menjalankan Training

### Opsi 1: Menjalankan SEMUA Eksperimen Sekaligus (1-Klik Runner) ⭐ *Paling Praktis*
Script ini akan otomatis melakukan preprocessing, menjalankan **EXP-01, EXP-02, EXP-03, EXP-05**, lalu menghasilkan tabel ringkasan perbandingan (`master_summary.csv` dan `master_summary.md`):

```bash
python 04_scripts/run_all_experiments.py --all
```

*Estimasi Waktu di PC dengan GPU NVIDIA:* **~5–10 menit total**.

---

### Opsi 2: Menjalankan Eksperimen Satu per Satu Secara Terpisah

Jika Anda ingin mengecek hasil per eksperimen secara bertahap:

#### 1. Jalankan EXP-02 (Keypoint Classifiers - MLP & XGBoost)
```bash
python 04_scripts/training/train_keypoint_classifiers.py
```
*Hasil:* `07_results/experiments/EXP-PE-KP/` & `EXP-IK-KP/`

#### 2. Jalankan EXP-01 (EfficientNet-B0 CNN Image Baseline)
```bash
python 04_scripts/training/train_efficientnet_baseline.py
```
*Hasil:* `07_results/experiments/EXP-PD-CNN/` & `EXP-SPD-CNN/`

#### 3. Jalankan EXP-03 (YOLO Pose 17 Keypoints + MLP/XGBoost)
```bash
python 04_scripts/training/train_yolo_pose_classifier.py
```
*Hasil:* `07_results/experiments/EXP-YOLO-POSE-PROJECT_DESIGN/` & `EXP-YOLO-POSE-SITTING_POSTURE_DETECTION/`

#### 4. Jalankan EXP-05 (Cross-Dataset Evaluation)
```bash
python 04_scripts/evaluation/evaluate_cross_dataset.py
```
*Hasil:* `07_results/experiments/EXP-05-CROSS-DATASET/`

---

## 4. Sinkronisasi Hasil Kembali ke GitHub

Setelah training selesai di PC:
```bash
git add 07_results/
git commit -m "Upload complete experiment results from GPU PC"
git push origin main
```

Lalu di perangkat ini, Anda cukup menjalankan:
```bash
git pull origin main
```
Seluruh tabel metrik, grafik, dan analisis komparasi siap langsung diolah untuk laporan tugas akhir / skripsi!
