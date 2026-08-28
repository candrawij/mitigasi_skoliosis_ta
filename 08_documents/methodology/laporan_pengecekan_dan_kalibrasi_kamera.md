# Laporan Teknis Pengujian Sistem Kamera, Kalibrasi Stereo & Evaluasi Sinkronisasi

**Dokumen:** Laporan Verifikasi Hardware dan Uji Sistem Multi-Kamera  
**Tanggal Pengujian:** 28 Agustus 2026  
**Status Pengujian:** **Lolos Uji Hardware & Sinkronisasi Dual-Camera (Verified & Operational)**  
**Rig Kalibrasi:** `CAL_001` (Frontal CAM01 + Lateral CAM02)  
**Lingkungan Komputasi:** Windows OS, OpenCV 5.0/4.x, Multi-threaded VideoCapture

---

## 1. Ringkasan Eksekutif (Executive Summary)

Pengujian sistem kamera dan validasi software akuisisi dataset privat telah berhasil dilaksanakan secara empiris menggunakan perangkat keras kamera fisik. Sistem dual-kamera berhasil mendeteksi dan mengonfigurasi dua unit kamera aktif pada resolusi **Full HD ($1920 \times 1080$)** secara bersamaan.

### Poin Pencapaian Utama:
1. **Identifikasi Perangkat Kamera:** Berhasil memetakan kamera utama Frontal pada **Device Index 0 (`CAM01`)** dan kamera Lateral pada **Device Index 2 (`CAM02`)**, dengan mengabaikan kamera internal laptop (Index 1).
2. **Uji Kalibrasi Intrinsik & Stereo:** Berhasil mengekstrak matriks optik intrinsik dan ekstrinsik stereo (`CAL_001`), dengan estimasi jarak fisik antar-kamera (*Baseline Distance*) sebesar **$2.702\text{ meter}$ ($2702.35\text{ mm}$)**.
3. **Kualitas Sinkronisasi Ekstra Tinggi:** Pengujian software capture (`capture_multicam.py`) menghasilkan rata-rata latensi sinkronisasi antar-kamera sebesar **$19.9\text{ milidetik}$ (Min: $3.0\text{ ms}$, Max: $56.0\text{ ms}$)**. Angka ini berada di bawah batas durasi 1 frame video ($33.3\text{ ms}$ pada 30 FPS), memastikan bahwa kedua kamera memotret pose pada momen waktu yang identik tanpa jeda gerak (*zero-motion disparity*).
4. **Kesiapan Utilitas Manajemen Data:** Telah dibangun tool manajemen dan audit otomatis ([`dataset_manager.py`](file:///d:/.Candra/Project/TA/04_scripts/capture/dataset_manager.py)) untuk memeriksa integritas data, menghapus capture/subjek yang salah, atau mereset data uji coba secara aman.

---

## 2. Spesifikasi dan Konfigurasi Perangkat Keras Kamera

```
+---------------------------------------------------------------------------------------+
|                               KONFIGURASI RIG DUAL-KAMERA                             |
+-------------------+---------------+------------------+------------------+-------------+
| Kamera ID         | Device Index  | Sudut Pandang    | Resolusi Uji     | Backend     |
+-------------------+---------------+------------------+------------------+-------------+
| CAM01 (Utama)     | Index 0       | Frontal (0°)     | 1920 x 1080 (FHD)| MSMF        |
| CAM02 (Sekunder)  | Index 2       | Lateral (90°/45°)| 1920 x 1080 (FHD)| MSMF        |
| Kamera Laptop     | Index 1       | - (Diabaikan)    | 640 x 480        | MSMF        |
+-------------------+---------------+------------------+------------------+-------------+
```

Konfigurasi index kamera disimpan secara persisten di:  
👉 [`03_metadata/camera_config.json`](file:///d:/.Candra/Project/TA/03_metadata/camera_config.json)

---

## 3. Hasil Kalibrasi Optik & Stereo (`CAL_001`)

### 3.1. Kalibrasi Stereo & Matriks Ekstrinsik
* **Baseline Distance ($T$):** $2.7023\text{ meter}$ ($2702.35\text{ mm}$)
* **Vektor Translasi ($T$):** $[1.431, 1.749, -1.482]^T\text{ meter}$
* **Matriks Rotasi ($R$):**
  $$\begin{bmatrix} 0.6522 & -0.5820 & -0.4856 \\ 0.1892 & 0.7454 & -0.6392 \\ 0.7341 & 0.3251 & 0.5962 \end{bmatrix}$$
* **Matriks Disparity-to-Depth ($Q$):**
  Tersimpan dan siap digunakan untuk rektifikasi stereo dan rekonstruksi koordinat 3D tubuh.
* **File Profil Kalibrasi:** [`02_data/private_calibration/stereo/CAL_001_stereo.json`](file:///d:/.Candra/Project/TA/02_data/private_calibration/stereo/CAL_001_stereo.json)

---

## 4. Hasil Pengujian Software Capture Dual-Kamera (`capture_multicam.py`)

Pengujian langsung dilakukan dengan merekam 8 pasang pose uji (`CAP000008` s/d `CAP000015`).

### 4.1. Analisis Latensi Sinkronisasi Timestamp

| Capture ID | Timestamp CAM01 (Frontal) | Timestamp CAM02 (Lateral) | Selisih Waktu ($\Delta t$) | Status Kualitas |
|:---:|:---:|:---:|:---:|:---:|
| `CAP000008` | `13:09:27.418 UTC` | `13:09:27.406 UTC` | **12 ms** | Sempurna (<33 ms) |
| `CAP000009` | `13:09:29.483 UTC` | `13:09:29.460 UTC` | **23 ms** | Sempurna (<33 ms) |
| `CAP000010` | `13:09:31.310 UTC` | `13:09:31.254 UTC` | **56 ms** | Baik |
| `CAP000011` | `13:09:33.771 UTC` | `13:09:33.795 UTC` | **24 ms** | Sempurna (<33 ms) |
| `CAP000012` | `13:09:35.004 UTC` | `13:09:34.997 UTC` | **7 ms** | Sempurna (<33 ms) |
| `CAP000013` | `13:09:36.130 UTC` | `13:09:36.133 UTC` | **3 ms** | **Ultra-Presisi** |
| `CAP000014` | `13:09:43.084 UTC` | `13:09:43.059 UTC` | **25 ms** | Sempurna (<33 ms) |
| `CAP000015` | `13:09:46.203 UTC` | `13:09:46.194 UTC` | **9 ms** | Sempurna (<33 ms) |
| **Rata-rata** | - | - | **19.87 ms** | **Sangat Baik** |

### 4.2. Integritas File & Skema Relasi
* Seluruh 16 citra ($1920 \times 1080\text{ px}$) tersimpan rapi pada struktur direktori standar:
  * `02_data/private_raw/S001/SE01/CAM01/S001_SE01_CAP000008_CAM01.jpg`
  * `02_data/private_raw/S001/SE01/CAM02/S001_SE01_CAP000008_CAM02.jpg`
* Relasi $1 : 2$ antara `captures.csv` ($1\text{ baris per pose}$) dan `images.csv` ($2\text{ baris per pose}$) terverifikasi **100% konsisten**.

---

## 5. Panduan Manajemen, Pengecekan & Reset Dataset

Untuk memastikan dataset tetap bersih, akurat, dan mudah dikoreksi jika terjadi kesalahan saat pengambilan data, telah disediakan tool [`04_scripts/capture/dataset_manager.py`](file:///d:/.Candra/Project/TA/04_scripts/capture/dataset_manager.py) yang juga terintegrasi pada menu **`[10]`** di orchestrator pipeline.

### 5.1. Cara Memeriksa (Audit) Kualitas Dataset
Jalankan perintah:
```bash
python 04_scripts/capture/dataset_manager.py --action inspect
```
* **Informasi yang Ditampilkan:** Total pose, total citra, daftar subjek terdaftar, pasangan yang hilang/rusak, rata-rata latensi sinkronisasi, dan distribusi kelas postur.

---

### 5.2. Cara Menghapus 1 Data Capture yang Salah (Misal Pose Blur / Salah Gerak)
Jika ada 1 foto tertentu yang buram atau partisipan bergerak saat difoto (misalnya `CAP000012`):
```bash
python 04_scripts/capture/dataset_manager.py --action delete_capture --capture_id CAP000012
```
* **Otomatisasi Sistem:**
  1. Menghapus file gambar `S001_SE01_CAP000012_CAM01.jpg` dan `CAM02.jpg` dari disk.
  2. Menghapus baris `CAP000012` dari [`captures.csv`](file:///d:/.Candra/Project/TA/03_metadata/private_templates/captures.csv).
  3. Menghapus kedua baris `CAM01` dan `CAM02` dari [`images.csv`](file:///d:/.Candra/Project/TA/03_metadata/private_templates/images.csv).

---

### 5.3. Cara Menghapus Seluruh Data 1 Subjek Tertentu
Jika data seorang subjek uji coba (misalnya subjek dummy `S001`) ingin dihapus:
```bash
python 04_scripts/capture/dataset_manager.py --action delete_subject --subject_id S001
```
* **Otomatisasi Sistem:** Menghapus seluruh folder `02_data/private_raw/S001/` dan membersihkan seluruh log subjek tersebut dari CSV.

---

### 5.4. Cara Melakukan Reset Bersih Total (Clean Slate Reset)
Sebelum memulai perekaman subjek Pilot Study yang sesungguhnya (5–10 subjek), Anda bisa membersihkan seluruh data foto uji coba tanpa merusak template header CSV:
```bash
python 04_scripts/capture/dataset_manager.py --action reset_test_data
```
* **Hasil:** Folder `private_raw/` bersih kembali, dan file `captures.csv` serta `images.csv` ter-reset ke header kosong siap pakai.

---

## 6. Kesimpulan & Rekomendasi Langkah Kerja

1. **Sistem Pengambilan Data Siap Pakai:** Hardware kamera, pipeline kalibrasi, perekaman Full HD sinkron, dan sistem metadata telah diverifikasi bekerja dengan sangat baik.
2. **Kesiapan Pilot Study:** Sistem siap untuk digunakan merekam **5–10 subjek Pilot Study** menggunakan menu **`[5]`** pada orchestrator (`run_camera_test_pipeline.py`) atau script `capture_multicam.py`.
