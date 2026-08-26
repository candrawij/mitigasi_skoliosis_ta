# Konvensi Penamaan, Sistem Identifikasi (ID), dan Standar Metadata Dataset Privat

Dokumen ini adalah acuan baku teknis untuk penamaan file, sistem identifier, struktur direktori, dan pengelolaan metadata dataset privat penelitian mitigasi postur duduk.

---

## 1. Aturan Penamaan File (File Naming Convention)

Setiap citra yang diambil oleh sistem kamera **wajib** mengikuti pola penamaan standar berikut:

```text
{subject_id}_{session_id}_{capture_id}_{camera_id}.jpg
```

### Contoh Nyata:
* `S001_SE01_CAP000001_CAM01.jpg` (Citra subjek 1, sesi 1, pose instance 1, dari kamera frontal)
* `S001_SE01_CAP000001_CAM02.jpg` (Citra subjek 1, sesi 1, pose instance 1, dari kamera samping)
* `S002_SE02_CAP000124_CAM01.jpg`

> ⚠️ **PENTING: Jangan Menaruh Nama Kelas pada Nama File!**  
> Label postur (`primary_posture`) disimpan di dalam file metadata `captures.csv` menggunakan relasi `capture_id`, bukan di nama file. Hal ini mencegah kebocoran label (*label leakage*) saat audit buta (*blind audit*) dan mempermudah perbaikan label bila ada revisi.

---

## 2. Spesifikasi Sistem Identifier (ID)

| Identifier | Pola Format | Contoh Nilai | Deskripsi & Aturan |
|---|---|---|---|
| **`subject_id`** | `S` + 3 digit angka | `S001`, `S002`, `S030` | Kode unik anonim peserta penelitian. **Nama asli peserta dilarang keras dicantumkan.** |
| **`session_id`** | `SE` + 2 digit angka | `SE01`, `SE02` | Nomor sesi pengambilan data subjek (hari/kondisi berbeda). |
| **`capture_id`** | `CAP` + 6 digit angka | `CAP000001`, `CAP000145` | ID tunggal unik untuk **satu momen pose**. Semua kamera yang memotret pose tersebut secara bersamaan memiliki `capture_id` yang sama persis. |
| **`camera_id`** | `CAM` + 2 digit angka | `CAM01`, `CAM02`, `CAM03` | Kode kamera: `CAM01` (Frontal 0°), `CAM02` (Lateral 90°), `CAM03` (Diagonal 45° - opsional). |
| **`calibration_id`** | `CAL_` + 3 digit angka | `CAL_001`, `CAL_002` | Kode konfigurasi fisik rig kamera. Terpisah (*decoupled*) dari `session_id`. Jika posisi/sudut kamera diubah, buat `calibration_id` baru. |

---

## 3. Skema Metadata dan Relasi Data

```text
participants.csv (subject_id)
      │
      └─── captures.csv (capture_id, subject_id, session_id, calibration_id)
                 │                                        │
                 ├─── images.csv (image_id, capture_id)   └─── calibration_map.csv (calibration_id)
                 └─── qc_audit_log.csv (capture_id)
```

### 3.1. Kebijakan Privasi (`participants.csv`)
* **Minimal Wajib:** `subject_id`, `session_count`, `consent_rgb`, `consent_public`, `notes`.
* **Data Sensitif/Opsional:** `age_group`, `height_cm` hanya dicatat jika peserta setuju dan dibutuhkan untuk analisis variasi antropometri.
* **Kerahasiaan Identitas:** Dokumen informed consent fisik yang berisi nama asli disimpan secara terpisah di brankas/folder terenkripsi lokal dan tidak dimasukkan ke dalam repositori kode git publik.

### 3.2. Dekopling `calibration_id`
* Satu `calibration_id` (misal `CAL_001`) berlaku untuk semua sesi/subjek selama posisi fisik kamera CAM01 dan CAM02 tidak bergeser.
* Jika tripod dipindahkan atau ruangan berganti, lakukan proses kalibrasi ulang dan catat sebagai `CAL_002`.

---

## 4. Status Taksonomi dan Representasi Target

* **Target Keypoint Representation:** **COCO-17** (Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles).  
  *Catatan: COCO-17 adalah representasi pose yang ditargetkan dan diekstrak oleh model pose (YOLO-Pose), bukan kewajiban anotasi manual titik per titik pada citra mentah.*
* **Daftar Kelas Inti (Status: DRAFT):**
  1. `upright` (Tegak netral)
  2. `leaning_forward` (Torso condong ke depan)
  3. `leaning_backward` (Torso bersandar ke belakang)
  4. `leaning_left` (Torso condong ke kiri subjek)
  5. `leaning_right` (Torso condong ke kanan subjek)
  6. `slouching` (Punggung membulat / kifosis)
  7. `forward_head` (Kepala maju ke depan)
* **Kondisi Penolak / Eksklusi (Status: DRAFT):**
  1. `standing` (Peserta berdiri)
  2. `transition` (Perpindahan antar pose / belum siap)
  3. `no_person` (Kamera kosong / tertutup)

*Validasi apakah `slouching` dan `forward_head` dapat dibedakan secara visual konsisten dari kelas lain akan difinalisasi melalui evaluasi Pilot Study (5–10 subjek).*

---

## 5. Pedoman Quality Control (QC) Reprojection Error

Batas toleransi reprojection error digunakan sebagai **panduan mutu (QC guideline)**, bukan eliminasi otomatis:
* **< 1.0 px:** Kualitas kalibrasi sangat baik (Optimal untuk triangulasi 3D).
* **1.0 – 2.0 px:** Kualitas kalibrasi baik dan dapat diterima (*Acceptable*).
* **2.0 – 3.0 px:** Memerlukan pemeriksaan visual pada epipolar lines.
* **> 3.0 px:** Disarankan untuk mengambil ulang frame kalibrasi checkerboard.
