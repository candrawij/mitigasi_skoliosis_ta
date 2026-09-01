# LAPORAN METODOLOGI & AUDIT 3D REPROJECTION FINAL (23 SUBJEK)

**Judul Tugas Akhir:** Sistem Deteksi Postur Duduk & Mitigasi Skoliosis Berbasis Multi-View Pose Estimation & Stereo Vision  
**Cakupan Data:** **23 Partisipan Penuh (`S001` s/d `S023`)**  
**Total Data Terkumpul:** **851 Pasang Capture (1.702 Citra Full HD 1080p)**  
**Status Validasi:** 🟢 **TERVALIDASI LENGKAP & TERISOLASI BERSIH (2D vs 3D)**  
**Tanggal Update:** 2 September 2026  

---

## 1. Status Checkpoint Penelitian Terkunci

```
========================================================================================
                      STATUS KUNCI PIPELINE RISET (23 SUBJEK)
========================================================================================

  YOLO / Person Selection             🟢 LOCKED (Akurasi 99.47%, Wrong Person 0.35%)
  2D Keypoint Extraction              🟢 LOCKED (1.699 / 1.702 Citra Lengkap)
  2D Dataset QC Protocol              🟢 LOCKED (823 Capture Siap Latih 2D)
  Calibration Mapping                 🟢 LOCKED (7 Rig Terpetakan Resmi)

  3D Triangulation                    🟢 WORKING (Normalisasi Resolusi 1080p <-> 480p)
  CAL_010 Degenerate Rig              🟢 CONFIRMED (Tz=124m Corrupt -> Excluded 3D)
  CAL_006 Degenerate/Invalid 3D       🟢 EXCLUDED (Baseline 4.53m Terlalu Lebar)

  3D Reprojection QC (T5.E.2)         🟢 LOCKED (Spine Core Error 3-35px, Joint Masking)
  CAL_008 (S008 - S010)               🟢 VERIFIED & LOCKED (107/111 Capture Usable 3D)
  3D Anatomical Sanity QC             🟢 LOCKED (Filter Rentang Metrik Bahu & Kedalaman Z)
  Final 3D Dataset Decoupling         🟢 LOCKED (747 Capture 3D vs 823 Capture 2D)

========================================================================================
```

---

## 2. Hasil Verifikasi Matematis & Rerun `CAL_008` (111 Captures)

* **Temuan Empiris:** Rig `CAL_008` (digunakan oleh `S008`, `S009`, `S010`) memiliki parameter ekstrinsik yang sehat (Baseline $= 1.319\text{ m}$).
* **Penyebab Spike Reprojection:** Terjadi akibat oklusi fisik alami pada panggul jauh saat subjek melakukan postur miring (*Leaning Left / Right*).
* **Solusi Joint-Level Masking:** Keypoint panggul yang tertutup di-masking menjadi `NaN` tanpa membuang fitur bahu dan kepala.
* **Hasil Akhir `CAL_008`:** **$107 / 111\text{ capture}$ ($96.40\%$)** terverifikasi aman dan berintegritas tinggi untuk cabang 3D ($12\text{ Full 3D} + 95\text{ Masked 3D}$, hanya $4\text{ capture reject/outlier}$ yang diexclude).

---

## 3. Matriks Keputusan 3D Final per Rig Kalibrasi (851 Captures)

File CSV: [`07_results/private_audit/private_3d_qc_final.csv`](file:///d:/.Candra/Project/TA/07_results/private_audit/private_3d_qc_final.csv)

```text
========================================================================================================================
                     MATRIKS KEPUTUSAN 3D QC PER SETUP RIG KALIBRASI (851 CAPTURES)
========================================================================================================================
Rig ID   | Baseline | Cakupan Subjek | INCLUDE_3D_FULL | INCLUDE_3D_MASKING | EXCLUDE_3D (Degenerate/Outlier) | Status Rig
---------+----------+----------------+-----------------+--------------------+---------------------------------+------------
CAL_001  | 1.289 m  | S001 – S002    |   57 capture    |     23 capture     |     2 capture                   | 🟢 EXCELLENT
CAL_004  | 1.214 m  | S003 – S004    |   22 capture    |     51 capture     |     1 capture                   | 🟢 WORKING
CAL_005  | 2.401 m  | S005 – S006    |   66 capture    |      5 capture     |     1 capture                   | 🟢 EXCELLENT
CAL_006  | 4.534 m  | S007           |    0 capture    |      0 capture     |    39 capture (Baseline 4.5m)   | 🔴 EXCLUDE-3D
CAL_008  | 1.319 m  | S008 – S010    |   12 capture    |     95 capture     |     4 capture                   | 🟢 VERIFIED
CAL_009  | 2.501 m  | S011 – S022    |  308 capture    |    108 capture     |    20 capture                   | 🟢 EXCELLENT
CAL_010  | 124.8 m  | S023           |    0 capture    |      0 capture     |    37 capture (Corrupt Tz)      | 🔴 EXCLUDE-3D
---------+----------+----------------+-----------------+--------------------+---------------------------------+------------
TOTAL    | 7 Rig    | 23 SUBJEK      | 465 (54.64%)    |   282 (33.14%)     |   104 (12.22%)                  | LOCKED
========================================================================================================================
```

* **Total Data 3D Usable (Lolos QC):** **$747\text{ capture}$ ($87.78\%$)** ($465\text{ Full 3D} + 282\text{ Masked 3D}$).
* **Total Data 3D Excluded:** **$104\text{ capture}$ ($12.22\%$)** ($76\text{ dari rig corrupt CAL_006 & CAL_010} + 28\text{ sampel negatif/outlier}$).

---

## 4. Tabel Statistik Dataset Privat (23 Subjek) — Siap Laporan Skripsi

```text
====================================================================================================
                            TABEL STATISTIK DATASET PRIVAT (23 SUBJEK)
====================================================================================================
Parameter Evaluasi                                              Nilai Kuantitatif
--------------------------------------------------------------+-------------------------------------
1. Total Partisipan (Subjects)                                 | 23 Responden (S001 s/d S023)
2. Total Pasangan Pengambilan (Captures)                       | 851 Pasang Capture
3. Total Citra Fisik Full HD (1920 × 1080)                     | 1.702 Citra (851 CAM01 + 851 CAM02)
4. Distribusi Kelas Postur:                                    |
   - Upright (Tegak Normal)                                    | 121 capture (14.22%)
   - Leaning Forward (Condong Depan)                           | 119 capture (13.98%)
   - Forward Head (Kepala Maju)                                | 116 capture (13.63%)
   - Leaning Left (Skoliosis Kiri)                             | 115 capture (13.51%)
   - Leaning Right (Skoliosis Kanan)                           | 115 capture (13.51%)
   - Slouching (Kifosis Bungkuk)                               | 115 capture (13.51%)
   - Leaning Backward (Condong Belakang)                       | 114 capture (13.40%)
   - Reject (Sampel Negatif / Out of Frame)                    |  36 capture ( 4.23%)
5. Total Data Valid Cabang Model 2D (PASS + REVIEW)            | 823 Capture (96.71%)
6. Total Data Valid Cabang Model 3D (Include Full + Masked)    | 747 Capture (87.78%)
7. Total Data Excluded Cabang 3D (Rig Corrupt CAL_006/010)     | 104 Capture (12.22%)
====================================================================================================
```
