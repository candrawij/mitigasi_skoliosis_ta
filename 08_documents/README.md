# 📚 Master Index Dokumentasi Tugas Akhir (TA)

**Judul Penelitian:** Sistem Deteksi dan Klasifikasi Postur Duduk Berbasis *Multi-View 2D Keypoint* dan *Stereo 3D Reconstruction* Menggunakan XGBoost untuk Mitigasi Risiko Masalah Tulang Belakang  
**Repositori:** `mitigasi_skoliosis_ta`  
**Status Terakhir:** Seluruh eksperimen evaluasi 2D vs 3D, canonicalization, benchmark latency, dan integrasi pipeline telah selesai dan terkunci (*locked*).

---

## 🗂️ Struktur Pengelompokan Dokumen

Seluruh dokumentasi laporan, panduan teknis, kontrol kualitas, dan naskah publikasi telah dikelompokkan secara terstruktur ke dalam 5 direktori tematik di bawah `08_documents/`:

```text
08_documents/
│
├── README.md                                  <-- (Dokumen ini) Master Indeks & Panduan Navigasi
│
├── 01_laporan_evaluasi_dan_progres/           <-- Laporan Evaluasi, Progres Berkala & Temuan Ilmiah
│   ├── README.md                              (Panduan direktori laporan evaluasi)
│   ├── laporan_progress_stereo_3d_dan_evaluasi_final.md  <-- ⭐ [DOKUMEN UTAMA] Fair Comparison, 3D Canonicalization, Latency
│   ├── LAPORAN_PROGRES_TA_20260922.md         <-- Laporan pengujian live dual camera (Akurasi 93.1%)
│   ├── laporan_progres_komprehensif_penelitian.md
│   ├── laporan_akhir_pipeline_final_skripsi.md
│   ├── laporan_hasil_eksperimen_holistik.md
│   ├── laporan_pengecekan_dan_kalibrasi_kamera.md
│   └── laporan_fase_5c_target_person_selection_dan_qc.md
│
├── 02_laporan_qc_dataset/                     <-- Kontrol Kualitas & Kurasi Dataset Subjek Riil
│   ├── README.md                              (Panduan evolusi audit dataset)
│   ├── laporan_qc_dataset_keseluruhan_24_subjek.md  <-- ⭐ [FINAL DATASET] Audit lengkap 24 subjek
│   └── arsip_historis_qc/                     (Arsip kemajuan QC bertahap)
│       ├── laporan_qc_dataset_keseluruhan_23_subjek.md
│       ├── laporan_qc_dataset_keseluruhan_21_subjek.md
│       ├── laporan_qc_dataset_keseluruhan_10_subjek.md
│       ├── laporan_komprehensif_dataset_6_subjek_dan_evaluasi_pipeline.md
│       ├── pilot_dataset_quality_report.md
│       └── private_dataset_pilot_qc_report.md
│
├── 03_panduan_dan_sop/                        <-- Prosedur Operasional Standar & Panduan Teknis
│   ├── README.md                              (Panduan SOP & eksekusi)
│   ├── Panduan_Pengumpulan_Dataset_Privat_Postur_Duduk.md  <-- ⭐ SOP Protokol Pengambilan Data Subjek
│   ├── PANDUAN_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md <-- ⭐ Arsitektur & Spesifikasi Pipeline
│   └── panduan_eksekusi_training_pc_gpu.md    <-- Panduan training pada PC GPU
│
├── 04_perencanaan_dan_task/                   <-- Roadmap Riset, Desain Sistem & Pelacak Tugas
│   ├── README.md                              (Panduan roadmap & task tracking)
│   ├── RENCANA_IMPLEMENTASI_FINAL_SETELAH_EVALUASI_3D.md  <-- ⭐ Roadmap Penyelesaian Pasca Evaluasi 3D
│   ├── TASK_PENYELESAIAN_STEREO_3D_FINAL.md   <-- Tracker eksekusi tugas stereo 3D (3D-01 s/d 3D-17)
│   ├── TASK_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md <-- Tracker tugas pipeline 2D/3D XGBoost
│   ├── rencana_penelitian_postur_duduk_revisi.md
│   └── pipeline_penelitian_deteksi_postur.md
│
├── 05_naskah_dan_publikasi/                   <-- Naskah Publikasi & Outline Artikel Ilmiah
│   ├── README.md                              (Panduan penulisan naskah publikasi)
│   └── OUTLINE_ARTIKEL_ILMIAH_POSTUR.md       <-- ⭐ Outline lengkap artikel jurnal/prosiding internasional
│
└── methodology/                               <-- Direktori migrasi lama (dipertahankan untuk kompatibilitas)
    └── README.md                              (Peta migrasi file lama ke lokasi baru)
```

---

## 🧭 Panduan Akses Cepat Berdasarkan Kebutuhan

| Kebutuhan Anda | Dokumen Rekomendasi | Lokasi File |
|---|---|---|
| **Melihat hasil akhir perbandingan 2D vs 3D** | Laporan Evaluasi Komprehensif Final | [`01_laporan_evaluasi_dan_progres/laporan_progress_stereo_3d_dan_evaluasi_final.md`](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/laporan_progress_stereo_3d_dan_evaluasi_final.md) |
| **Melihat hasil pengujian sistem kamera live** | Laporan Progres Pengujian Live (93.1%) | [`01_laporan_evaluasi_dan_progres/LAPORAN_PROGRES_TA_20260922.md`](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/LAPORAN_PROGRES_TA_20260922.md) |
| **Mengetahui spesifikasi dataset 24 subjek** | Laporan QC Final 24 Subjek | [`02_laporan_qc_dataset/laporan_qc_dataset_keseluruhan_24_subjek.md`](file:///d:/.Candra/Project/TA/08_documents/02_laporan_qc_dataset/laporan_qc_dataset_keseluruhan_24_subjek.md) |
| **Memahami SOP pengambilan data subjek** | Panduan SOP Pengumpulan Dataset | [`03_panduan_dan_sop/Panduan_Pengumpulan_Dataset_Privat_Postur_Duduk.md`](file:///d:/.Candra/Project/TA/08_documents/03_panduan_dan_sop/Panduan_Pengumpulan_Dataset_Privat_Postur_Duduk.md) |
| **Memahami kode dan arsitektur pipeline** | Panduan Implementasi Pipeline Final | [`03_panduan_dan_sop/PANDUAN_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md`](file:///d:/.Candra/Project/TA/08_documents/03_panduan_dan_sop/PANDUAN_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md) |
| **Mengecek roadmap & task penyelesaian** | Rencana & Tracker Stereo 3D Final | [`04_perencanaan_dan_task/RENCANA_IMPLEMENTASI_FINAL_SETELAH_EVALUASI_3D.md`](file:///d:/.Candra/Project/TA/08_documents/04_perencanaan_dan_task/RENCANA_IMPLEMENTASI_FINAL_SETELAH_EVALUASI_3D.md) |
| **Menulis naskah skripsi / artikel jurnal** | Outline Komprehensif Artikel Ilmiah | [`05_naskah_dan_publikasi/OUTLINE_ARTIKEL_ILMIAH_POSTUR.md`](file:///d:/.Candra/Project/TA/08_documents/05_naskah_dan_publikasi/OUTLINE_ARTIKEL_ILMIAH_POSTUR.md) |

---

## 📊 Ringkasan Hasil Eksperimen Kunci (*Executive Summary*)

Berikut ringkasan angka performa akhir yang telah divalidasi dengan protokol ketat (*Subject-Aware 5-Fold Cross-Validation*, zero subject leakage):

### 1. Perbandingan Model (*Fair Comparison* pada 403 Sampel Irisan Identik)

| Model | Fitur | Pooled Macro F1 | Fold Mean Macro F1 | Fold 3 F1 (Uji Rig CAL_004) | Status Performa |
|---|:---:|:---:|:---:|:---:|:---:|
| **2D Core Baseline** | 36f | **0.6863** | 0.6669 ± 0.0846 | 0.6187 | Stabil di seluruh rig |
| **2D Best Practical** | 42f (+telinga) | **0.7039** | **0.6875 ± 0.0701** | 0.6187 | **Performa tertinggi & paling konsisten** |
| **3D Current (Camera Frame)** | 25f | 0.6182 | 0.6046 ± 0.2519 | 0.1052 | Unggul di fold seragam, collapse di Fold 3 |
| **3D Canonical (Body Frame)** | 22f | 0.3259 | 0.3237 ± 0.1241 | 0.1246 | Menghilangkan rig shift tapi kehilangan sinyal |

### 2. Efisiensi Komputasi & Inferensi Real-Time (Task 3D-16 CPU Benchmark)

- **Total Latency End-to-End:** **191.50 ms** (~**5.2 FPS** pada CPU standar)
- **Komposisi Beban:**
  - Estimasi Pose YOLO (CAM01 + CAM02): **188.09 ms (98.2% latency)**
  - Triangulasi Stereo 3D + QC: **0.72 ms (<0.5%)**
  - Ekstraksi Fitur 3D: **0.27 ms (<0.2%)**
  - Klasifikasi XGBoost: **2.40 ms (~1.2%)**
- **Kesimpulan:** Seluruh komponen 3D non-YOLO hanya memerlukan **3.4 ms**. Bottleneck utama murni berada pada inferensi detektor CNN YOLOv8n-Pose.

---

## 🔍 Detail Setiap Kategori Dokumen

### 1. `01_laporan_evaluasi_dan_progres/`
Berisi seluruh laporan berkala, evaluasi pipeline, dan analisis kausalitas:
- [laporan_progress_stereo_3d_dan_evaluasi_final.md](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/laporan_progress_stereo_3d_dan_evaluasi_final.md): Laporan evaluasi final komprehensif (9 bagian) mencakup investigasi numerik Fold 3, audit integritas data, eksperimen canonicalization (FIX-01 s/d FIX-06), uji raw stereo pair (3D-15), dan latency benchmark (3D-16).
- [LAPORAN_PROGRES_TA_20260922.md](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/LAPORAN_PROGRES_TA_20260922.md): Laporan progres pengujian live streaming kamera fisik CAM01 & CAM02, keberhasilan membedakan slouching vs leaning forward (konfusi turun dari 26.1% ke 3.6%), dan lonjakan akurasi live ke 93.1%.
- [laporan_progres_komprehensif_penelitian.md](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/laporan_progres_komprehensif_penelitian.md): Laporan progres menyeluruh fase transisi sebelum pengujian 3D intensif.
- [laporan_akhir_pipeline_final_skripsi.md](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/laporan_akhir_pipeline_final_skripsi.md): Laporan evaluasi formal pipeline skripsi versi awal (uji signifikansi McNemar dan t-test).
- [laporan_hasil_eksperimen_holistik.md](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/laporan_hasil_eksperimen_holistik.md): Hasil evaluasi holistik model terdahulu.
- [laporan_pengecekan_dan_kalibrasi_kamera.md](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/laporan_pengecekan_dan_kalibrasi_kamera.md): Evaluasi visual dan verifikasi kualitas kalibrasi kamera stereo.
- [laporan_fase_5c_target_person_selection_dan_qc.md](file:///d:/.Candra/Project/TA/08_documents/01_laporan_evaluasi_dan_progres/laporan_fase_5c_target_person_selection_dan_qc.md): Logika seleksi subjek utama (*target person selection*) dan gerbang QC.

### 2. `02_laporan_qc_dataset/`
Berisi dokumentasi kontrol kualitas data tangkapan subjek riil dari pilot hingga subjek ke-24:
- [laporan_qc_dataset_keseluruhan_24_subjek.md](file:///d:/.Candra/Project/TA/08_documents/02_laporan_qc_dataset/laporan_qc_dataset_keseluruhan_24_subjek.md): **Laporan Final Kualitas Dataset**. Merinci audit terhadap 727 tangkapan postur 6-kelas, reprojection error kalibrasi rig, cakupan sendi panggul kanan, dan penetapan 403 sampel *Model-Ready 3D*.
- `arsip_historis_qc/`: Menyimpan riwayat bertahap pengumpulan dataset untuk auditabilitas akademik (pilot, 6 subjek, 10 subjek, 21 subjek, dan 23 subjek).

### 3. `03_panduan_dan_sop/`
Berisi panduan prosedural dan teknis:
- [Panduan_Pengumpulan_Dataset_Privat_Postur_Duduk.md](file:///d:/.Candra/Project/TA/08_documents/03_panduan_dan_sop/Panduan_Pengumpulan_Dataset_Privat_Postur_Duduk.md): SOP baku pengambilan data subjek di laboratorium (jarak kamera, ketinggian meja/kursi, pencahayaan, variasi baju, dan urutan 8 postur).
- [PANDUAN_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md](file:///d:/.Candra/Project/TA/08_documents/03_panduan_dan_sop/PANDUAN_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md): Dokumen arsitektur teknis pipeline 2D multi-view dan stereo 3D, definisi skema fitur, normalisasi, dan integrasi XGBoost.
- [panduan_eksekusi_training_pc_gpu.md](file:///d:/.Candra/Project/TA/08_documents/03_panduan_dan_sop/panduan_eksekusi_training_pc_gpu.md): Panduan menjalankan pipeline training dan eksperimen pada komputer dengan akselerasi GPU.

### 4. `04_perencanaan_dan_task/`
Berisi perencanaan strategis, peta jalan penelitian, dan daftar periksa implementasi:
- [RENCANA_IMPLEMENTASI_FINAL_SETELAH_EVALUASI_3D.md](file:///d:/.Candra/Project/TA/08_documents/04_perencanaan_dan_task/RENCANA_IMPLEMENTASI_FINAL_SETELAH_EVALUASI_3D.md): Rencana aksi sistematis perbaikan stereo 3D (Fase 1: Canonicalization FIX-01 s/d FIX-06, Fase 2: Smoke test 3D-15 & latency 3D-16, Fase 3: Penulisan artikel 3D-17).
- [TASK_PENYELESAIAN_STEREO_3D_FINAL.md](file:///d:/.Candra/Project/TA/08_documents/04_perencanaan_dan_task/TASK_PENYELESAIAN_STEREO_3D_FINAL.md): Checklist task stereo 3D (Task 3D-01 s/d 3D-17) lengkap dengan justifikasi metodologis.
- [TASK_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md](file:///d:/.Candra/Project/TA/08_documents/04_perencanaan_dan_task/TASK_IMPLEMENTASI_PIPELINE_FINAL_2D_3D_XGBOOST.md): Checklist pelaksanaan pipeline XGBoost 2D dan 3D.
- [rencana_penelitian_postur_duduk_revisi.md](file:///d:/.Candra/Project/TA/08_documents/04_perencanaan_dan_task/rencana_penelitian_postur_duduk_revisi.md): Dokumen perencanaan awal riset revisi.
- [pipeline_penelitian_deteksi_postur.md](file:///d:/.Candra/Project/TA/08_documents/04_perencanaan_dan_task/pipeline_penelitian_deteksi_postur.md): Konsep alur pipeline awal penelitian.

### 5. `05_naskah_dan_publikasi/`
Berisi bahan penulisan publikasi ilmiah dan skripsi:
- [OUTLINE_ARTIKEL_ILMIAH_POSTUR.md](file:///d:/.Candra/Project/TA/08_documents/05_naskah_dan_publikasi/OUTLINE_ARTIKEL_ILMIAH_POSTUR.md): Outline komprehensif artikel ilmiah dengan struktur IMRaD (Introduction, Related Work, Proposed Methodology, Experimental Setup, Results & In-Depth Discussion, Conclusion & Future Work), dilengkapi landasan biomekanik (CVA, Plumb Line), perumusan matematika fitur, dan tabel-tabel hasil pengujian.

---

## 🏷️ Catatan Pemeliharaan Dokumen

1. **Integritas Link Internal:** Semua dokumen dalam folder ini menggunakan tautan relatif yang saling terhubung antar folder.
2. **Keterikatan Metadata:** Keputusan taksonomi kelas (`03_metadata/taxonomy_kelas.md`) dan keputusan dataset (`03_metadata/dataset_decisions.md`) tetap berada di folder metadata pusat repositori.
3. **Hasil Eksperimen Numerik:** File CSV dan JSON hasil pelatihan/evaluasi tetap tersimpan aman di `07_results/experiments/private_final/`.
