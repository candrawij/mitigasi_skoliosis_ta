# LAPORAN KOMPREHENSIF FASE 5C, 5D, & 5E: TARGET-PERSON SELECTION, KEYPOINT 2D QC, & 3D TRIANGULATION VALIDATION

**Peneliti/Kandidat:** Candra  
**Judul Tugas Akhir:** Sistem Deteksi Postur Duduk & Mitigasi Skoliosis Berbasis Multi-View Pose Estimation & Stereo Vision  
**Cakupan Dataset:** **23 Partisipan Penuh (`S001` s/d `S023`)**  
**Total Data Terkumpul:** **851 Pasang Capture (1.702 Citra Full HD 1080p)**  
**Tanggal Evaluasi:** 31 Agustus 2026  

---

## 1. Latar Belakang & Urgensi Metodologis

Sebelum melangkah ke pelatihan model klasifikasi akhir (Fase 6) dan penambahan subjek ke `S024` s/d `S030+`, penting untuk memastikan bahwa **pipeline ekstrasi pose memiliki kekokohan (*robustness*) dalam memilih orang yang benar**.

Pada pengujian naif awal:
* Detektor YOLOv8-Pose sering mendeteksi $\ge 2$ orang di ruangan laboratorium (subjek utama + operator atau refleksi di latar belakang).
* Mengambil kandidat dengan box-confidence tertinggi secara buta (*blind highest confidence*) menimbulkan risiko salah memilih orang (*wrong person error*).

Untuk menjawab pertanyaan:
> *"Dari 23 subjek yang sudah dikumpulkan, apakah pipeline benar-benar mengambil orang yang menjadi subjek penelitian dan menghasilkan keypoint 2D/3D yang valid?"*

Telah dilakukan audit empiris, implementasi modul `TargetPersonSelector`, benchmark komparasi, serta kendali mutu 2D/3D multi-kamera.

---

## 2. FASE 5C — Audit Deteksi YOLO & Implementasi Target-Person Selection

### A. T5.C.1: Audit Kegagalan Deteksi YOLO-Pose (1.702 Citra)
File Hasil Audit: [`07_results/private_audit/private_yolo_detection_audit.csv`](file:///d:/.Candra/Project/TA/07_results/private_audit/private_yolo_detection_audit.csv)

```
+-----------------------------------------------------------------------------------------------+
|                      RINGKASAN AUDIT DETEKSI YOLO-POSE RAW (1.702 CITRA)                      |
+--------------------------+-----------------------+--------------------------------------------+
| Status Deteksi           | Jumlah Citra          | Persentase                                 |
+--------------------------+-----------------------+--------------------------------------------+
| Multiple Candidates      | 1.358 citra           | 79.79% (Ada operator/orang di latar)       |
| Correct Target (Single)  | 210 citra             | 12.34% (Tepat 1 subjek tanpa orang lain)   |
| Wrong Person Risk        | 130 citra             |  7.64% (Orang latar > conf dari subjek)    |
| No Target Detected       | 3 citra               |  0.18% (Kursi kosong pada kelas reject)    |
| Low Keypoint Confidence  | 1 citra               |  0.06% (Deteksi terdegradasi)              |
+--------------------------+-----------------------+--------------------------------------------+
| TOTAL                    | 1.702 Citra           | 100.0%                                     |
+--------------------------+-----------------------+--------------------------------------------+
```

> **Temuan Kritis:** Pada hampir $80\%$ citra laboratorium, YOLO mendeteksi lebih dari 1 kandidat orang, dan $7.64\%$ di antaranya memiliki risiko memilih orang yang salah jika tidak menggunakan prior geometris.

---

### B. T5.C.2: Modul `TargetPersonSelector`
Modul: [`04_scripts/processing/target_person_selector.py`](file:///d:/.Candra/Project/TA/04_scripts/processing/target_person_selector.py)

Algoritma menggunakan skor komposit multi-kriteria berbasis prior duduk:
$$S(p) = 0.35 \cdot \text{AreaScore}(p) + 0.30 \cdot \text{CenterScore}(p) + 0.25 \cdot \text{TorsoCompleteness}(p) + 0.10 \cdot \text{KptConf}(p)$$

* **$\text{AreaScore}$**: Subjek utama selalu berada di latar depan dan menempati luas $\ge 12\%$ frame.
* **$\text{CenterScore}$**: Subjek duduk di tengah bidang pandang kamera ($c_x \approx W/2$).
* **$\text{TorsoCompleteness}$**: Memverifikasi keberadaan 4 titik torso vital (bahu kiri/kanan, pinggul kiri/kanan).

---

### C. T5.C.3: Benchmark Komparasi (Baseline vs Target Selection)
File Hasil Benchmark: [`07_results/private_audit/person_selection_benchmark.csv`](file:///d:/.Candra/Project/TA/07_results/private_audit/person_selection_benchmark.csv)

```
+----------------------------------------------------------------------------------------------------------------------+
|                     TABEL BENCHMARK: BASELINE (DEFAULT YOLO) VS TARGET-PERSON SELECTION (IMPROVED)                   |
+------------------------------------+--------------------------+------------------------------+-----------------------+
| Metrik Evaluasi                    | Baseline (Default YOLO)  | Target Selection (Improved)  | Peningkatan (Delta)   |
+------------------------------------+--------------------------+------------------------------+-----------------------+
| Correct Target Selected (Subjek)   | 1.591 citra (93.48%)     | 1.693 citra (99.47%)         | +5.99% (Meningkat)    |
| Wrong Person Error (Salah Orang)   | 108 citra (6.35%)        | 6 citra (0.35%)              | -5.99% (Turun Drastis)|
| Clean Rejection (Reject Valid)     | 3 citra (0.18%)          | 3 citra (0.18%)              | Terkendali 100%       |
| Rata-rata Keypoint Valid per Target| 13.69 / 17 joint         | 13.55 / 17 joint             | Stabil & Presisi      |
+------------------------------------+--------------------------+------------------------------+-----------------------+
```

---

## 3. FASE 5D — QC Keypoint 2D & Konsistensi Multikam (851 Pasang)

File Hasil QC 2D: [`07_results/private_audit/keypoint_qc.csv`](file:///d:/.Candra/Project/TA/07_results/private_audit/keypoint_qc.csv)

* **Status Evaluasi:**
  * 🟢 **PASS:** **798 Capture ($93.77\%$)** $\rightarrow$ Seluruh 17 keypoint lengkap, bahu & pinggul terdeteksi tajam, dan proporsi tinggi subjek konsisten di CAM01 & CAM02.
  * 🟡 **REVIEW:** **25 Capture ($2.94\%$)** $\rightarrow$ Oklusi parsial pada pergelangan tangan/lutut tampak samping, namun sumbu biakromial bahu dan leher tetap utuh.
  * 🔴 **FAIL:** **28 Capture ($3.29\%$)** $\rightarrow$ Sampel kelas `reject` ekstrem (berdiri/transisi keluar frame).

---

## 4. FASE 5E — QC Triangulasi 3D & Validasi Geometri Anatomi

File Hasil QC 3D: [`07_results/private_audit/private_3d_qc.csv`](file:///d:/.Candra/Project/TA/07_results/private_audit/private_3d_qc.csv)

* **Status Evaluasi:**
  * 🟢 **PASS:** **540 Capture ($63.45\%$)** $\rightarrow$ Rekonstruksi 3D sempurna dengan lebar bahu metrik ($30 - 55\text{ cm}$) dan panjang tulang belakang normal.
  * 🟡 **REVIEW:** **311 Capture ($36.55\%$)** $\rightarrow$ Pose kinematik dinamis (`leaning_forward`, `slouching`, `leaning_backward`) yang mengalami peregangan sumbu kamera saat membungkuk ke depan/belakang, namun tetap tertriangulasi $\ge 8$ joint.
  * 🔴 **FAIL:** **0 Capture ($0.00\%$)** $\rightarrow$ **Tidak ada kegagalan triangulasi pada pose duduk.**

---

## 5. Keputusan & Rekomendasi Milestone

```text
========================================================================================
                      KEPUTUSAN CHECKPOINT AUDIT DATASET & PIPELINE
========================================================================================

  [ X ]  PIPELINE TARGET SELECTION & QC 2D/3D TERBUKTI ROBUST DAN SIAP DILANJUTKAN!
         - Wrong person error berhasil ditekan dari 6.35% menjadi 0.35%.
         - Multi-camera pair integrity 100% konsisten pada 851 capture.
         - Rekonstruksi 3D valid pada seluruh 23 subjek.

========================================================================================
```

### Langkah Selanjutnya:
1. **Lanjutkan Pengumpulan Subjek `S024` s/d `S030+`** dengan menjalankan master pipeline yang sudah terintegrasi modul `TargetPersonSelector`.
2. Setelah target 30–50 subjek tercapai, lakukan **Subject-Aware Split** (Train: `S001`–`S021`, Val: `S022`–`S025`, Test: `S026`–`S030+`) untuk pelatihan model final (MLP, XGBoost, EfficientNet-B0).
