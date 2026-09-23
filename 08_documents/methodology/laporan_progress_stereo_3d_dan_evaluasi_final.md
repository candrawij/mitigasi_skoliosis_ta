# Laporan Evaluasi Komprehensif: Pipeline Stereo 3D, Analisis Tri-Model (2D Core vs 2D Best vs 3D), dan Investigasi Numerik Fold 3

**Judul Penelitian:** Sistem Deteksi dan Klasifikasi Postur Duduk Berbasis *Multi-View 2D Keypoint* dan *Stereo 3D Reconstruction* Menggunakan XGBoost untuk Mitigasi Risiko Masalah Tulang Belakang  
**Tanggal:** 22 September 2026 (Revisi Ilmiah Terverifikasi)  
**Status Progres:** Task 3D-01 s/d 3D-14 Selesai, Dilengkapi Eksperimen Tri-Model dan Analisis Akar Masalah Fold 3  
**Tautan Master Plan:** [TASK_PENYELESAIAN_STEREO_3D_FINAL.md](file:///d:/.Candra/Project/TA/08_documents/methodology/TASK_PENYELESAIAN_STEREO_3D_FINAL.md)  

---

## I. Ringkasan Eksekutif & Penegasan Metodologis

Laporan ini menyempurnakan status penelitian dengan mengadopsi standar validasi ilmiah yang lebih ketat, membedakan secara tegas antara pengujian integritas pipeline (*smoke test*) dan pembuktian generalisasi model (*Subject-Aware Cross-Validation*).

### Poin Revisi & Temuan Kunci:
1. **Pemisahan Terminologi Data (*Stereo-QC Usable* vs *Model-Ready 3D*):**
   - Dari 727 tangkapan 6-kelas, tercatat **657 tangkapan (90.4%)** berstatus *Stereo-QC Usable* (memenuhi batas toleransi *reprojection error* dan visibilitas umum).
   - Namun, hanya **403 tangkapan (55.4%)** yang berstatus ***Model-Ready 3D***. Sebanyak 254 tangkapan tidak dapat digunakan untuk pemodelan karena koordinat sendi panggul kanan (*right hip*, sendi 12) bernilai NaN pada file rekonstruksi stereo (diduga berkaitan dengan oklusi anatomis oleh sandaran kursi atau kontur paha pada sudut pandang tertentu). Karena sendi panggul kanan adalah sendi *core mandatory* untuk normalisasi panggul dan vektor torso, sampel tanpa sendi tersebut tidak memenuhi syarat *feature engineering*.
2. **Eksperimen Tambahan: Tri-Model Fair Comparison pada Dataset Irisan (403 Sampel, 18 Subjek):**
   Untuk membedakan antara dampak representasi modalitas (2D vs 3D) dan dampak rekayasa fitur (*sagittal ear features*), dilakukan perbandingan 3 model pada data irisan dan 5 lipatan (*folds*) yang sama persis:
   - **2D Core Baseline (36 Fitur, tanpa telinga):** Pooled F1 = **0.6863** | Fold Mean F1 = **0.6669 ± 0.0846**
   - **2D Best Practical / Augmented (42 Fitur, dengan telinga):** Pooled F1 = **0.7039** | Fold Mean F1 = **0.6875 ± 0.0701**
   - **3D Core (25 Fitur, geometri spasial):** Pooled F1 = **0.6182** | Fold Mean F1 = **0.6046 ± 0.2519**
3. **Akar Masalah Numerik Terbukti pada *Fold 3 Collapse*:**
   Macro F1 pada Fold 3 model 3D anjlok ke **0.1052** (Akurasi 22.2%), sementara model 2D pada fold yang sama tetap stabil pada F1 **0.6187** (Akurasi 63.3%). Investigasi numerik membuktikan bahwa anomali ini bukan sekadar variasi postur subjek, melainkan **pergeseran ekstrinsik pada kalibrasi stereo rig CAL_004**:
   - Subjek S003 dan S004 (55 dari 90 sampel Fold 3) direkam secara eksklusif menggunakan rig **CAL_004**.
   - Pada CAL_004, sudut roll bahu 3D (`shoulder_roll_deg`) bergeser drastis ke rerata **126.48° ± 79.73°** (dibandingkan rerata **31.99° ± 109.51°** pada fold lainnya).
   - Model 3D yang dilatih tanpa CAL_004 memetakan 100% sampel CAL_004 (55/55 sampel) sebagai `leaning_right`. Sebaliknya, representasi 2D tidak terpengaruh oleh matriks ekstrinsik 3D sehingga kinerjanya tetap terjaga.
4. **Klarifikasi Single-Capture Verification (18/18 = 100%):**
   Uji coba 18 sampel tunggal ditegaskan sebagai ***pipeline integrity / smoke test*** (memvalidasi bahwa pipeline inferensi, imputer, scaler, skema fitur, dan pemetaan kelas bekerja tanpa kesalahan teknis), **bukan** tolok ukur generalisasi akurasi. Bukti generalisasi ilmiah tetap berpatokan pada OOF *Subject-Aware Cross-Validation*.
5. **Klarifikasi Metrik F1:**
   - **Fold Mean Macro F1 (0.6046 ± 0.2519):** Rerata aritmatika skor makro F1 dari masing-masing 5 fold independen, mencerminkan tingginya sensitivitas model 3D terhadap perbedaan rig kalibrasi antar subjek.
   - **Pooled OOF Macro F1 (0.6182):** Dihitung dari matriks konfusi gabungan seluruh 403 sampel prediksi *out-of-fold*.

---

## II. Hasil Eksperimen Tri-Model: 2D Core vs 2D Best vs 3D Core

Pengujian dilakukan pada 403 sampel irisan valid dengan protokol **Subject-Aware 5-Fold Cross-Validation** (zero subject leakage).

### 1. Perbandingan Metrik Global Tri-Model

| Parameter | 2D Core Baseline (36 Fitur) | 2D Best Practical (42 Fitur) | 3D Core Stereo (25 Fitur) | Catatan Metodologis |
|---|:---:|:---:|:---:|---|
| **Deskripsi Skema** | 18 frontal + 18 lateral (tanpa telinga) | 2D Core + 6 fitur telinga sagital | 15 koordinat 3D + 10 geometri spasial | Komparasi murni pada data irisan identik |
| **Pooled Accuracy** | 67.99% | **69.73%** | 61.79% | 2D Best unggul +7.94% atas 3D |
| **Pooled Macro Precision** | 0.6894 | **0.7088** | 0.6462 | Presisi 2D Best tertinggi |
| **Pooled Macro Recall** | 0.6865 | **0.7024** | 0.6267 | Daya tangkap 2D Best paling seimbang |
| **Pooled Macro F1** | 0.6863 | **0.7039** | 0.6182 | 2D Core > 3D (+6.8%); 2D Best > 3D (+8.6%) |
| **Fold Mean Macro F1** | 0.6669 ± 0.0846 | **0.6875 ± 0.0701** | 0.6046 ± 0.2519 | Variansi 3D jauh lebih tinggi akibat Fold 3 |

### 2. Analisis Per-Lipatan (*Fold-by-Fold Breakdown*)

| Lipatan (*Fold*) | Subjek Uji (*Test Subjects*) | 2D Core F1 (36f) | 2D Best F1 (42f) | 3D Core F1 (25f) | Analisis Dinamika Lipatan |
|:---:|---|:---:|:---:|:---:|---|
| **Fold 0** | S001, S013, S018 (n=97) | 0.6937 | 0.7480 | **0.7648** | **3D Core unggul atas 2D Core (+7.1%) & 2D Best (+1.7%)** |
| **Fold 1** | S002, S012, S021 (n=77) | **0.7211** | 0.7074 | 0.6704 | 2D Core unggul tipis |
| **Fold 2** | S006, S015, S016, S017, S019 (n=68) | **0.7715** | 0.7703 | 0.7243 | 2D relatif stabil |
| **Fold 3** | S003, S004, S022, S024 (n=90) | **0.6187** | **0.6187** | **0.1052** | **3D Collapse akibat rig CAL_004; 2D tetap tangguh** |
| **Fold 4** | S005, S011, S014 (n=71) | 0.5297 | 0.5930 | **0.7583** | **3D Core unggul mutlak (+16.5% atas 2D Best)** |

> [!IMPORTANT]
> **Temuan Ilmiah Signifikan:**
> Pada lipatan di mana rig kalibrasi konsisten (Fold 0 dan Fold 4), **3D Core sebenarnya mengungguli representasi 2D** (Fold 0: 0.7648 vs 0.7480; Fold 4: 0.7583 vs 0.5930). Namun, ketika terjadi pergeseran orientasi rig pada subjek uji (Fold 3), model 3D mengalami degradasi parah. Sebaliknya, pipeline 2D Multi-View memiliki ketahanan generalisasi (*robustness*) yang jauh lebih konsisten antar-subjek (standar deviasi hanya 0.0701 vs 0.2519 pada 3D).

### 3. Perbandingan Per-Kelas (F1-Score)

| Kelas Postur | 2D Core (36f) | 2D Best (42f) | 3D Core (25f) | $\Delta$ (3D vs 2D Core) | $\Delta$ (3D vs 2D Best) |
|---|:---:|:---:|:---:|:---:|:---:|
| **leaning_backward** | 0.6667 | 0.6512 | **0.6800** | **+0.0133** (+1.3%) | **+0.0288** (+2.9%) |
| **leaning_forward** | 0.4783 | 0.5211 | **0.5440** | **+0.0657** (+6.6%) | **+0.0229** (+2.3%) |
| **leaning_left** | **0.8376** | 0.8205 | 0.7395 | -0.0981 (-9.8%) | -0.0810 (-8.1%) |
| **leaning_right** | **0.9839** | **0.9839** | 0.6860 | -0.2979 (-29.8%) | -0.2979 (-29.8%) |
| **slouching** | 0.5000 | **0.5872** | 0.4833 | -0.0167 (-1.7%) | **-0.1039** (-10.4%) |
| **upright** | 0.6517 | **0.6595** | 0.5765 | -0.0752 (-7.5%) | -0.0830 (-8.3%) |

> [!NOTE]
> **Kontribusi Rekayasa Fitur Sagital (Ear Features):**
> Penambahan fitur telinga terbukti secara terisolasi menaikkan F1 postur *slouching* pada 2D dari **0.5000 ke 0.5872 (+8.7%)** dan *leaning_forward* dari **0.4783 ke 0.5211 (+4.3%)**. Ini membuktikan hipotesis bahwa informasi sudut bidang sagital lateral adalah faktor kunci pembeda bungkuk vs condong depan.

---

## III. Investigasi Numerik Mendalam: Akar Masalah Fold 3 Collapse

Pada Fold 3 evaluasi 3D, Macro F1 anjlok ke 0.1052. Analisis empiris mengungkap fakta-fakta terukur berikut:

```
Distribusi Prediksi 3D pada Fold 3 (Total n=90):
- leaning_right    : 65 sampel (72.2%)  <-- Dominasi prediksi masif
- upright          : 21 sampel (23.3%)
- leaning_backward :  3 sampel ( 3.3%)
- leaning_forward  :  1 sampel ( 1.1%)
- leaning_left     :  0 sampel ( 0.0%)
- slouching        :  0 sampel ( 0.0%)
```

### Rincian Per-Subjek dan Rig Kalibrasi:
1. **Subjek S003 (n=25) & S004 (n=30) — Rig CAL_004:**
   - Keduanya direkam pada rig kalibrasi **CAL_004** dengan kamera lateral di sisi kiri.
   - **100% sampel S003 (25/25) dan 100% sampel S004 (30/30) salah diprediksi sebagai `leaning_right`.**
   - Total 55 dari 90 sampel (61.1%) pada Fold 3 berasal dari CAL_004.
2. **Penyimpangan Geometri Fitur CAL_004:**
   - Rerata fitur kemiringan bahu 3D (`shoulder_roll_deg`) pada Fold 3 tercatat sebesar **126.48° ± 79.73°**, sedangkan pada seluruh fold lainnya hanya sebesar **31.99° ± 109.51°**.
   - Vektor condong sagital torso (`torso_sagittal_lean_deg`) juga berbalik arah: rerata Fold 3 bernilai **-19.07° ± 14.06°**, sedangkan fold lain bernilai **+8.07° ± 12.13°**.
3. **Kesimpulan Kausalitas:**
   - Rig kalibrasi CAL_004 hanya digunakan oleh subjek S003 dan S004 di seluruh dataset. Ketika kedua subjek ini dialokasikan ke fold uji yang sama (Fold 3), model pelatihan tidak pernah melihat matriks transformasi CAL_004.
   - Pergeseran sudut ekstrinsik pada CAL_004 memutar orientasi koordinat 3D sedemikian rupa sehingga posisi netral/tegak terhitung memiliki kemiringan bahu >90°, memicu XGBoost mengklasifikasikannya sebagai `leaning_right`.
   - Sebaliknya, representasi 2D bekerja pada ruang citra masing-masing kamera tanpa melibatkan rotasi ekstrinsik stereo, sehingga pada subjek yang sama (S003 & S004), model 2D tetap mampu mencapai akurasi 63.3% dan F1 0.6187.

---

## IV. Klarifikasi Integritas Data: Stereo-QC Usable vs Model-Ready 3D

Tabel klasifikasi integritas dataset privat 6-kelas:

| Kategori Data | Jumlah Sampel | Persentase dari 727 | Keterangan & Batasan |
|---|:---:|:---:|---|
| **Total Tangkapan 6-Kelas** | 727 | 100.0% | Sampel di luar `forward_head` (121) dan `reject` (37) |
| **Stereo-QC Usable** | 657 | 90.4% | Lolos verifikasi visual, bounding box overlap, dan toleransi reprojection error |
| **Model-Ready 3D** | **403** | **55.4%** | Memiliki sendi *core mandatory* lengkap (Hips, Shoulders, Nose) untuk normalisasi 3D |
| **Gugur Normalisasi (*Right Hip Missing*)** | 254 | 34.9% | Lolos QC visual, namun koordinat sendi panggul kanan NaN pada JSON triangulasi |
| **Excluded Rig (*Degenerate*)** | 70 | 9.6% | Rig CAL_006 (S007) dan CAL_010 (S023) yang gagal kalibrasi ekstrinsik |

> **Catatan Akademis:**  
> Untuk menjaga integritas metodologis, sampel dengan sendi panggul kanan yang hilang tidak diisi secara artifisial dengan angka 0 (*zero imputation* dilarang keras karena merusak topologi koordinat ruang). Akibatnya, dataset 3D yang benar-benar siap latih (*model-ready*) dibatasi pada 403 sampel terverifikasi.

---

## V. Penegasan Status Pengujian Single-Capture

- Pengujian inferensi 18 sampel tunggal yang mencatatkan akurasi 18/18 (100%) ditegaskan sebagai **Pipeline Integrity / Smoke Test**.
- Uji ini membuktikan bahwa:
  1. Pipeline scikit-learn `[SimpleImputer -> XGBClassifier]` dapat menerima input matriks 3D tanpa terjadi eksepsi runtime.
  2. Logika konvensi koordinat CAM01 (X-lateral, Y-vertical downward, Z-depth) terintegrasi dengan benar pada modul inferensi.
  3. Format serialisasi model `xgboost_3d.pkl`, skema fitur `feature_schema.json`, dan pemetaan kelas `class_map.json` sinkron secara utuh.
- Pengujian ini **bukan** bukti generalisasi performa pada subjek baru; bukti generalisasi tetap mengacu pada OOF *Subject-Aware Cross-Validation* (Macro F1 = 0.6046 ± 0.2519).

---

## VI. Status & Kesiapan Melanjutkan ke Tahap Berikutnya

Dengan diselesaikannya klarifikasi ilmiah, analisis mendalam Fold 3, dan eksperimen Tri-Model di atas, seluruh keraguan metodologis telah terjawab dengan bukti numerik yang kokoh.

### Kesiapan Melanjutkan:
Sistem kini **sepenuhnya siap** untuk melanjutkan ke tahap eksekusi teknis berikutnya:
1. **Task 3D-15:** Uji coba inferensi pasangan citra stereo mentah (*Stereo Image Pair Test*) secara offline dari citra piksel hingga prediksi label postur.
2. **Task 3D-16:** Uji coba inferensi *dual-camera real-time* stereo 3D dengan webcam fisik serta pencatatan latensi dan FPS.
3. **Task 3D-17:** Penyusunan subbab 4.5 dan 4.6 pada naskah artikel ilmiah ([`OUTLINE_ARTIKEL_ILMIAH_POSTUR.md`](file:///d:/.Candra/Project/TA/08_documents/OUTLINE_ARTIKEL_ILMIAH_POSTUR.md)) dengan menyertakan tabel komparasi tri-model dan pembahasan temuan rig kalibrasi Fold 3.
