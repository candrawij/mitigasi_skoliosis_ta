# Panduan Lengkap Pengumpulan Dataset Privat Postur Duduk Berbasis Citra dan 3D Keypoint

## 1. Tujuan Dokumen

Dokumen ini menjadi panduan operasional untuk merancang, mengumpulkan, mengelola, mengaudit, dan menyiapkan dataset privat untuk penelitian:

> **Klasifikasi postur duduk berbasis citra RGB dan representasi keypoint 2D/3D untuk mendukung pemantauan serta mitigasi kebiasaan postur tidak ergonomis dan postur asimetris.**

Dataset dirancang agar dapat digunakan untuk:

1. klasifikasi beberapa jenis postur duduk;
2. estimasi keypoint 2D;
3. rekonstruksi atau estimasi keypoint 3D;
4. perbandingan model berbasis RGB, keypoint 2D, dan keypoint 3D;
5. pengembangan sistem feedback korektif;
6. evaluasi generalisasi antar-subjek.

Dataset **tidak dirancang untuk** diagnosis skoliosis, pengukuran Cobb angle, penentuan tingkat keparahan skoliosis, pengganti pemeriksaan dokter/fisioterapis, atau klaim klinis pencegahan skoliosis.

Klaim yang aman:

> Sistem membantu memantau postur duduk tidak ergonomis dan postur asimetris yang berkaitan dengan kesehatan muskuloskeletal dan tulang belakang.

---

# 2. Keputusan Desain Utama

## 2.1 Tugas utama

Tugas utama penelitian adalah:

> **Multi-class sitting posture classification.**

Pose estimation merupakan tahap ekstraksi representasi tubuh, sedangkan output akhir model berupa kelas postur.

```text
Citra RGB
→ Deteksi orang
→ Estimasi keypoint 2D
→ Rekonstruksi/estimasi keypoint 3D
→ Normalisasi skeleton
→ Klasifikasi postur
→ Penjelasan geometris
→ Feedback korektif
```

## 2.2 Posisi duduk dan berdiri

Fokus inti dataset adalah **postur duduk**. Posisi berdiri dikumpulkan dalam jumlah terbatas sebagai kelas penolak, misalnya `standing`, `not_seated`, dan `no_person`. Posisi berdiri tidak dijadikan kelas postur utama karena akan mengubah tugas menjadi pengenalan aktivitas umum.

---

# 3. Jenis Dataset 3D

## 3.1 Opsi utama: multi-camera RGB terkalibrasi

```text
Beberapa kamera RGB sinkron
→ Keypoint 2D setiap view
→ Kalibrasi kamera
→ Triangulasi
→ Keypoint 3D
```

| Jumlah kamera | Status | Catatan |
|---:|---|---|
| 1 | Tidak cukup untuk true 3D | Hanya menghasilkan pseudo-3D |
| 2 | Minimum | Dapat digunakan untuk triangulasi |
| 3 | Direkomendasikan | Lebih tahan terhadap occlusion |
| 4+ | Opsional | Lebih kompleks dan mahal |

Penempatan yang direkomendasikan:

- **CAM01:** frontal;
- **CAM02:** lateral 90°;
- **CAM03:** diagonal 45°.

## 3.2 Alternatif: kamera RGB-D

RGB-D dapat menghasilkan informasi kedalaman, tetapi membutuhkan perangkat khusus, hasil tidak lagi murni RGB, kualitas depth tetap perlu diaudit, dan perangkat harus dikalibrasi.

## 3.3 Alternatif: pseudo-3D monocular

```text
Satu citra RGB
→ Estimator pose 2D
→ Model 2D-to-3D
→ Estimated 3D keypoints
```

Gunakan istilah `estimated 3D keypoints`, `predicted 3D pose`, atau `pseudo-3D representation`. Jangan menyebut hasil monocular sebagai *ground-truth 3D*.

---

# 4. Struktur Label Postur

## 4.1 Kelas inti single-label

| ID | Nama kelas | Definisi operasional |
|---:|---|---|
| 0 | `upright` | Kepala, bahu, torso, dan panggul relatif netral |
| 1 | `leaning_forward` | Torso condong ke depan sebagai satu kesatuan |
| 2 | `leaning_backward` | Torso condong atau bersandar ke belakang |
| 3 | `leaning_left` | Torso condong ke sisi kiri subjek |
| 4 | `leaning_right` | Torso condong ke sisi kanan subjek |
| 5 | `slouching` | Punggung atas membulat, bahu maju, dan postur kehilangan tegak |
| 6 | `forward_head` | Kepala maju terhadap pusat bahu, torso relatif tetap |

Arah kiri dan kanan selalu mengikuti **sisi tubuh subjek**, bukan sisi pengamat.

## 4.2 Label tambahan multi-label

### Kepala dan leher

- `head_neutral`
- `head_tilt_left`
- `head_tilt_right`
- `head_forward`
- `head_rotation_left`
- `head_rotation_right`

### Bahu

- `shoulders_level`
- `left_shoulder_high`
- `right_shoulder_high`

### Panggul

- `pelvis_level`
- `left_pelvis_high`
- `right_pelvis_high`

### Torso

- `torso_neutral`
- `torso_forward`
- `torso_backward`
- `torso_left`
- `torso_right`
- `torso_rotation_left`
- `torso_rotation_right`

Contoh:

```text
primary_posture = leaning_left
head_state      = forward_head
shoulder_state  = left_shoulder_high
pelvis_state    = level
```

## 4.3 Kelas penolak

Jumlah disarankan sekitar 10–20% dari jumlah pose utama:

- `standing`
- `not_seated`
- `transition`
- `no_person`
- `severe_occlusion`
- `body_out_of_frame`

---

# 5. Definisi Operasional Kelas

## 5.1 Upright

- sumbu mid-shoulder ke mid-hip relatif vertikal;
- bahu relatif sejajar;
- kepala tidak terlalu maju;
- torso tidak condong lateral.

## 5.2 Leaning forward

- torso maju sebagai satu unit;
- mid-shoulder bergeser ke depan terhadap mid-hip;
- bukan hanya kepala yang menunduk;
- punggung tidak selalu membulat.

## 5.3 Leaning backward

- torso bergeser ke belakang;
- bukan hanya kepala menengadah;
- perubahan terjadi pada sumbu torso.

## 5.4 Leaning left dan right

- perubahan utama terjadi pada torso;
- garis bahu–panggul menunjukkan deviasi lateral;
- bukan hanya kepala miring atau menoleh.

## 5.5 Slouching

- punggung atas membulat;
- bahu maju/protracted;
- dada turun;
- berbeda dari leaning forward yang mempertahankan bentuk torso relatif lurus.

## 5.6 Forward head

- kepala maju terhadap pusat bahu;
- torso relatif lebih tegak dibanding `leaning_forward`;
- dapat dihitung dari perpindahan hidung/telinga terhadap mid-shoulder.

---

# 6. Skema Keypoint

## 6.1 Skema minimum: COCO-17

| Indeks | Keypoint |
|---:|---|
| 0 | Nose |
| 1 | Left eye |
| 2 | Right eye |
| 3 | Left ear |
| 4 | Right ear |
| 5 | Left shoulder |
| 6 | Right shoulder |
| 7 | Left elbow |
| 8 | Right elbow |
| 9 | Left wrist |
| 10 | Right wrist |
| 11 | Left hip |
| 12 | Right hip |
| 13 | Left knee |
| 14 | Right knee |
| 15 | Left ankle |
| 16 | Right ankle |

## 6.2 Keypoint penting

Prioritas tertinggi:

- nose;
- left/right ear;
- left/right shoulder;
- left/right hip;
- left/right knee.

Titik turunan:

```text
mid_shoulder = (left_shoulder + right_shoulder) / 2
mid_hip      = (left_hip + right_hip) / 2
```

Fitur geometris:

- shoulder tilt;
- pelvic tilt;
- trunk inclination;
- head displacement;
- shoulder–pelvis alignment;
- lateral torso angle;
- torso depth displacement;
- head–shoulder angle.

## 6.3 Visibility

| Nilai | Makna |
|---:|---|
| 0 | Tidak dianotasi/tidak tersedia |
| 1 | Teranotasi tetapi tertutup |
| 2 | Terlihat |

---

# 7. Sistem Koordinat 3D

```text
X = kiri–kanan
Y = vertikal
Z = depan–belakang
```

Origin dapat menggunakan titik tetap pada lantai atau pusat papan kalibrasi.

Simpan:

1. koordinat 3D mentah dalam meter;
2. koordinat 3D hasil normalisasi.

Normalisasi:

```text
origin   = mid_hip
scale    = shoulder width atau torso length
rotation = diselaraskan terhadap sumbu tubuh
```

Jangan menghapus koordinat mentah setelah normalisasi.

---

# 8. Kriteria Peserta

## 8.1 Jumlah subjek

| Tahap | Jumlah subjek | Tujuan |
|---|---:|---|
| Pilot minimum | 5 | Uji setup dan anotasi |
| Pilot ideal | 8–10 | Uji variasi tubuh dan kamera |
| Minimum penelitian utama | 20–30 | Evaluasi subject-wise awal |
| Target kuat | 30–50 | Generalisasi lebih baik |
| Sangat kuat | 50+ | Jika sumber daya memungkinkan |

Rekomendasi utama:

> **30 subjek sebagai batas minimum praktis, 40–50 subjek sebagai target ideal.**

## 8.2 Kriteria inklusi

- usia 18 tahun atau lebih;
- mampu duduk tanpa bantuan;
- memahami instruksi;
- bersedia direkam;
- tidak mengalami nyeri saat mengikuti sesi.

## 8.3 Kriteria penghentian

Pengambilan dihentikan jika peserta merasa nyeri, pusing, lelah, tidak nyaman, atau meminta berhenti.

## 8.4 Variasi peserta

Usahakan variasi pada tinggi badan, proporsi torso, bentuk tubuh, jenis kelamin, gaya rambut, penggunaan kacamata, serta pakaian longgar dan ketat. Jangan mengumpulkan atribut sensitif yang tidak diperlukan.

---

# 9. Jumlah Data yang Direkomendasikan

## 9.1 Pilot minimum

```text
5 subjek
× 7 kelas
× 3 pengulangan
× 3 kamera
= 315 citra
```

Unit pose independen:

```text
5 × 7 × 3 = 105 pose instance
```

## 9.2 Pilot ideal

```text
10 subjek
× 7 kelas
× 5 pengulangan
× 3 kamera
= 1.050 citra
```

Unit pose independen:

```text
10 × 7 × 5 = 350 pose instance
```

## 9.3 Dataset utama minimum

```text
30 subjek
× 2 sesi
× 7 kelas
× 5 pengulangan
= 2.100 pose instance
```

Dengan dua kamera:

```text
2.100 × 2 = 4.200 citra
```

Dengan tiga kamera:

```text
2.100 × 3 = 6.300 citra
```

## 9.4 Dataset utama ideal

```text
40 subjek
× 2 sesi
× 7 kelas
× 5 pengulangan
= 2.800 pose instance
```

Dengan tiga kamera:

```text
2.800 × 3 = 8.400 citra
```

## 9.5 Dataset kuat

```text
50 subjek
× 2 sesi
× 7 kelas
× 5 pengulangan
= 3.500 pose instance
```

Dengan tiga kamera:

```text
3.500 × 3 = 10.500 citra
```

## 9.6 Rejection class

Tambahkan sekitar 10–20% dari total pose instance utama. Untuk 2.100 pose utama, targetnya 210–420 pose rejection.

## 9.7 Catatan

Tiga kamera pada satu momen tetap dihitung sebagai **satu pose instance**, bukan tiga sampel independen. Jumlah subjek lebih penting daripada jumlah frame.

---

# 10. Pengulangan dan Sesi

## Pengulangan per kelas

| Pengulangan | Status |
|---:|---|
| 1–2 | Terlalu sedikit |
| 3 | Minimum pilot |
| 5 | Direkomendasikan |
| 8–10 | Opsional jika subjek sedikit |

## Jumlah sesi

| Sesi | Status |
|---:|---|
| 1 | Minimum |
| 2 | Direkomendasikan |
| 3 | Sangat baik tetapi lebih berat |

Dua sesi sebaiknya berbeda dalam hari, pakaian, pencahayaan ringan, posisi kecil kamera, dan kondisi natural peserta.

---

# 11. Perangkat dan Setup Kamera

## 11.1 Kamera

Kriteria minimum:

- resolusi 1920×1080;
- tripod stabil;
- fokus terkunci;
- exposure dikunci;
- tidak menggunakan zoom digital;
- frame rate konsisten jika menggunakan video untuk memilih frame statis.

## 11.2 Penempatan

- CAM01 frontal;
- CAM02 lateral;
- CAM03 diagonal 45°.

Target visibilitas:

- kepala terlihat penuh;
- kedua bahu terlihat;
- kedua pinggul terlihat;
- lutut terlihat;
- pergelangan kaki terlihat jika full-body;
- meja tidak menutup pinggul;
- sandaran kursi tidak menutup torso.

## 11.3 Kursi

Catat jenis kursi, tinggi dudukan, keberadaan sandaran, posisi kursi, dan jarak ke kamera. Untuk subset utama, gunakan kursi yang sama.

---

# 12. Kalibrasi Kamera

## 12.1 Intrinsik

- focal length;
- principal point;
- distortion coefficient.

## 12.2 Ekstrinsik

- rotasi;
- translasi;
- posisi relatif antar kamera.

## 12.3 Peralatan

Gunakan checkerboard atau Charuco board.

## 12.4 Waktu kalibrasi

Kalibrasi dilakukan sebelum pilot, setelah kamera/tripod dipindahkan, dan sebelum sesi baru jika setup berubah.

## 12.5 Target reprojection error

| Error | Interpretasi |
|---:|---|
| <1 px | Sangat baik |
| 1–2 px | Masih baik |
| 2–3 px | Perlu pemeriksaan |
| >3 px | Kalibrasi sebaiknya diulang |

Angka ini merupakan target internal praktis, bukan standar universal.

---

# 13. Sinkronisasi Kamera

Setiap pose memiliki satu `capture_id`:

```text
CAP000145
├── CAM01
├── CAM02
└── CAM03
```

Metode sinkronisasi:

1. satu remote trigger;
2. aplikasi kontrol kamera;
3. perangkat keras sinkron;
4. burst dengan timestamp;
5. peserta mempertahankan pose beberapa detik.

Target:

- ≥95% capture memiliki seluruh view;
- perbedaan waktu antarkamera serendah mungkin;
- pose tidak berubah selama pengambilan.

---

# 14. Prosedur Pengambilan Data

## 14.1 Sebelum sesi

1. Pastikan informed consent selesai.
2. Berikan `subject_id`.
3. Periksa kamera dan kalibrasi.
4. Pastikan seluruh keypoint terlihat.
5. Rekam pose neutral.
6. Jelaskan seluruh kelas.
7. Lakukan latihan.
8. Acak urutan kelas.

## 14.2 Saat sesi

1. Tampilkan contoh pose.
2. Peserta mengambil pose.
3. Tunggu 2–3 detik.
4. Ambil citra sinkron.
5. Periksa hasil secara cepat.
6. Ulangi sesuai jumlah repetisi.
7. Beri jeda bila perlu.

## 14.3 Setelah sesi

1. Periksa jumlah file.
2. Cek kecocokan view.
3. Tandai capture gagal.
4. Buat checksum.
5. Backup ke dua lokasi.
6. Jangan hapus file kamera sebelum backup terverifikasi.

---

# 15. Randomisasi Pengambilan

Jangan memakai urutan kelas yang sama untuk seluruh peserta. Gunakan urutan acak per peserta untuk mengurangi bias urutan, efek kelelahan, pola transisi, dan korelasi nama file dengan kelas.

---

# 16. Controlled dan Robustness Subset

## 16.1 Controlled subset

- kursi sama;
- kamera tetap;
- pencahayaan stabil;
- latar tetap;
- pose diarahkan;
- seluruh keypoint terlihat.

Target: 80–90% dataset utama.

## 16.2 Robustness subset

- kursi berbeda;
- latar berbeda;
- pakaian berbeda;
- pencahayaan berubah;
- occlusion ringan;
- kamera bergeser;
- jarak kamera berubah.

Target: 10–20% dataset.

---

# 17. Struktur Folder

```text
private_posture_dataset/
├── 00_docs/
│   ├── ethics/
│   ├── consent_forms/
│   └── pose_manual/
├── 01_calibration/
├── 02_raw/
│   ├── S001/
│   │   ├── SE01/
│   │   │   ├── CAM01/
│   │   │   ├── CAM02/
│   │   │   └── CAM03/
│   │   └── SE02/
│   └── S002/
├── 03_annotations/
│   ├── keypoints_2d/
│   ├── keypoints_3d/
│   └── posture_labels/
├── 04_metadata/
│   ├── participants.csv
│   ├── captures.csv
│   ├── images.csv
│   └── calibration_map.csv
├── 05_processed/
├── 06_splits/
├── 07_models/
└── 08_audit/
```

---

# 18. Penamaan File

Gunakan:

```text
S001_SE01_CAP000145_CAM01.jpg
```

Komponen:

- `S001`: subject;
- `SE01`: session;
- `CAP000145`: pose instance;
- `CAM01`: kamera.

Jangan menyimpan nama kelas pada nama file. Label ditempatkan pada metadata.

---

# 19. Metadata

## 19.1 participants.csv

```csv
subject_id,age_group,height_cm,session_count,consent_rgb,consent_public
S001,18-25,170,2,yes,no
```

## 19.2 captures.csv

```csv
capture_id,subject_id,session_id,calibration_id,primary_posture,head_state,shoulder_state,pelvis_state,repetition,subset,quality,notes
```

## 19.3 images.csv

```csv
image_id,capture_id,camera_id,image_path,timestamp,width,height,blur_score,exposure,annotation_status
```

## 19.4 calibration_map.csv

```csv
session_id,calibration_id,cam01_file,cam02_file,cam03_file
```

---

# 20. Anotasi Keypoint

## 20.1 Metode semi-otomatis

```text
Citra
→ Model pose pralatih
→ Prediksi keypoint
→ Koreksi manual
→ Anotasi final
```

Prediksi otomatis tidak langsung dianggap ground truth.

## 20.2 Langkah anotasi

1. Jalankan estimator COCO-17.
2. Visualisasikan keypoint.
3. Koreksi posisi.
4. Tandai visibility.
5. Triangulasikan 3D.
6. Proyeksikan kembali.
7. Periksa reprojection error.
8. Koreksi titik bermasalah.

## 20.3 Anotasi ganda

Audit 10–20% data dengan dua anotator. Periksa kesepakatan kelas, jarak keypoint, konflik visibility, dan pertukaran kiri–kanan.

---

# 21. Rekonstruksi Keypoint 3D

```text
Keypoint CAM01
+ Keypoint CAM02
+ Keypoint CAM03
→ Triangulasi
→ Keypoint 3D
→ Reprojection
→ Quality control
```

Titik dianggap invalid jika terlihat pada kurang dari dua kamera, reprojection error terlalu tinggi, berada di luar posisi anatomi masuk akal, atau salah pasangan kiri–kanan.

Contoh output:

```json
{
  "capture_id": "CAP000145",
  "unit": "meter",
  "coordinate_system": "CAL001_WORLD",
  "keypoints": {
    "left_shoulder": {
      "x": -0.18,
      "y": 1.21,
      "z": 0.42,
      "confidence": 0.97,
      "reprojection_error": 0.82
    }
  }
}
```

---

# 22. Audit Kualitas

## 22.1 Integritas file

- file rusak;
- file hilang;
- view tidak lengkap;
- timestamp tidak sesuai;
- duplikasi;
- resolusi tidak konsisten.

## 22.2 Audit label

- label salah;
- kelas ambigu;
- arah kiri–kanan tertukar;
- pose campuran;
- peserta gagal mengikuti instruksi.

## 22.3 Audit keypoint

- keypoint di luar tubuh;
- keypoint hilang;
- keypoint tertukar;
- visibility salah;
- triangulasi tidak stabil;
- reprojection error tinggi.

## 22.4 Audit distribusi

Hitung jumlah subjek, sesi, kelas, pose per kelas, view per capture, capture gagal, dan persentase occlusion.

---

# 23. Kriteria Kelulusan Dataset

## 23.1 Kriteria teknis

| Kriteria | Target |
|---|---:|
| File rusak | 0 |
| Capture tanpa pasangan view | <5% |
| Capture sinkron lengkap | ≥95% |
| Bahu terlihat | ≥95% |
| Pinggul terlihat | ≥95% |
| Keypoint utama valid | ≥90% |
| Capture perlu diulang | <10% |
| Label ambigu | <10% |
| Salah label setelah audit | <5% |
| Reprojection error median | ≤1–2 px |
| Duplikasi lintas split | 0 |

## 23.2 Kriteria metodologis

- seluruh kelas ada pada semua fold;
- split berdasarkan subjek;
- tidak ada subjek di train dan test;
- semua view satu capture tetap bersama;
- semua sesi satu subjek mengikuti split yang sama;
- jumlah subjek memadai;
- definisi kelas konsisten.

## 23.3 Kriteria penggunaan 3D

Dataset layak disebut memiliki true 3D bila kamera terkalibrasi, capture sinkron, keypoint 2D divalidasi, triangulasi dilakukan, sistem koordinat dijelaskan, unit disimpan, dan reprojection error dilaporkan.

---

# 24. Pembagian Data

## 24.1 Subject-wise split

Contoh 30 subjek:

| Split | Jumlah subjek | Proporsi |
|---|---:|---:|
| Train | 21 | 70% |
| Validation | 4–5 | 15% |
| Test | 4–5 | 15% |

Aturan:

- semua sesi satu subjek berada pada satu split;
- semua kamera satu capture berada pada satu split;
- semua augmentasi mengikuti split asal;
- tidak ada random image split.

## 24.2 Leave-One-Subject-Out

```text
Setiap fold:
1 subjek = test
subjek lain = train/validation
```

Laporkan `mean ± standard deviation`.

---

# 25. Augmentasi

Augmentasi dilakukan setelah split.

## Aman

- brightness ringan;
- contrast ringan;
- noise ringan;
- scale ringan;
- crop tanpa memotong keypoint.

## Perlu hati-hati

- rotation;
- perspective;
- occlusion;
- horizontal flip.

Horizontal flip wajib menukar:

```text
left_shoulder ↔ right_shoulder
left_hip      ↔ right_hip
leaning_left  ↔ leaning_right
head_tilt_left ↔ head_tilt_right
```

---

# 26. Pilot Study

## 26.1 Pilot minimum

```text
5 subjek
× 7 kelas
× 3 repetisi
× 3 kamera
= 315 citra
```

## 26.2 Pilot ideal

```text
10 subjek
× 7 kelas
× 5 repetisi
× 3 kamera
= 1.050 citra
```

## 26.3 Pertanyaan pilot

1. Apakah kamera sinkron?
2. Apakah semua keypoint terlihat?
3. Apakah pinggul tertutup kursi?
4. Apakah forward dan slouch dapat dibedakan?
5. Apakah triangulasi stabil?
6. Apakah peserta memahami kelas?
7. Berapa lama satu sesi?
8. Berapa banyak capture gagal?
9. Berapa lama anotasi?
10. Apakah baseline sederhana dapat membedakan kelas?

## 26.4 Keputusan pilot

Lanjut ke dataset utama hanya jika seluruh kelas dapat dibedakan, bahu dan pinggul terlihat, error triangulasi stabil, label ambigu <10%, capture gagal <10%, dan proses anotasi realistis.

---

# 27. Etika dan Privasi

Dokumen minimum:

1. lembar informasi peserta;
2. informed consent;
3. persetujuan penggunaan internal;
4. persetujuan publikasi citra;
5. persetujuan publikasi keypoint;
6. prosedur penarikan data;
7. prosedur penyimpanan;
8. persetujuan etik institusi bila diwajibkan.

Prinsip privasi:

- gunakan `subject_id`;
- jangan menyimpan nama dalam dataset;
- simpan file penghubung identitas secara terpisah;
- enkripsi data;
- batasi akses;
- tentukan masa penyimpanan;
- pertimbangkan hanya merilis keypoint anonim;
- peserta dapat menolak publikasi wajah.

---

# 28. Pipeline Eksperimen Setelah Dataset Siap

## Baseline RGB

```text
RGB image
→ CNN/MobileNet/EfficientNet
→ 7 kelas
```

## Baseline 2D

```text
2D keypoints
→ Normalisasi
→ MLP/XGBoost/GCN
→ 7 kelas
```

## Model 3D

```text
3D keypoints
→ Normalisasi
→ KeypointNet/GCN/MLP
→ 7 kelas
```

## Hybrid opsional

```text
RGB features + 3D keypoint features
→ Feature fusion
→ 7 kelas
```

Metrik klasifikasi:

- macro-F1;
- balanced accuracy;
- macro-recall;
- F1 per kelas;
- confusion matrix;
- mean ± standard deviation antar-subjek.

Metrik pose:

- 2D pose mAP/OKS;
- reprojection error;
- MPJPE;
- PA-MPJPE;
- persentase keypoint valid.

---

# 29. Checklist Pra-Pengumpulan

## Ilmiah

- [ ] Tujuan penelitian sudah dibekukan.
- [ ] Tujuh kelas sudah didefinisikan.
- [ ] Kelas penolak sudah ditentukan.
- [ ] Skema COCO-17 sudah dibekukan.
- [ ] Aturan kiri–kanan sudah jelas.
- [ ] Pipeline 2D dan 3D sudah dipilih.
- [ ] Rencana evaluasi subject-wise sudah dibuat.

## Etika

- [ ] Persetujuan etik sudah diperiksa.
- [ ] Informed consent selesai.
- [ ] Kebijakan publikasi wajah selesai.
- [ ] Prosedur penghapusan data tersedia.
- [ ] Sistem penyimpanan aman tersedia.

## Teknis

- [ ] Kamera tersedia.
- [ ] Tripod tersedia.
- [ ] Sinkronisasi tersedia.
- [ ] Checkerboard/Charuco tersedia.
- [ ] Kamera sudah dikalibrasi.
- [ ] Kursi dan posisi peserta ditandai.
- [ ] Struktur folder sudah dibuat.
- [ ] Backup tersedia.

## Pilot

- [ ] Lima sampai sepuluh peserta pilot tersedia.
- [ ] Semua kelas diuji.
- [ ] Bahu dan pinggul terlihat.
- [ ] Triangulasi berhasil.
- [ ] Reprojection error diperiksa.
- [ ] Waktu anotasi dihitung.
- [ ] Protokol direvisi.

---

# 30. Rekomendasi Final

## Paket minimum yang masih layak

```text
30 subjek
× 2 sesi
× 7 kelas
× 5 repetisi
× 2 kamera
= 4.200 citra
```

## Paket yang direkomendasikan

```text
30 subjek
× 2 sesi
× 7 kelas
× 5 repetisi
× 3 kamera
= 6.300 citra
```

## Paket ideal

```text
40–50 subjek
× 2 sesi
× 7 kelas
× 5 repetisi
× 3 kamera
= 8.400–10.500 citra
```

Tambahkan 10–20% rejection/robustness samples.

> Dataset privat yang paling kuat adalah dataset multi-view terkalibrasi dengan minimal 30 subjek, dua sesi, tujuh kelas postur duduk, lima pengulangan, tiga kamera, anotasi COCO-17, rekonstruksi keypoint 3D, dan evaluasi berbasis subjek.

Mulai dari pilot kecil. Jangan melakukan pengumpulan skala besar sebelum kalibrasi, sinkronisasi, definisi kelas, visibilitas pinggul, konsistensi anotasi, dan triangulasi 3D terbukti berhasil.
