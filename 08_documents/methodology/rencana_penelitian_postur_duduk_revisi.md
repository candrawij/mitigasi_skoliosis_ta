# Rencana Penelitian --- Postur Duduk (Revisi)

Dokumen ini adalah revisi rencana penelitian setelah pemeriksaan
`contact_sheets` dataset `sitting_posture_detection_initial`.

## 1. Kesimpulan audit contact sheet

Contact sheet menunjukkan bahwa `sitting_posture_detection_initial`
sangat heterogen. Terdapat foto orang duduk dengan berbagai kursi,
lingkungan, sudut, framing, dan kondisi postur. Selain foto yang relatif
bersih, terlihat pula gambar komposit/collage, gambar dengan lebih dari
satu orang, stock image/ilustrasi, framing yang tidak seragam, dan
contoh yang tidak selalu memberikan konteks postur tubuh secara lengkap.

**Implikasi:** dataset tidak sebaiknya langsung digunakan untuk training
final. Dataset perlu melalui audit, curation, deduplication, validasi
label, dan group-aware split.

## 2. Dataset dan peran

  -------------------------------------------------------------------------
  Dataset                               Peran yang disarankan
  ------------------------------------- -----------------------------------
  `project_design_20242025`             Image/frame baseline dan
                                        robustness; split berdasarkan
                                        video/source/person jika
                                        memungkinkan

  `Postureexercise`                     Keypoint baseline; mapping 7
                                        keypoint dan arti kelas wajib
                                        diverifikasi

  `IKORN`                               Image/pose experiment setelah
                                        penanganan near-duplicate; audit
                                        sebelumnya mencatat sekitar 655
                                        gambar dan sekitar 3551 pasangan
                                        near-duplicate

  `sitting_posture_detection_initial`   Curated image dataset dan
                                        eksperimen YOLO

  Dataset privat                        Controlled dataset dan
                                        final/generalization evaluation
  -------------------------------------------------------------------------

Dataset privat tetap penting, tetapi sebaiknya tidak langsung
dikumpulkan dalam jumlah besar sebelum empat dataset public selesai
diaudit.

## 3. Pipeline yang dikunci

``` text
Dataset
  ↓
Audit
  ↓
Curation
  ↓
Group-aware split
  ↓
Preprocessing
  ├── Image → CNN baseline
  ├── Keypoint → normalization → classifier
  └── Image → YOLO Pose → keypoint → classifier
  ↓
Evaluation
  ↓
Cross-dataset / private-dataset test
```

### Image baseline

`Image → preprocessing → EfficientNet-B0 pretrained ImageNet → posture class`

EfficientNet-B0 dipakai sebagai baseline utama karena relatif ringan dan
cocok untuk transfer learning pada dataset terbatas. ResNet-18/50 dapat
menjadi alternatif, tetapi tidak perlu memperbanyak arsitektur tanpa
alasan penelitian.

### Keypoint baseline

`Keypoint → validation → normalization → pose features → MLP/classifier → posture class`

### YOLO Pose

`Image → YOLO Pose → body keypoints → validation → normalization → feature extraction → classifier → posture class`

YOLO Pose menghasilkan lokasi/keypoint tubuh; **YOLO Pose bukan otomatis
label kelas postur penelitian**. Tahap klasifikasi tetap diperlukan.

## 4. Eksperimen

  ID       Eksperimen                        Status
  -------- --------------------------------- -----------------------------------
  EXP-01   EfficientNet-B0 image baseline    Wajib
  EXP-02   Keypoint + classifier             Wajib jika keypoint valid
  EXP-03   YOLO Pose + classifier            Wajib jika pose pipeline feasible
  EXP-04   Image + Pose fusion               Opsional
  EXP-05   Cross-dataset evaluation          Sangat disarankan
  EXP-06   Public → Private generalization   Sangat disarankan

## 5. Dataset `project_design_20242025`

Dataset berasal dari frame video dan banyak gambar
berkorespondensi/berurutan. Karena frame dari video yang sama tidak
independen, random image split berisiko data leakage.

Strategi:

``` text
video/source → group → train/validation/test
```

Bukan:

``` text
random frame → train/test
```

Dataset ini juga tidak ideal untuk full-body pose jika banyak gambar
hanya sampai bahu.

## 6. Dataset `Postureexercise`

Terdapat 7 keypoint dan kelas yang tercatat:

`nga_phai`, `nga_trai`, `nghieng_phai`, `nghieng_trai`, `thang`.

Nama kelas tampak berasal dari bahasa Vietnam, tetapi arti spesifiknya
**belum boleh diasumsikan**. Mapping indeks 0--6 ke bagian tubuh juga
perlu diverifikasi dari sumber asli.

Sebelum training:

1.  mapping 7 keypoint;
2.  arti setiap kelas;
3.  distribusi kelas;
4.  validitas keypoint;
5.  kesesuaian kelas dengan tujuan penelitian.

## 7. Dataset IKORN

Audit sebelumnya menunjukkan near-duplicate tinggi. Karena itu tidak
disarankan re-split agresif hanya untuk mengurangi jumlah data.

Strategi:

``` text
Original data
  ↓
Near-duplicate detection
  ↓
Duplicate clusters
  ↓
Representative selection
  ↓
Group-aware split
```

Tujuan utamanya adalah mencegah gambar yang hampir sama masuk ke train
dan test sekaligus.

## 8. Dataset `sitting_posture_detection_initial`

Contact sheet memperlihatkan campuran:

-   foto nyata;
-   stock image;
-   ilustrasi/komposit;
-   satu atau beberapa orang;
-   berbagai jenis kursi;
-   background sangat beragam;
-   framing berbeda;
-   sebagian gambar kurang ideal untuk analisis pose.

Kriteria curation:

  -----------------------------------------------------------------------
  Kondisi                             Perlakuan
  ----------------------------------- -----------------------------------
  Orang jelas dan postur duduk jelas  Keep

  Orang terlalu kecil                 Review/Reject

  Tubuh terpotong sehingga postur     Reject/khusus
  tidak dapat dianalisis              

  Komposit/collage                    Review

  Multi-person                        Review

  Ilustrasi                           Pisahkan

  Source tidak jelas                  Tandai

  Duplicate/near-duplicate            Cluster/remove

  Label meragukan                     Review
  -----------------------------------------------------------------------

## 9. Dataset privat

Dataset privat tetap menjadi kandidat utama untuk data yang paling
sesuai dengan tujuan penelitian karena kamera, subjek, kursi, sudut,
pencahayaan, dan metadata dapat dikontrol.

Konfigurasi minimum yang disarankan:

-   Kamera 1: frontal;
-   Kamera 2: lateral 90°;
-   Kamera 3: 45° jika tersedia.

Untuk pipeline pose, full-body lebih disarankan daripada gambar hanya
sampai bahu karena informasi kepala, bahu, torso, pinggul, lutut, dan
kaki dapat digunakan.

Metadata dapat mencakup:

`session_id`, `subject_id`, `chair_id`, `camera_id`, `view_angle`,
`posture_class`, `backrest`, `seat_height`, `seat_depth`, `seat_width`,
`lighting_condition`, `occlusion`, `notes`.

Atribut kursi seperti sandaran, tinggi, kedalaman, dan lebar **sebaiknya
menjadi metadata terlebih dahulu, bukan subclass**.

## 10. Class vs metadata vs subclass

**Class** = target yang diprediksi model, misalnya `upright`,
`slouched`, `lean_left`, `lean_right`.

**Metadata** = kondisi pengambilan data, misalnya `backrest=yes`,
`seat_height=45 cm`, `camera_angle=90`.

**Subclass** = kategori turunan yang memang ingin dianalisis. Tidak
perlu dibuat sejak awal.

Taxonomy kelas final belum sebaiknya dikunci sebelum referensi dan pilot
collection selesai.

## 11. Split dan data leakage

Split utama harus subject-independent jika memungkinkan.

Contoh yang benar:

``` text
Subject A → train
Subject B → train
Subject C → validation
Subject D → test
```

Hindari:

``` text
Subject A image 1 → train
Subject A image 2 → test
```

Hal yang sama berlaku untuk frame video: group berdasarkan video/source.

## 12. Evaluasi

Minimal:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix

Untuk imbalance:

-   Macro F1
-   Balanced Accuracy

Jika mengevaluasi tahap detection YOLO secara terpisah, laporkan metrik
detection seperti Precision, Recall, dan mAP. Metrik detection dan
klasifikasi postur harus dipisahkan.

## 13. Research gap sementara

Dataset public yang tersedia menunjukkan masalah framing tidak seragam,
ketergantungan pada frame video, keypoint terbatas, variasi label,
dan/atau near-duplicate. Jika random image split digunakan, performa
dapat menjadi terlalu optimistis.

Arah kontribusi:

> Mengembangkan dan mengevaluasi pipeline klasifikasi postur duduk
> berbasis pose dengan curation terkontrol dan subject-independent
> evaluation, serta membandingkan image-based, keypoint-based, dan
> pipeline otomatis berbasis YOLO Pose.

Dataset privat digunakan untuk menguji pipeline pada data yang
dikumpulkan dengan konfigurasi kamera dan kondisi kursi yang terkontrol.

## 14. Urutan pekerjaan berikutnya

1.  Finalisasi audit empat dataset public.
2.  Curation `sitting_posture_detection_initial`.
3.  Verifikasi mapping 7 keypoint dan arti kelas `Postureexercise`.
4.  Finalisasi duplicate clusters IKORN.
5.  Finalisasi grouping `project_design_20242025`.
6.  Tetapkan taxonomy kelas penelitian.
7.  Jalankan EXP-01.
8.  Jalankan EXP-02.
9.  Jalankan EXP-03.
10. Cross-dataset evaluation.
11. Baru finalisasi protokol dan pengumpulan dataset privat.

## 15. Status

-   [x] Kandidat dataset public
-   [x] Audit awal
-   [x] Contact sheet `sitting_posture_detection_initial`
-   [x] Pipeline image baseline
-   [x] Pipeline keypoint
-   [x] Pipeline YOLO Pose
-   [x] Dataset privat sebagai kandidat penting
-   [ ] Mapping 7 keypoint
-   [ ] Arti kelas Postureexercise
-   [ ] Curation `project_design_20242025`
-   [ ] Curation `sitting_posture_detection_initial`
-   [ ] Finalisasi IKORN duplicate clusters
-   [ ] Finalisasi taxonomy kelas
-   [ ] Protokol dataset privat
-   [ ] Training baseline
-   [ ] YOLO Pose experiment
-   [ ] Cross-dataset evaluation
-   [ ] Private dataset evaluation

## 16. Keputusan praktis

**Belum perlu mengumpulkan dataset privat dalam jumlah besar sekarang.**
Audit empat dataset public perlu diselesaikan terlebih dahulu karena
hasilnya akan menentukan kebutuhan keypoint, kelas, kondisi kursi,
variasi kamera, dan jenis data yang benar-benar masih kurang.

Perbaikan utama terhadap rencana sebelumnya bukan mengganti pipeline,
tetapi membuat **curation dan group-aware evaluation sebagai tahap
wajib** serta menempatkan setiap dataset sesuai karakteristiknya.
