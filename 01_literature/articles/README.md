# Indeks & Pemetaan Artikel Literatur Penelitian (01_literature/articles)

Dokumen ini berisi inventaris lengkap, klasifikasi tematik, dan analisis relevansi dari artikel ilmiah yang dikoleksi untuk penelitian:  
**"Deteksi dan Klasifikasi Postur Duduk Berisiko Skoliosis Menggunakan Estimasi Pose (2D YOLOv8-Pose vs Stereo 3D) dan Machine Learning (XGBoost)"**.

---

## 1. Analisis Relevansi Terhadap Proyek

> **Kesimpulan:** Seluruh file artikel yang ada di folder ini (**100% dari 16 artikel unik**) adalah referensi yang **sangat relevan, kontekstual, dan langsung dapat digunakan** untuk penulisan Skripsi dan Naskah Publikasi (Bab 1 Latar Belakang, Bab 2 Tinjauan Pustaka, Bab 3 Desain Metodologi, dan Bab 4 Pembahasan Komparatif).

### Keterkaitan Langsung per Komponen Proyek:

| Komponen Proyek Kita | Artikel Terkait Langsung | Relevansi & Peran Spesifik |
|---|---|---|
| **Klasifikasi Postur dengan XGBoost** | Pawitra et al. (2026) | Membuktikan efektivitas XGBoost untuk deteksi postur duduk ergonomis, menjadi baseline pembanding langsung untuk pemilihan model tabular. |
| **Estimasi Pose 2D & YOLO-based Posture** | Li et al. (2025), Cao et al. (2025), Pawitra et al. (2025), Ding et al. (2024) | Justifikasi penggunaan model pose 2D ringan (YOLO/Keypoints), multi-view recognition, dan ekstraksi fitur sudut/tulang. |
| **Estimasi Postur 3D & Analisis Kedalaman** | Shimomae & Takemura (2024), Jin et al. (2024) | Rujukan untuk rekonstruksi 3D seated posture, integrasi sensor kedalaman (depth sensor), serta tantangan domain shift spasial 3D. |
| **Eksplorasi Algoritma Alternatif (MLP vs KAN)** | Carneros-Prado et al. (2024), Wang et al. (2025a, 2025b) | Perbandingan arsitektur klasifikasi tabular/postur duduk (MLP vs KAN, Neuro-symbolic), memperkuat justifikasi pemilihan XGBoost. |
| **Tinjauan Pustaka Komprehensif (SOTA)** | Nadeem et al. (2024), Krauter et al. (2024), Odesola et al. (2024) | Fondasi Bab 2: taksonomi sistem sensor (vision-based vs chair-based/wearable), tantangan oklusi, dan intervensi feedback. |
| **Penerapan IoT & Sistem Mitigasi Real-Time** | Sahoo et al. (2026), Wang et al. (2025) | Dasar perancangan Bab 5 & Future Work untuk arsitektur IoT pemantauan postur duduk berisiko penyakit muskuloskeletal. |

---

## 2. Struktur Pengelompokan Folder

Koleksi artikel telah dikelompokkan ke dalam 5 folder tematik dan 1 folder arsip/duplikat:

```text
01_literature/articles/
├── 01_literature_reviews/                (3 artikel survei & tinjauan pustaka SOTA)
├── 02_2d_vision_and_pose_estimation/     (5 artikel estimasi pose 2D & CNN)
├── 03_3d_pose_and_depth_sensors/         (2 artikel estimasi postur 3D & depth)
├── 04_machine_learning_and_classifiers/  (4 artikel klasifikasi ML & XGBoost)
├── 05_iot_and_risk_management_systems/   (2 artikel sistem IoT & mitigasi risiko)
├── _drafts_and_duplicates/               (2 file duplikat/draft pre-print)
└── README.md                             (Indeks ini)
```

---

## 3. Daftar Inventaris Lengkap Artikel

### Folder `01_literature_reviews/` (Tinjauan Pustaka & Survey SOTA)

| No | File | Sitasi / Penulis | Judul & Publikasi | Relevansi untuk Skripsi |
|:---:|---|---|---|---|
| 1 | [Nadeem_2024](file:///01_literature/articles/01_literature_reviews/Nadeem_2024_Sitting_Posture_Recognition_Systems_Review.pdf) | Nadeem, Elbasi, Zreikat, & Sharsheer (2024) | *Sitting Posture Recognition Systems: Comprehensive Literature Review and Analysis* (MDPI Appl. Sci., 14(18), 8557) | Referensi utama Bab 2 untuk taksonomi metode deteksi postur duduk: sensor tekanan vs visual/kamera, kelebihan & kelemahan masing-masing. |
| 2 | [Krauter_2024](file:///01_literature/articles/01_literature_reviews/Krauter_2024_Sitting_Posture_Recognition_and_Feedback_Review.pdf) | Krauter, Angerbauer, Calepso, Achberger, Mayer, & Sedlmair (2024) | *Sitting Posture Recognition and Feedback: A Literature Review* (ACM CHI/HCI Review) | Digunakan untuk latar belakang dampak sedentary behaviour, jenis intervensi feedback (haptic/visual), dan kepatuhan pengguna. |
| 3 | [Odesola_2024](file:///01_literature/articles/01_literature_reviews/Odesola_2024_Smart_Sensing_Chairs_Sitting_Posture_Review.pdf) | Odesola, Kulon, Verghese, Partlow, & Gibson (2024) | *Smart Sensing Chairs for Sitting Posture Detection, Classification, and Monitoring: A Comprehensive Review* (MDPI Sensors, 24(9), 2940) | Justifikasi mengapa vision-based/stereo camera lebih fleksibel dan non-intrusif dibandingkan smart chair berbasis sensor tekanan yang mahal dan spesifik perangkat. |

---

### Folder `02_2d_vision_and_pose_estimation/` (Estimasi Pose 2D & Deep Learning)

| No | File | Sitasi / Penulis | Judul & Publikasi | Relevansi untuk Skripsi |
|:---:|---|---|---|---|
| 4 | [Li_2025](file:///01_literature/articles/02_2d_vision_and_pose_estimation/Li_2025_LSP_YOLO_Lightweight_Sitting_Posture_Embedded.pdf) | Li, Hao, Wang, & Wang (2025) | *LSP-YOLO: A Lightweight Single-Stage Network for Sitting Posture Recognition on Embedded Devices* (arXiv:2511.14322) | Sangat relevan untuk Bab 3 & Bab 4: penggunaan arsitektur YOLO untuk pengenalan postur duduk pada perangkat komputasi terbatas / edge devices. |
| 5 | [Cao_2025](file:///01_literature/articles/02_2d_vision_and_pose_estimation/Cao_2025_KeypointNet_MultiView_Sitting_Posture.pdf) | Cao, Wu, Wu, Jiao, Xiao, Zhang, & Zhou (2025) | *KeypointNet: An Efficient Deep Learning Model with Multi-View Recognition Capability for Sitting Posture Recognition* (MDPI Electronics, 14(4), 718) | Rujukan penting untuk sistem *multi-view* recognition; sejajar dengan setup dual-kamera (stereo) proyek kita untuk mengatasi oklusi tubuh. |
| 6 | [Ding_2024](file:///01_literature/articles/02_2d_vision_and_pose_estimation/Ding_2024_Attention_CNN_Whole_Body_Postural_Assessment.pdf) | Ding, Li, Yang, Ogunbona, & Qin (2024) | *An attention-based CNN for automatic whole-body postural assessment* (Elsevier Expert Systems with Applications, 238, 122391) | Dasar klinis dan visual untuk penilaian postur seluruh tubuh berbasis attention mechanism. |
| 7 | [Pawitra_2025](file:///01_literature/articles/02_2d_vision_and_pose_estimation/Pawitra_2025_Detection_Ergonomic_Sitting_Postures_Office.pdf) | Pawitra, Sitania, Burhandenny, Wijaya, Septiarini, Hamdani, & Fareez (2025) | *Detection of Ergonomic Sitting Postures in Office Environments* (Journal of Image and Graphics, 13(6), 686-701) | Studi empiris deteksi postur duduk ergonomis di lingkungan kantor Indonesia menggunakan computer vision. |
| 8 | [Ogundokun_2022](file:///01_literature/articles/02_2d_vision_and_pose_estimation/Ogundokun_2022_Human_Posture_Augmentation_Transfer_Learning.pdf) | Ogundokun, Maskeliūnas, & Damaševičius (2022) | *Human Posture Detection Using Image Augmentation and Hyperparameter-Optimized Transfer Learning Algorithms* (MDPI Appl. Sci., 12(19), 10156) | Rujukan teknik augmentasi data citra dan optimasi hyperparameter untuk klasifikasi postur tubuh. |

---

### Folder `03_3d_pose_and_depth_sensors/` (Estimasi Postur 3D & Sensor Kedalaman)

| No | File | Sitasi / Penulis | Judul & Publikasi | Relevansi untuk Skripsi |
|:---:|---|---|---|---|
| 9 | [Shimomae_2024](file:///01_literature/articles/03_3d_pose_and_depth_sensors/Shimomae_2024_Monocular_3D_Seated_Posture_Smart_Chair.pdf) | Shimomae & Takemura (2024) | *Monocular Image-Based 3D Seated Posture Estimation Integrated with a Seat-Angle Adjustable Smart Chair* (SciTePress / Kyushu Inst. of Tech.) | Studi krusial yang mengkaji rekonstruksi 3D skeleton keypoints untuk klasifikasi postur duduk (good, slightly poor, poor) dan membuktikan keunggulan fitur tubuh tanpa ekstremitas. |
| 10 | [Jin_2024](file:///01_literature/articles/03_3d_pose_and_depth_sensors/Jin_2024_SitPose_Ensemble_Learning_Depth_Sensor.pdf) | Jin, He, Wang, Zhu, Jiang, & Zhou (2024) | *SitPose: Real-Time Detection of Sitting Posture and Sedentary Behavior Using Ensemble Learning With Depth Sensor* (IEEE Sensors Journal / arXiv:2412.12216) | Pembanding langsung untuk pipeline 3D/depth: memadukan sensor kedalaman dengan ensemble learning (mirip kombinasi stereo depth + XGBoost di proyek kita). |

---

### Folder `04_machine_learning_and_classifiers/` (Klasifikasi ML & XGBoost)

| No | File | Sitasi / Penulis | Judul & Publikasi | Relevansi untuk Skripsi |
|:---:|---|---|---|---|
| 11 | [Pawitra_2026](file:///01_literature/articles/04_machine_learning_and_classifiers/Pawitra_2026_Automated_Ergonomic_Sitting_Postures_XGBoost.pdf) | Pawitra, Sitania, Septiarini, & Hamdani (2026) | *Automated ergonomic sitting postures detection for office workstation using XGBoost method* (IAES IJ-AI, 15(1), 506-514) | **SANGAT PENTING (KUNCI):** Paper ini secara langsung menggunakan **metode XGBoost** untuk klasifikasi postur duduk ergonomis di lingkungan kerja Indonesia. Menjadi basis rujukan metodologi klasifikasi kita. |
| 12 | [Carneros_Prado_2024](file:///01_literature/articles/04_machine_learning_and_classifiers/Carneros_Prado_2024_MLP_vs_KAN_Sitting_Posture.pdf) | Carneros-Prado, Cabañero-Gómez, Johnson, González, Fontecha, & Hervás (2024) | *A Comparison Between Multilayer Perceptrons and Kolmogorov-Arnold Networks for Multi-Task Classification in Sitting Posture Recognition* (IEEE Access, 12, 180034-180045) | Studi perbandingan mutakhir antara model berbasis jaringan (MLP) dan arsitektur baru (KAN) untuk multi-task sitting posture recognition. |
| 13 | [Wang_2025a](file:///01_literature/articles/04_machine_learning_and_classifiers/Wang_2025_MSBN_SPose_Bayesian_Neuro_Symbolic.pdf) | Wang, Tavares, Lima, Gomes, Zhang, & Liang (2025) | *MSBN-SPose: A Multi-Scale Bayesian Neuro-Symbolic Approach for Sitting Posture Recognition* (MDPI Electronics, 14(19), 3889) | Pendekatan neuro-symbolic multi-skala untuk pengenalan postur duduk yang lebih interpretable dan robust terhadap noise spasial. |
| 14 | [Wang_2025b](file:///01_literature/articles/04_machine_learning_and_classifiers/Wang_2025_LAViTSPose_Lightweight_Cascaded_Framework.pdf) | Wang, Tavares, Lima, Gomes, Zhang, Zhao, & Liang (2025) | *LAViTSPose: A Lightweight Cascaded Framework for Robust Sitting Posture Recognition via Detection–Segmentation–Classification* (MDPI Entropy, 27(12), 1196) | Framework berjenjang (deteksi-segmentasi-klasifikasi) untuk menangani variasi posisi subjek dalam postur duduk. |

---

### Folder `05_iot_and_risk_management_systems/` (Sistem Terintegrasi IoT & Manajemen Risiko)

| No | File | Sitasi / Penulis | Judul & Publikasi | Relevansi untuk Skripsi |
|:---:|---|---|---|---|
| 15 | [Sahoo_2026](file:///01_literature/articles/05_iot_and_risk_management_systems/Sahoo_2026_ALIGN_AI_Driven_IoT_Sitting_Posture.pdf) | Sahoo, Patel, Swain, Gerogiannis, Kanavos, Singh, Kumar, & Acharya (2026) | *ALIGN: An AI-Driven IoT Framework for Real-Time Sitting Posture Detection* (MDPI Algorithms, 19(1), 48) | Kerangka kerja integrasi IoT dan AI untuk deteksi postur duduk real-time. Relevan untuk pembahasan implementasi praktis dan arsitektur sistem mitigasi skoliosis di masa depan. |
| 16 | [Wang_2025c](file:///01_literature/articles/05_iot_and_risk_management_systems/Wang_2025_LGCSPNet_Sitting_Posture_Risk_Management.pdf) | Wang, Pang, Cao, Song, An, Wu, Zhao, Wang, & Zhou (2025) | *A Lightweight Model LGCSPNet for Sitting Posture Risk Management Applications* (Elsevier Expert Systems with Applications, 291, 128486) | Model lightweight graph convolutional untuk aplikasi manajemen risiko postur duduk; sangat relevan untuk pembahasan mitigasi bahaya muskuloskeletal jangka panjang. |

---

### Folder `_drafts_and_duplicates/` (Salinan Duplikat & Draft Pre-print)

Folder ini menyimpan file yang sengaja dipisahkan agar tidak membingungkan referensi utama:
- `algorithms_19_00048_duplicate_of_Sahoo2026.pdf`: File identik 100% teks dengan versi resmi `algorithms-19-00048.pdf` (Sahoo et al., 2026).
- `Carneros_Prado_2024_MLP_vs_KAN_early_proof.pdf`: Draft unpaginated proof awal dari Carneros-Prado et al. (2024), di mana versi final yang bernomor volume dan halaman lengkap sudah tersimpan di folder `04_machine_learning_and_classifiers/`.

---

## 4. Panduan Sitasi dalam Naskah Skripsi & Publikasi

Ketika menyusun Bab 1 hingga Bab 5, artikel-artikel di atas dapat dirujuk secara spesifik sebagai berikut:

1. **Bab 1 (Pendahuluan):**
   - Rujuk **Krauter et al. (2024)** dan **Nadeem et al. (2024)** untuk statistik bahaya postur duduk buruk (*prolonged sitting*) terhadap tulang belakang dan kesehatan muskuloskeletal (skoliosis).
   - Rujuk **Odesola et al. (2024)** untuk membandingkan pendekatan sensor matras/kursi dengan computer vision.

2. **Bab 2 (Tinjauan Pustaka):**
   - Rujuk **Pawitra et al. (2026)** untuk penggunaan XGBoost pada klasifikasi postur kerja kantor di Indonesia.
   - Rujuk **Cao et al. (2025)** untuk pentingnya pengenalan multi-view dalam mengatasi oklusi postur duduk.
   - Rujuk **Shimomae & Takemura (2024)** dan **Jin et al. (2024)** untuk studi komparasi estimasi pose 2D vs 3D/depth.

3. **Bab 3 (Metodologi Penelitian):**
   - Rujuk **Li et al. (2025)** sebagai justifikasi arsitektur berbasis keypoint pose untuk perangkat komputasi real-time.
   - Rujuk **Pawitra et al. (2026)** dan **Carneros-Prado et al. (2024)** untuk pemilihan algoritma klasifikasi tabular (XGBoost) berbasis vektor fitur geometri sudut tubuh.

4. **Bab 4 (Hasil dan Pembahasan):**
   - Rujuk temuan **Shimomae & Takemura (2024)** saat mendiskusikan performa model 3D vs 2D: rekonstruksi 3D memiliki tantangan sensitivitas sudut kamera dan variasi antarsubjek (*canonicalization*), sehingga model 2D terbukti lebih stabil dan unggul secara praktis (*Macro F1 0.7039* vs *0.6182*).
