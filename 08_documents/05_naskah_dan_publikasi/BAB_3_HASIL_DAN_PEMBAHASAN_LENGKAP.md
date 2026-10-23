# BAB III HASIL DAN PEMBAHASAN

Bab ini menyajikan temuan empiris, analisis performa kuantitatif, serta pembahasan ilmiah dari sistem klasifikasi postur duduk berbasis visi stereo ganda. Pembahasan diorganisasikan secara sistematis mengikuti tahapan eksperimen yang telah dirancang pada Bab II, mencakup hasil kalibrasi sistem kamera stereo, karakteristik dataset hasil penapisan kontrol kualitas (*quality control*), evaluasi komparatif performa empat representasi fitur utama (2D vs 3D), investigasi kesalahan klasifikasi per kelas (*failure analysis*), hingga pengujian operasional inferensi waktu-nyata (*real-time benchmark*).

---

### 3.1 Hasil Kalibrasi Sistem Kamera Stereo

Kalibrasi sistem kamera ganda (*dual USB webcams*) dilakukan untuk memperoleh parameter intrinsik masing-masing kamera serta relasi geometris ekstrinsik antar-kamera sebagai fondasi rekonstruksi spasial 3D.

#### 3.1.1 Parameter Geometri Optik Kamera (Intrinsik dan Ekstrinsik)
Proses kalibrasi menggunakan pola papan catur (*checkerboard*) berdimensi $9 \times 6$ titik sudut interior ($54$ titik sudut) dengan ukuran kotak $25\text{ mm}$ yang direkam secara simultan pada berbagai orientasi sudut kemiringan ($\ge 30$ pasangan citra). Estimasi parameter optik menghasilkan matriks intrinsik $K$ dan vektor koefisien distorsi lensa $D$ untuk Kamera Depan (CAM01) dan Kamera Samping (CAM02) sebagaimana disajikan pada Tabel 3.1.

**Tabel 3.1** Parameter intrinsik optik dan koefisien distorsi lensa kamera stereo
| Parameter Optik | Kamera Depan (CAM01) | Kamera Samping (CAM02) | Satuan / Keterangan |
| :--- | :---: | :---: | :--- |
| Panjang Fokus Horizontal ($f_x$) | 1342,85 | 1358,12 | Piksel |
| Panjang Fokus Vertikal ($f_y$) | 1345,10 | 1360,45 | Piksel |
| Titik Utama Horizontal ($c_x$) | 638,42 | 642,18 | Piksel |
| Titik Utama Vertikal ($c_y$) | 359,74 | 362,05 | Piksel |
| Distorsi Radial ($k_1, k_2$) | $-0{,}0412, \ 0{,}1154$ | $-0{,}0385, \ 0{,}0982$ | Koefisien tanpa dimensi |
| Distorsi Tangensial ($p_1, p_2$) | $0{,}0012, \ -0{,}0008$ | $0{,}0009, \ -0{,}0005$ | Koefisien tanpa dimensi |
| Resolusi Citra Operasional | $1280 \times 720$ | $1280 \times 720$ | Piksel (HD 720p) |

Relasi ekstrinsik stereo yang memodelkan orientasi spasial relatif CAM02 terhadap CAM01 menghasilkan matriks rotasi stereo $R_{\text{stereo}}$ dengan sudut rotasi konvergen mendekati siku-siku ($\theta \approx 88{,}4^\circ$) serta vektor translasi stereo $T_{\text{stereo}} = [-1045{,}2, \ 82{,}4, \ 874{,}6]^T\text{ mm}$, yang merefleksikan jarak fisik dasar (*baseline distance*) sekitar $1{,}36\text{ meter}$.

#### 3.1.2 Evaluasi Akurasi dan Validasi Reprojection Error
Tingkat keandalan geometri kalibrasi dievaluasi menggunakan metrik kesalahan proyeksi ulang rata-rata (*mean reprojection error*). Hasil pengukuran kalibrasi pada setup rig operasional menunjukkan nilai kesalahan sebesar $0{,}38\text{ piksel}$ untuk CAM01, $0{,}42\text{ piksel}$ untuk CAM02, dan kesalahan stereo gabungan sebesar $0{,}85\text{ piksel}$. Nilai ini berada jauh di bawah batas ambang toleransi yang ditetapkan pada Bab II ($< 1{,}5\text{ piksel}$), yang mengonfirmasi bahwa distorsi optik telah terkoreksi secara akurat dan geometri garis epipolar telah memenuhi syarat presisi untuk triangulasi stereo 3D menggunakan metode *Direct Linear Transform* (DLT).

---

### 3.2 Karakteristik Dataset Final dan Analisis Penapisan Kualitas (QC Audit)

Pengumpulan data primer menghasilkan rekaman citra dari 24 subjek sukarelawan (`S001` hingga `S024`) pada lingkungan kerja semi-terbuka dengan latar belakang alami. Keragaman antropometri subjek diilustrasikan pada Gambar 3.1.

![Gambar 3.1 Keragaman Antropometri 24 Subjek Partisipan](file:///d:/.Candra/Project/TA/07_results/private_audit/contact_sheets/contact_sheet_all_24_subjects_overview.jpg)
**Gambar 3.1** Matriks visual kompilasi 24 subjek partisipan penelitian (`S001`–`S024`), memperlihatkan variasi karakteristik antropometri tubuh (tinggi badan, rentang lebar bahu, dan proporsi torso) pada lingkungan semi-terbuka alami.

Setiap partisipan memperagakan variasi postur duduk kerja yang mencakup enam kelas target ergonomis sebagaimana ditunjukkan pada contoh pasangan citra stereo Gambar 3.2.

![Gambar 3.2 Contoh Sampel Pasangan Citra Enam Kelas Postur](file:///d:/.Candra/Project/TA/07_results/private_audit/contact_sheets/contact_sheet_7_posture_classes.jpg)
**Gambar 3.2** Sampel pasangan citra stereo (Kamera Depan CAM01 dan Kamera Samping CAM02) untuk seluruh kelas target postur duduk yang diteliti: `upright`, `leaning_forward`, `leaning_backward`, `leaning_left`, `leaning_right`, `slouching`, dan kondisi `reject`.

#### 3.2.1 Distribusi Sampel Pasangan Citra pada Subset Irisan Valid
Proses penyaringan data mentah melalui protokol kontrol kualitas bertingkat (*two-layer quality control*) menghasilkan penyusutan bertahap dari data akuisisi awal hingga menjadi subset data model akhir, seperti dirangkum pada Tabel 3.2.

**Tabel 3.2** Tahapan penyaringan data akuisisi dan distribusi kelas pada subset irisan valid
| Kelas Postur Duduk | Tangkapan Mentah 6 Kelas | Lolos Stereo-QC (2D Coverage) | Lolos Feature-QC (Intersection 2D-3D) | Persentase Dataset Final |
| :--- | :---: | :---: | :---: | :---: |
| `upright` | 148 | 143 | **82** | 20,35% |
| `leaning_forward` | 132 | 129 | **75** | 18,61% |
| `slouching` | 119 | 114 | **65** | 16,13% |
| `leaning_left` | 112 | 108 | **62** | 15,38% |
| `leaning_right` | 110 | 106 | **61** | 15,14% |
| `leaning_backward` | 106 | 104 | **58** | 14,39% |
| **Total 6 Kelas Utama** | **727** | **704 (96,84%)** | **403 (55,43%)** | **100,00%** |
| Subjek Partisipan | 24 Subjek | 24 Subjek | **18 Subjek** | - |

Dari total 885 rekaman pasangan citra mentah, sebanyak 158 rekaman disaring di awal (121 rekaman *forward head* terisolasi dan 37 rekaman *reject* gerakan transisi/berdiri), menyisakan 727 pasangan citra enam kelas postur utama. Dari jumlah tersebut, sebanyak 704 pasangan citra (96,84%) berhasil memenuhi kriteria deteksi pose 2D (*Stereo-QC*). Namun, ketika protokol *Feature-QC* diterapkan untuk keperluan rekonstruksi 3D murni, dataset menyusut menjadi **403 sampel pasangan citra valid** yang berasal dari **18 subjek partisipan unik** (`S001`–`S006`, `S011`–`S019`, `S021`, `S022`, dan `S024`).

#### 3.2.2 Analisis Oklusi Anatomis Panggul Lateral (Investigasi Eliminasi 6 Subjek)
Penyusutan dari 24 subjek menjadi 18 subjek (kehilangan subjek `S007`, `S008`, `S009`, `S010`, `S020`, dan `S023`) dianalisis secara mendalam melalui audit visual rekaman citra. Hasil audit membuktikan bahwa seluruh kegagalan disebabkan oleh **terjadinya oklusi anatomis pada titik sendi panggul kanan (`right_hip`) di Kamera Samping (CAM02)**. 

Pada posisi duduk di atas kursi bangku (*stool*), segmen paha subjek sering kali menutupi sendi trochanter mayor panggul terhadap sudut pandang lateral, terutama pada subjek dengan pakaian longgar atau saat pencahayaan lokal menghasilkan bayangan pekat pada area panggul bawah. Karena triangulasi stereo mensyaratkan ketersediaan koordinat titik di kedua kamera secara simultan untuk menghitung koordinat kedalaman ($Z$), hilangnya titik panggul kanan di salah satu kamera menyebabkan vektor torso tidak dapat dibentuk secara geometris.

Keputusan untuk menolak sampel ber-oklusi ini—dan tidak menggantikannya dengan imputasi nol (*zero imputation*) atau estimasi interpolasi buatan—merupakan langkah ilmiah krusial. Jika titik $(0,0)$ dipaksakan masuk ke algoritma DLT, triangulasi spasial akan menghasilkan nilai koordinat 3D yang divergen/anomali, yang dapat merusak integritas model klasifikasi. Sebanyak 403 sampel dari 18 subjek inilah yang menjadi **subset irisan valid (*intersection subset*)**, menjamin pengujian komparatif yang setara (*apple-to-apple*) antara representasi 2D dan 3D.

---

### 3.3 Evaluasi Komparatif Kinerja Empat Representasi Fitur (Hasil Utama)

Keempat varian representasi fitur dievaluasi menggunakan model klasifikasi XGBoost yang dikonfigurasi identik melalui protokol *5-Fold Subject-Stratified Cross-Validation* pada subset irisan data 403 sampel.

#### 3.3.1 Perbandingan Performa Makro: 2D Core, 2D Best, 3D Camera, dan 3D Canonical
Hasil evaluasi agregat *Pooled Out-of-Fold* (OOF) dan stabilitas antarlipatan dari keempat model disajikan pada Tabel 3.3 dan divisualisasikan pada Gambar 3.3.

**Tabel 3.3** Rangkuman evaluasi komparatif kinerja empat varian model postur pada subset irisan data valid (N = 403, 18 Subjek)
| Varian Representasi Fitur | Dimensi Fitur | Akurasi Pooled | Macro Precision | Macro Recall | **Macro F1-Score (Pooled OOF)** | Fold Mean F1 (%) | Fold Std F1 ($\sigma_{\text{fold}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D Best Practical** | **42** | **69,73%** | **0,7174** | **0,7046** | **0,7039** | **68,75%** | **0,0701** |
| 2D Core Baseline | 36 | 67,99% | 0,6897 | 0,6901 | 0,6863 | 66,69% | 0,0846 |
| 3D Camera Frame | 25 | 61,79% | 0,6462 | 0,6267 | 0,6182 | 60,46% | 0,2519 |
| 3D Canonical Body Frame | 22 | 33,00% | 0,3378 | 0,3305 | 0,3259 | 32,37% | 0,1241 |

![Gambar 3.3 Perbandingan Kinerja Empat Varian Model](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_5_komparasi_empat_model.png)
**Gambar 3.3** Perbandingan kinerja klasifikasi postur duduk pada subset irisan valid: (a) Perbandingan Akurasi dan Macro F1-Score Pooled OOF, dan (b) Stabilitas generalisasi model antarlipatan (Fold Mean $\pm$ Fold Std).

Berdasarkan Tabel 3.3, model **2D Best Practical (42 fitur)** mencapai kinerja tertinggi di antara seluruh varian yang diuji, dengan **Macro F1-Score sebesar 0,7039** dan Akurasi **69,73%**. Capaian ini melampaui varian 2D Core Baseline (F1 = 0,6863) serta mengungguli representasi spasial 3D Camera Frame (F1 = 0,6182) dengan margin peningkatan yang substansial sebesar $+8{,}57\%$ F1-score.

#### 3.3.2 Analisis Kestabilan Generalisasi Antarsubjek (Fold Standard Deviation)
Evaluasi variabilitas performa antarlipatan (*Fold Standard Deviation*, $\sigma_{\text{fold}}$) pada Gambar 3.3(b) memperlihatkan perbedaan stabilitas model yang sangat kontras saat diuji pada subjek baru (*unseen subjects*):
* **2D Best Practical** menunjukkan konsistensi antarsubjek terbaik dengan deviasi standar terkecil ($\sigma_{\text{fold}} = 0{,}0701$ atau $\pm 7{,}01\%$), dengan skor per lipatan berkisar stabil antara $59{,}30\%$ hingga $77{,}03\%$.
* **3D Camera Frame** memperlihatkan ketidakstabilan performa yang tinggi dengan deviasi standar sebesar $\sigma_{\text{fold}} = 0{,}2519$ ($\pm 25{,}19\%$). Pemeriksaan mendalam terhadap skor per fold membuktikan bahwa pada Fold 0, Fold 1, Fold 2, dan Fold 4 model 3D mencapai F1 di atas $67\%–76\%$, namun performanya anjlok drastis pada Fold 3 (F1 = $10{,}52\%$). Hal ini membuktikan bahwa rekonstruksi metrik 3D sangat rentan terhadap variasi geometri individual subjek pada fold tertentu (misalnya subjek dengan proporsi torso yang lebih pendek atau posisi duduk yang sedikit bergeser dari sumbu optik).

#### 3.3.3 Signifikansi Peningkatan Fitur Telinga Sagital pada Varian 2D Best (42f)
Keberhasilan varian 2D Best mengungguli 2D Core memberikan konfirmasi empiris atas hipotesis penelitian: penambahan enam fitur telinga sagital ($\Delta x$, $\Delta y$, jarak Euclidean telinga-bahu ternormalisasi, dan sudut inklinasi leher dari CAM02) secara efektif mengatasi kelemahan sudut torso global. Pada postur membungkuk (*slouching*), pelvis subjek meluncur ke depan sementara kepala bertranslasi ke arah anterior (*forward head posture*). Orientasi garis torso sering kali tampak menyerupai posisi duduk tegak netral jika diamati hanya dari kemiringan batang tubuh. Enam fitur telinga lateral menangkap pergeseran relatif segmen servikal-kraniofasial terhadap sabuk bahu, menghasilkan peningkatan performa klasifikasi secara konsisten.

#### 3.3.4 Degradasi Performa pada Model 3D Canonical (Hilangnya Referensi Gravitasi)
Temuan paling mencolok dalam evaluasi komparatif ini adalah anjloknya kinerja model **3D Canonical Body Frame (22 fitur)**, yang hanya menghasilkan Akurasi $33{,}00\%$ dan Macro F1-Score sebesar $0{,}3259$.

Secara teoritis, normalisasi Gram-Schmidt dirancang untuk membuat representasi tubuh invarian terhadap posisi rig kamera. Namun, analisis ilmiah membuktikan bahwa transformasi ini justru **menghilangkan referensi sumbu gravitasi vertikal bumi (*world vertical gravity reference*)**. Postur duduk ergonomis manusia didefinisikan berdasarkan orientasi tubuh terhadap gaya tarik gravitasi bumi (misal: condong ke kiri, condong ke depan, atau tegak lurus lantai). Ketika koordinat metrik 3D diputar ke dalam kerangka acuan lokal tubuh di mana sumbu longitudinal tubuh ditetapkan sebagai sumbu $Y_{\text{can}}$ lokal:
* Subjek yang duduk tegak netral (*upright*) dan subjek yang miring ke kiri (*leaning left*) sama-sama memiliki orientasi vektor torso yang berimpit pada sumbu $Y_{\text{can}}$ lokal;
* Sudut kemiringan lateral dan sagital torso terdegenerasi menjadi konstan $0^\circ$ dengan varians nol;
* Model kehilangan informasi absolut mengenai apakah subjek sedang condong terhadap ruang gravitasi dunia atau berada dalam posisi tegak lurus.

Akibatnya, model XGBoost 3D Canonical gagal membedakan kelas-kelas postur yang bergantung pada orientasi kemiringan batang tubuh, membuktikan bahwa fitur postur berbasis tubuh lokal murni tidak memadai untuk klasifikasi postur kerja tanpa adanya sensor gravitasi eksternal.

---

### 3.4 Analisis Performa Per Kelas dan Investigasi Kesalahan (Failure Analysis)

Evaluasi rinci per kelas postur dilakukan untuk membedah keunggulan spesifik dan pola kelemahan masing-masing representasi, sebagaimana dirangkum pada Tabel 3.4.

**Tabel 3.4** Rincian metrik evaluasi per kelas (F1-score, Precision, Recall) untuk empat varian model
| Model / Representasi | `leaning_backward` (F1 / Prec / Rec) | `leaning_forward` (F1 / Prec / Rec) | `leaning_left` (F1 / Prec / Rec) | `leaning_right` (F1 / Prec / Rec) | `slouching` (F1 / Prec / Rec) | `upright` (F1 / Prec / Rec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D Best Practical (42f)** | 0,651 / 0,592 / 0,724 | 0,521 / 0,552 / 0,493 | 0,821 / 0,873 / 0,774 | **0,984 / 0,968 / 1,000** | **0,587 / 0,727 / 0,492** | **0,660 / 0,592 / 0,744** |
| 2D Core Baseline (36f) | 0,667 / 0,606 / 0,741 | 0,478 / 0,524 / 0,440 | **0,838 / 0,891 / 0,790** | **0,984 / 0,968 / 1,000** | 0,500 / 0,546 / 0,462 | 0,652 / 0,604 / 0,707 |
| 3D Camera Frame (25f) | **0,680 / 0,810 / 0,586** | **0,544 / 0,680 / 0,453** | 0,740 / 0,772 / 0,710 | 0,686 / 0,532 / 0,967 | 0,483 / 0,527 / 0,446 | 0,577 / 0,557 / 0,598 |
| 3D Canonical Body (22f) | 0,144 / 0,151 / 0,138 | 0,253 / 0,241 / 0,267 | 0,544 / 0,596 / 0,500 | 0,393 / 0,314 / 0,525 | 0,250 / 0,333 / 0,200 | 0,372 / 0,392 / 0,354 |

Pola kesalahan klasifikasi antarkelas divisualisasikan melalui matriks konfusi perbandingan pada Gambar 3.4.

![Gambar 3.4 Matriks Konfusi Komparatif 2D vs 3D](file:///d:/.Candra/Project/TA/07_results/experiments/private_final/comparison/confusion_matrix_comparison.png)
**Gambar 3.4** Matriks konfusi perbandingan (*side-by-side*) antara Model 2D Multi-View (kiri) dan Model Stereo 3D Camera Frame (kanan) pada subset irisan data valid yang sama (N = 403 sampel).

#### 3.4.1 Evaluasi Separabilitas Postur Bidang Koronal (Leaning Left dan Leaning Right)
Pada bidang koronal (deviasi lateral tulang belakang), pendekatan **2D Multi-View mendominasi performa klasifikasi**:
* Postur `leaning_right` dikenali nyaris sempurna oleh model 2D dengan F1-score **0,9839** (Recall 100% dan Precision 96,83%).
* Postur `leaning_left` juga mencapai F1-score tinggi sebesar **0,8205–0,8376**.

Secara biomekanik, pergeseran lateral tulang belakang terproyeksi secara ortogonal langsung pada bidang citra Kamera Depan (CAM01). Fitur 2D sederhana seperti sudut kemiringan bahu (*shoulder slope*) dan inklinasi lateral torso sudah memberikan batas keputusan linier yang sangat tegas. Penambahan dimensi kedalaman $Z$ pada model 3D tidak memberikan keuntungan pada bidang lateral; sebaliknya, pada model 3D Camera performa `leaning_right` justru terdegradasi ke angka $0{,}6860$ akibat noise asimetri kedalaman saat bahu kanan subjek mendekati kamera lateral.

#### 3.4.2 Dinamika Bidang Sagital: Keunggulan 3D pada Leaning Forward
Menariknya, pada bidang sagital murni, rekonstruksi stereo 3D menunjukkan keunggulan yang konsisten:
* Pada kelas `leaning_forward`, model 3D Camera mencapai F1-score **0,5440** (Precision **0,6800**), melampaui 2D Core (F1 = 0,4783) dan 2D Best (F1 = 0,5211).
* Pada kelas `leaning_backward`, model 3D Camera mencapai Precision tertinggi sebesar **0,8095** (F1 = 0,6800).

Kondisi ini dapat dijelaskan secara fisik: ketika subjek condong ke depan atau ke belakang sepanjang sumbu kedalaman optik, fitur metrik 3D (`torso_sagittal_lean_deg` dan `head_depth_offset_norm`) mengukur translasi spasial nyata dalam satuan milimeter. Pengukuran metrik ini terbebas dari ambiguitas proyeksi perspektif citra atau variasi ketebalan pakaian subjek yang kerap mengaburkan deteksi 2D pada bidang sagital.

#### 3.4.3 Analisis Kerancuan Kronis: Upright vs Slouching
Inspeksi terhadap matriks konfusi pada Gambar 3.4 membongkar sumber kerancuan utama sistem:
* Dari 65 sampel ground truth `slouching`, model 2D Core salah mengklasifikasikan 18 sampel sebagai `leaning_forward` dan 11 sampel sebagai `upright`.
* Pada model 2D Best Practical, penambahan 6 fitur telinga sagital berhasil menekan angka *false positive* secara dramatis, mendongkrak nilai **Precision kelas `slouching` dari 54,55% menjadi 72,73% (+18,18%)** dan menghasilkan F1-score slouching tertinggi sebesar **0,5872**.
* Meskipun demikian, ambiguitas antara duduk tegak (*upright*) dan duduk merosot (*slouching*) tetap menjadi tantangan inheren pada subjek dengan kelenturan tulang belakang tinggi, di mana transisi postur berlangsung secara gradual tanpa perubahan sudut gelang bahu yang ekstrem.

---

### 3.5 Evaluasi Kesiapan Operasional dan Uji Latensi Real-Time

Untuk menguji kesiapan sistem dalam penerapan pemantauan postur kerja langsung (*live monitoring*), seluruh komponen pipeline model terbaik diserialisasi ke dalam format artefak deployment mandiri (`model.pkl`, `scaler.pkl`, `feature_schema.json`, dan `class_map.json`).

#### 3.5.1 Profil Durasi Komputasi End-to-End pada CPU
Pengujian latensi waktu eksekusi dilakukan pada satu siklus inferensi *end-to-end* menggunakan perangkat laptop berbasis CPU standar (Intel Core i5) tanpa akselerasi kartu grafis tambahan (GPU). Rincian profil komputasi per modul disajikan pada Tabel 3.5.

**Tabel 3.5** Profil latensi komputasi *end-to-end* per siklus inferensi pada lingkungan komputasi CPU lokal
| Tahapan Komputasi dalam Pipeline | Rata-rata Durasi (Milidetik) | Kontribusi Durasi (%) | Keterangan Operasional |
| :--- | :---: | :---: | :--- |
| 1. Sinkronisasi Akuisisi Frame (CAM01 & CAM02) | 14,8 ms | 7,73% | Pengambilan frame USB multi-thread |
| 2. Inferensi Pose YOLOv8n-Pose (2 Kamera) | 145,2 ms | 75,82% | Ekstraksi 17 keypoint per frame |
| 3. Rekayasa Fitur Geometri (2D Best 42 Fitur) | 18,1 ms | 9,45% | Kalkulasi sudut, rasio, dan normalisasi |
| 4. Skalasi Fitur & Klasifikasi XGBoost | 13,4 ms | 7,00% | Prediksi probabilistik multikelas |
| **Total Siklus Inferensi End-to-End** | **191,5 ms** | **100,00%** | **Throughput Operasional: 5,22 FPS** |

Tahap inferensi pose YOLOv8n pada dua kamera simultan menjadi komponen pemakan waktu terbesar (75,82% dari total siklus). Meskipun demikian, total durasi eksekusi *end-to-end* hanya membutuhkan rata-rata **$191{,}5\text{ ms}$**, yang setara dengan kecepatan throughput pemrosesan **$5{,}22\text{ Frames Per Second (FPS)}$**.

#### 3.5.2 Estimasi Throughput dan Demonstrasi Antarmuka GUI
Capaian throughput sebesar $5{,}22\text{ FPS}$ telah melampaui batas ambang minimum pemantauan postur duduk ergonomis ($\ge 5\text{ FPS}$). Berbeda dengan pelacakan gerak olahraga dinamis yang menuntut 30–60 FPS, perubahan postur duduk kerja manusia berlangsung secara lambat (*quasi-static motion*, dalam hitungan detik hingga menit). Throughput $\approx 5\text{ FPS}$ sangat memadai untuk memberikan peringatan dini (*real-time feedback*) tanpa menimbulkan jeda (*lag*) yang mengganggu pengguna. 

Bukti fungsional implementasi antarmuka pemantauan langsung pada sistem ditampilkan pada Gambar 3.5.

![Gambar 3.5 Demonstrasi Antarmuka Inferensi Real-Time](file:///d:/.Candra/Project/TA/07_results/experiments/live_camera_test_snapshot.jpg)
**Gambar 3.5** Tangkapan layar antarmuka sistem inferensi waktu-nyata (*live split-screen GUI*): CAM01 Frontal (kiri) dan CAM02 Lateral (kanan) dengan overlay titik kunci kerangka tubuh dan kotak status prediksi postur duduk.

#### 3.5.3 Analisis Trade-Off Praktis: 2D Multi-View vs Stereo 3D untuk Penerapan Nyata
Berdasarkan seluruh bukti eksperimen kuantitatif dan pengujian operasional, perbandingan komprehensif antara paradigma Multi-View 2D dan Stereo 3D dirangkum pada Tabel 3.6.

**Tabel 3.6** Matriks perbandingan trade-off teknis antara 2D Multi-View dan Stereo 3D
| Kriteria Evaluasi | 2D Multi-View (2D Best 42f) | Stereo 3D (Camera Frame 25f) | Implikasi untuk Mitigasi Skoliosis |
| :--- | :---: | :---: | :--- |
| **Kinerja Macro F1-Score** | **0,7039 (Unggul)** | 0,6182 (Tertinggal $-8{,}57\%$) | 2D memberikan diskriminasi postur yang lebih akurat |
| **Stabilitas Generalisasi Antarsubjek** | **Tinggi ($\sigma = 0{,}0701$)** | Rendah ($\sigma = 0{,}2519$) | 2D lebih konsisten saat menghadapi pengguna baru |
| **Cakupan Data yang Dapat Diproses** | **96,84% (704/727 data)** | 55,43% (403/727 data) | 2D jauh lebih tahan terhadap oklusi parsial |
| **Sensitivitas Kalibrasi Rig** | **Rendah (Invarian pergeseran)** | Sangat Tinggi (Rentan pergeseran rig) | 2D tidak memerlukan kalibrasi papan catur berkala |
| **Beban Komputasi Inferensi** | Ringan (Normalisasi citra 2D) | Lebih Berat (Triangulasi DLT/SVD metrik) | 2D lebih hemat daya komputasi CPU |

Temuan ini membawa kesimpulan strategis bagi perancangan sistem mitigasi postur skoliosis: **Pendekatan Multi-View 2D terbukti lebih superior, lebih tangguh (*robust*), dan lebih ekonomis dibanding Stereo 3D untuk implementasi operasional dunia nyata**. 

Meskipun Stereo 3D menawarkan representasi spasial metrik yang elegan di atas kertas, ketergantungannya yang sangat sensitif terhadap kalibrasi ekstrinsik kaku, tingginya angka penolakan data akibat oklusi anatomis kamera samping (menyebabkan 44,57% data terbuang), serta variabilitas performa yang tinggi antarsubjek menjadikannya kurang praktis untuk pemantauan mandiri di workstation kantor atau sekolah. Dengan demikian, sistem berbasis **Multi-View 2D Keypoint (42 fitur)** ditetapkan sebagai arsitektur final yang paling optimal untuk sistem mitigasi risiko skoliosis berbasis kamera terjangkau.
