# BAB IV KESIMPULAN DAN SARAN

Bab ini menyajikan rangkuman menyeluruh dari hasil perancangan, implementasi, dan pengujian empiris sistem klasifikasi postur duduk berbasis visi komputer yang telah dibahas pada bab-bab sebelumnya. Kesimpulan dirumuskan secara terstruktur untuk menjawab tujuan dan rumusan masalah penelitian, menyajikan temuan ilmiah komparatif antara representasi *Multi-View 2D* dan *Stereo 3D*, serta mengevaluasi kesiapan operasional sistem. Selanjutnya, bab ini menguraikan keterbatasan (*limitations*) yang dihadapi selama penelitian serta menyajikan rekomendasi saran praktis maupun teoritis untuk pengembangan sistem mitigasi risiko skoliosis di masa mendatang.

---

### 4.1 Kesimpulan

Berdasarkan hasil analisis data, permodelan klasifikasi dengan *5-Fold Subject-Stratified Cross-Validation*, serta pengujian waktu-nyata (*real-time benchmark*) yang telah dilakukan, dapat ditarik beberapa kesimpulan utama sebagai berikut:

#### 4.1.1 Kesimpulan Terhadap Capaian Tujuan Penelitian
1. **Pengembangan Pipeline Klasifikasi Berbasis Visi Ganda:**
   Telah berhasil dibangun sebuah alur kerja (*pipeline*) klasifikasi postur duduk terintegrasi berbasis sistem kamera ganda terkalibrasi (*dual-view camera system*) dan model deteksi pose ringan YOLOv8n-Pose. Sistem mampu mengekstraksi titik-titik anatomis kerangka tubuh secara simultan dari sudut pandang frontal (CAM01) dan lateral (CAM02), melakukan penyaringan otomatis berbasis kontrol kualitas bertingkat (*two-layer QC*), dan mengklasifikasikan postur tubuh ke dalam enam kelas target ergonomis (`upright`, `leaning_forward`, `leaning_backward`, `leaning_left`, `leaning_right`, dan `slouching`) menggunakan pengklasifikasi *Extreme Gradient Boosting* (XGBoost).

2. **Kinerja Representasi Terbaik (*2D Best Practical*):**
   Model dengan representasi **2D Best Practical (42 fitur)** terbukti mencapai performa klasifikasi tertinggi di antara seluruh varian yang dievaluasi. Pada subset irisan data valid (*intersection subset*, $N = 403$ pasangan citra dari 18 subjek), model ini mencatatkan nilai **Akurasi sebesar 69,73%** dan **Macro F1-Score (Pooled OOF) sebesar 0,7039 (70,39%)**, dengan rata-rata antarlipatan (*Fold Mean F1*) sebesar **68,75%**. Capaian ini secara signifikan melampaui varian *2D Core Baseline* (Macro F1 = 0,6863) serta model spasial *3D Camera Frame* (Macro F1 = 0,6182) dengan selisih keunggulan sebesar $+8{,}57\%$.

3. **Peran Kritis Fitur Telinga Sagital:**
   Integrasi enam fitur telinga lateral dari kamera samping (koordinat relatif, jarak Euclidean telinga-bahu ternormalisasi, dan sudut inklinasi servikal) terbukti secara empiris menyelesaikan kelemahan deteksi sudut torso global pada bidang sagital. Penambahan fitur ini berhasil menekan angka positif palsu (*false positive*) pada postur duduk merosot (*slouching*), mendongkrak nilai presisi kelas *slouching* secara drastis dari **54,55% menjadi 72,73%** (peningkatan $+18{,}18\%$) serta menghasilkan F1-score *slouching* tertinggi sebesar **0,5872**.

4. **Kesiapan Komputasi Operasional Waktu-Nyata:**
   Pengujian operasional pada perangkat laptop berprosesor CPU standar (Intel Core i5) tanpa akselerasi GPU menghasilkan rata-rata latensi eksekusi *end-to-end* sebesar **$191{,}5\text{ milidetik}$ per siklus inferensi**, yang setara dengan *throughput* sebesar **$5{,}22\text{ Frames Per Second (FPS)}$**. Karena perubahan postur duduk manusia tergolong gerakan lambat (*quasi-static motion*), *throughput* $\approx 5\text{ FPS}$ ini telah memenuhi batas ambang kelayakan operasional sistem pemantauan interaktif untuk memberikan umpan balik koreksi postur secara langsung.

---

#### 4.1.2 Temuan Kunci Ilmiah dan Komparasi Paradigma (2D Multi-View vs. Stereo 3D)
Penelitian ini menghasilkan temuan ilmiah fundamental mengenai komparasi representasi data spasial untuk analisis postur manusia:

1. **Superioritas Praktis Representasi Multi-View 2D Dibandingkan Stereo 3D:**
   Secara teoritis, rekonstruksi spasial 3D menawarkan deskripsi metrik posisi tubuh dalam satuan milimeter. Namun, dalam aplikasi nyata pemantauan postur duduk terjangkau, pendekatan *Multi-View 2D* terbukti jauh lebih unggul dan tangguh:
   * **Stabilitas Generalisasi Antarsubjek:** Varian *2D Best Practical* menunjukkan deviasi standar antarlipatan yang rendah ($\sigma_{\text{fold}} = 0{,}0701$ atau $\pm 7{,}01\%$), mengindikasikan ketahanan model saat diuji pada subjek baru (*unseen subjects*). Sebaliknya, model *3D Camera Frame* menunjukkan instabilitas tinggi dengan deviasi standar $\sigma_{\text{fold}} = 0{,}2519$ ($\pm 25{,}19\%$), di mana akurasinya anjlok pada lipatan pengujian dengan karakteristik fisik subjek tertentu.
   * **Separabilitas Bidang Koronal:** Deviasi tulang belakang lateral (`leaning_left` dan `leaning_right`) yang menjadi indikator primer skoliosis dapat dipisahkan secara optimal oleh proyeksi 2D kamera frontal (F1 mencapai **0,9839** untuk *leaning right* dan **0,8205** untuk *leaning left*). Penambahan sumbu kedalaman 3D tidak memberikan nilai tambah diskriminatif pada bidang koronal, melainkan memperkenalkan derau pengukuran (*depth noise*).

2. **Degradasi Kritis pada Normalisasi 3D Tubuh Lokal (*Loss of World Gravity Reference*):**
   Transformasi koordinat metrik 3D ke kerangka acuan lokal tubuh menggunakan ortonormalisasi Gram-Schmidt (*3D Canonical Body Frame*) mengalami kegagalan performa berat, dengan Macro F1-Score hanya sebesar **0,3259** dan akurasi **33,00%**. Investigasi geometris membuktikan bahwa penyejajaran sumbu longitudinal tubuh ke sumbu lokal tubuh justru **menghilangkan vektor gravitasi vertikal bumi**. Karena definisi postur ergonomis (seperti condong ke kiri, miring ke kanan, atau tegak lurus) bertumpu pada orientasi tubuh terhadap gaya gravitasi dunia, model lokal kanonikal kehilangan informasi kemiringan global dan gagal mengenali orientasi duduk subjek.

3. **Dilema Oklusi Lateral pada Triangulasi Stereo:**
   Triangulasi stereo mensyaratkan keterlihatan titik anatomis secara simultan pada kedua kamera. Pada setup kursi kerja, sendi panggul lateral (`right_hip`) pada kamera samping sangat rentan mengalami oklusi oleh paha atau lipatan pakaian subjek. Hal ini menyebabkan **44,57% rekaman data tereliminasi** pada tahap *Feature-QC* (6 dari 24 subjek terbuang). Sebaliknya, representasi 2D memiliki toleransi oklusi parsial yang jauh lebih fleksibel (*coverage* mencapai **96,84%**), menjadikannya pilihan yang jauh lebih andal untuk lingkungan kerja nyata tanpa laboratorium khusus.

---

### 4.2 Keterbatasan Penelitian (*Limitations*)

Meskipun sistem telah berhasil mencapai tujuan yang dirancang, terdapat sejumlah batasan metodologis dan teknis yang perlu dicatat sebagai konteks evaluasi:

#### 4.2.1 Keterbatasan Dataset dan Akuisisi Citra
1. **Ukuran dan Keragaman Sampel Valid:**
   Penyaringan kontrol kualitas yang ketat menghasilkan subset irisan valid sebesar 403 sampel pasangan citra dari 18 subjek partisipan. Meskipun jumlah ini memadai untuk analisis perbandingan terkontrol (*controlled cross-validation*), rentang usia partisipan masih terkonsentrasi pada dewasa muda (18–22 tahun). Variasi bentuk tubuh lansia, anak-anak, atau individu dengan kelainan struktural tulang belakang tingkat lanjut belum terwakili secara mendalam.
2. **Ketergantungan Rig Kamera Kaku:**
   Rekonstruksi geometri stereo 3D sangat sensitif terhadap kestabilan mekanis rig kamera. Pergeseran fisik kamera sekecil beberapa milimeter atau perubahan sudut orientasi tripod akibat getaran meja dapat merusak parameter ekstrinsik kalibrasi awal, yang berpotensi meningkatkan *reprojection error* dan membiaskan perhitungan triangulasi kedalaman.
3. **Kondisi Busana dan Oklusi Anatomis:**
   Penggunaan pakaian longgar (*loose clothing*) atau jaket tebal pada beberapa partisipan mengaburkan penanda sendi panggul dan sabuk bahu pada estimasi pose 2D, yang menjadi pemicu utama kegagalan *Feature-QC* pada kamera lateral.

#### 4.2.2 Keterbatasan Pemodelan dan Lingkungan Operasional
1. **Asumsi Fasilitas Duduk Tanpa Sandaran (*Backless Stool*):**
   Pengambilan data primer menggunakan kursi bangku tanpa sandaran (*stool*) guna memastikan kelengkungan tulang belakang dan pergeseran panggul dapat teramati penuh oleh kamera lateral. Pada kursi kerja nyata yang memiliki sandaran tinggi (*high-back office chair*), sandaran kursi akan menutupi punggung dan panggul subjek dari sudut pandang lateral, sehingga ekstraksi fitur sagital memerlukan penyesuaian khusus.
2. **Ketiadaan Sensor Referensi Gravitasi Eksternal:**
   Model 3D murni dalam penelitian ini tidak dilengkapi dengan sensor unit pengukuran inersia (*Inertial Measurement Unit* / IMU) atau akselerometer pada kamera/kursi. Akibatnya, sistem tidak memiliki referensi gravitasi absolut independen ketika transformasi tubuh lokal (*canonical body frame*) diterapkan.
3. **Sifat Analisis Statis (*Frame-by-Frame*):**
   Klasifikasi saat ini memproses fitur pada setiap pasangan citra secara independen (*snapshot frame*). Sistem belum mengintegrasikan pemodelan temporal (*time-series analysis*) untuk melacak durasi menetapnya suatu postur buruk (misal: apakah subjek hanya merosot selama 5 detik untuk mengambil barang, atau menetap membungkuk selama lebih dari 20 menit).

---

### 4.3 Saran (*Future Works*)

Berdasarkan kesimpulan dan keterbatasan yang telah diidentifikasi, diajukan beberapa saran rekomendasi yang dibagi menjadi aspek pengembangan metodologi ilmiah dan aspek penerapan praktis:

#### 4.3.1 Saran Pengembangan Ilmiah dan Metodologi
1. **Eksplorasi Kamera Kedalaman Aktif (RGB-D Sensor):**
   Untuk penelitian yang ingin mempertahankan keunggulan dimensi 3D tanpa kendala kalibrasi ekstrinsik kaku dan triangulasi manual, disarankan untuk mengeksplorasi sensor kedalaman aktif terintegrasi (*depth camera* seperti Intel RealSense atau Microsoft Azure Kinect). Sensor kedalaman langsung menghasilkan awan titik (*point cloud*) metrik tanpa terpengaruh oleh oklusi visual pasangan stereo ganda.
2. **Integrasi Vektor Gravitasi Virtual pada Kerangka Kanonikal:**
   Pada penelitian transformasi kanonikal di masa depan, disarankan untuk menyertakan sudut kemiringan rig terhadap gravitasi bumi (baik melalui sensor akselerometer perangkat keras maupun estimasi bidang lantai/meja melalui segmentasi visual) sebagai fitur input tambahan sebelum normalisasi koordinat dilakukan. Hal ini akan mempertahankan referensi arah gravitasi bumi meskipun koordinat titik telah diputar ke dalam acuan tubuh subjek.
3. **Penerapan Arsitektur Temporal Berbasis Rangkaian Waktu:**
   Mengembangkan model klasifikasi dari model tabular statis (XGBoost) menjadi model berbasis sekuensial temporal, seperti *Long Short-Term Memory* (LSTM), *Gated Recurrent Unit* (GRU), atau *Temporal Convolutional Network* (TCN). Pendekatan ini memungkinkan sistem menilai akumulasi beban biomekanik tulang belakang berdasarkan akumulasi durasi waktu postur buruk dipertahankan.
4. **Augmentasi Data dan Penyesuaian Pose untuk Kursi Bersandaran:**
   Melakukan ekspansi dataset pada variasi kursi kerja bersandaran (*ergonomic office chair*) serta melatih ulang detektor pose dengan teknik augmentasi sintetik agar deteksi titik tulang belakang tetap andal meskipun sebagian punggung tertutup oleh sandaran kursi.

#### 4.3.2 Saran Penerapan Sistem dan Aspek Praktis
1. **Pengembangan Sistem Pemantauan Mandiri Latar Belakang (*Background Desktop Agent*):**
   Memanfaatkan representasi *2D Best Practical* (42 fitur) yang ringan dan hemat daya ke dalam aplikasi desktop pemantau ergonomi kerja di latar belakang (*background service*). Aplikasi dapat memanfaatkan webcam bawaan laptop sebagai Kamera Depan dan sebuah kamera USB terjangkau di sisi meja sebagai Kamera Samping.
2. **Mekanisme Peringatan Ergonomis Adaptif dan Bertingkat:**
   Menerapkan logika notifikasi bertingkat: sistem tidak memberikan peringatan audio/visual saat pergeseran postur terjadi secara sesaat (< 1–2 menit), melainkan memicu peringatan persuasif (seperti panduan peregangan mikro / *micro-break exercise*) hanya jika postur berisiko skoliosis (`leaning_left` atau `leaning_right`) atau postur kifotik (`slouching` / `leaning_forward`) terdeteksi menetap selama durasi ambang tertentu (misal: > 15–20 menit).
3. **Mekanisme Kalibrasi Mandiri Otomatis (*Self-Calibration Without Checkerboard*):**
   Untuk mempermudah penggunaan oleh masyarakat awam di kantor atau rumah, sistem dapat dikembangkan dengan modul kalibrasi mandiri yang mengestimasi sudut pandang lateral dan frontal secara otomatis berdasarkan deteksi proporsi wajah dan lebar bahu subjek pada saat awal sesi duduk tegak (*calibration pose*), sehingga meniadakan keharusan penggunaan papan catur fisik.
