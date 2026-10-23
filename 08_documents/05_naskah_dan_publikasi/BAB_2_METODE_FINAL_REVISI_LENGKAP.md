# BAB II METODOLOGI PENELITIAN

Penelitian ini bertujuan untuk mengembangkan dan mengevaluasi sistem klasifikasi postur duduk berbasis representasi *Multi-View 2D Keypoint* dan *Stereo 3D Reconstruction* sebagai upaya perancangan sistem mitigasi risiko skoliosis berbasis kamera. Tahap penelitian dirancang secara sistematis yang dimulai dari kalibrasi kamera, pengumpulan dataset, estimasi pose dan seleksi target, rekayasa fitur (*feature engineering*), pemodelan klasifikasi terkontrol, hingga evaluasi performa dan pengujian operasional waktu-nyata (*real-time*). Seluruh sistem dikembangkan dan diuji menggunakan konfigurasi dua kamera stereo terkalibrasi pada lingkungan komputasi lokal. Alur kerja penelitian ditunjukkan pada Gambar 2.1 berikut:

![Gambar 2.1 Alur Kerja Metodologi Penelitian](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_1_pipeline_penelitian.png)
**Gambar 2.1** Alur kerja metodologi penelitian deteksi dan klasifikasi postur duduk berisiko skoliosis, membandingkan representasi *Multi-View 2D* dan *Stereo 3D* menggunakan model XGBoost.

---

### A. Kalibrasi Kamera Stereo (*Stereo Camera Calibration*)

Tahap awal penelitian dimulai dengan pemasangan dan kalibrasi sistem kamera ganda (*dual-camera setup*) menggunakan dua kamera web USB (*dual USB webcams*) beresolusi Full HD ($1920 \times 1080$ piksel). Dua kamera dipasang pada posisi depan (*frontal view*, CAM01) dan samping (*lateral view*, CAM02) pada rig kaku dengan sudut konvergen sudut lebar (*wide-baseline* $\sim 90^\circ$). Jarak dan tinggi rata-rata kamera diatur sekitar 1 meter menghadap ke subjek sehingga diperoleh bidang pandang tumpang tindih (*overlapping field of view*) yang mencakup seluruh tubuh bagian atas subjek tanpa terhalang (*occlusion*), sebagaimana diilustrasikan pada Gambar 2.2.

![Gambar 2.2 Konfigurasi Penempatan Kamera Ganda](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_2_setup_kamera.png)
**Gambar 2.2** Diagram tata letak sistem kamera ganda konvergen menghadap meja kerja dan subjek pada kursi tanpa sandaran.

Untuk memungkinkan komputer membaca karakteristik lensa serta menghitung posisi relatif antarkamera, dilakukan proses kalibrasi menggunakan pola papan catur (*checkerboard*) berdimensi $9 \times 6$ titik sudut interior ($54$ titik sudut) dengan ukuran kotak berskala $25\text{ mm}$. Citra papan catur yang direkam bersamaan dari berbagai orientasi sudut kemiringan ($\ge 30$ pasangan citra) diolah untuk memodelkan parameter optik lensa (panjang fokus dan distorsi) serta relasi geometris spasial antara kamera depan dan samping sebagai dasar rekonstruksi tiga dimensi.

Hubungan matematis antara koordinat ruang 3D dunia $\mathbf{X}_w = [X, Y, Z, 1]^T$ dengan koordinat proyeksi piksel 2D pada bidang citra $\mathbf{x} = [u, v, 1]^T$ dimodelkan berdasarkan model kamera pinhole:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} R & T \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} \qquad (2.1)
$$

di mana $s$ menyatakan faktor skala proyeksi, matriks parameter intrinsik kamera $K$ didefinisikan sebagai:

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \qquad (2.2)
$$

dengan $f_x$ dan $f_y$ merupakan panjang fokus (*focal length*) dalam satuan piksel, serta $c_x$ dan $c_y$ adalah koordinat titik utama (*principal point*). Penyimpangan kelengkungan optik lensa dikoreksi menggunakan vektor koefisien distorsi radial ($k_1, k_2$) dan tangensial ($p_1, p_2$):

$$
D = \begin{bmatrix} k_1 & k_2 & p_1 & p_2 \end{bmatrix} \qquad (2.3)
$$

Relasi geometris spasial antara Kamera Depan (CAM01) dan Kamera Samping (CAM02) dimodelkan melalui matriks rotasi stereo $R_{\text{stereo}}$ dan vektor translasi stereo $T_{\text{stereo}}$:

$$
\mathbf{X}_{c2} = R_{\text{stereo}} \mathbf{X}_{c1} + T_{\text{stereo}} \qquad (2.4)
$$

Kualitas hasil kalibrasi divalidasi dengan ambang batas kesalahan proyeksi ulang rata-rata (*mean reprojection error*) di bawah $1{,}5\text{ piksel}$ guna menjamin presisi geometri triangulasi stereo.

---

### B. Pengumpulan Data (*Data Collection*)

Pengumpulan data menggunakan perangkat lunak berbasis bahasa pemrograman Python (`capture_multicam.py`) yang dirancang dengan mekanisme *multi-threading* untuk menangkap citra subjek secara sinkron dari kedua kamera dengan latensi antar-kamera yang minimal. Sistem perekaman dilengkapi dengan pencatatan metadata konfigurasi sesi sebelum pengambilan data, meliputi `subject_id`, `session_id`, `calibration_id`, `chair_id`, dan `lateral_side`.

Pengumpulan dataset melibatkan 24 subjek sukarelawan (`S001` hingga `S024`) yang memiliki variasi karakteristik fisik antropometri tubuh (tinggi badan, rentang lebar bahu, dan panjang torso), khususnya pada subjek dengan rentang usia 18 hingga 22 tahun. Berbeda dengan pengujian laboratorium tertutup, akuisisi data sengaja dilakukan pada lingkungan semi-terbuka dan ruang publik guna menguji ketahanan model terhadap latar belakang yang dinamis dan variasi fluktuasi pencahayaan alami. Fasilitas duduk yang digunakan didominasi oleh kursi bangku tanpa sandaran (*backless stool*, `CHR_001`) untuk mencegah oklusi sandaran terhadap profil kelengkungan tulang belakang saat direkam oleh kamera samping.

Perekaman dilakukan secara sinkron antara kamera depan dan samping, di mana setiap partisipan diminta memperagakan variasi kebiasaan duduk kerja yang terstandardisasi ke dalam enam kelas target ergonomis:
1. `upright`: postur duduk tegak netral ergonomis;
2. `leaning_forward`: condong ke arah depan (indikasi fleksi berlebih / kecenderungan kifosis);
3. `leaning_backward`: bersandar atau condong ke arah belakang;
4. `leaning_left`: deviasi lateral tulang belakang ke sisi kiri (indikasi kecenderungan skoliosis kiri);
5. `leaning_right`: deviasi lateral tulang belakang ke sisi kanan (indikasi kecenderungan skoliosis kanan);
6. `slouching`: postur duduk merosot dengan panggul meluncur ke depan dan bahu membulat (*kyphotic slump*).

Terdapat juga kelas penolakan (*reject*) dengan karakteristik subjek berdiri, bergerak transisi, atau keluar dari *frame* yang digunakan agar sistem dapat mengenali posisi selain posisi duduk kerja yang valid. Sesi akuisisi ini menghasilkan total **885 rekaman pasangan citra mentah** (*raw captures*) dari 24 subjek. Dari jumlah tersebut, sebanyak **727 rekaman pasangan citra** merepresentasikan enam kelas postur duduk target utama, setelah menyaring 121 rekaman kelas di luar cakupan utama (seperti *forward head posture* terisolasi) serta 37 rekaman data anomali (*reject*). Seluruh rekaman disimpan berpasangan disertai pencatatan metadata relasional subjek dan kalibrasi secara terstruktur pada `captures.csv` dan `images.csv`.

---

### C. Estimasi Pose dan Seleksi Subjek Sasaran (*Pose Estimation & Target Person Selection*)

Tahap deteksi tubuh memproses pasangan citra dari kedua kamera menggunakan model jaringan saraf tiruan konvolusional ringan **YOLOv8n-Pose** (*pretrained* COCO) untuk mengekstraksi 17 titik kunci (*keypoints*) anatomis kerangka tubuh $\mathbf{k}_i = (x_i, y_i)$ beserta nilai tingkat keyakinannya (*confidence score* $c_i \in [0, 1]$).

Untuk mengantisipasi kemunculan individu lain di latar belakang (*bystanders*) pada lingkungan terbuka, dirancang algoritma penyeleksi sasaran (*target person selector*) deterministik yang bekerja berdasarkan:
1. Luas kotak pembatas (*bounding box*) terbesar pada area kerja utama (*Region of Interest* / ROI);
2. Verifikasi kelengkapan deteksi sendi tubuh bagian atas (kedua bahu dan panggul);
3. Konsistensi pasangan identitas antarmuka stereo (*stereo cross-view consistency*).

Data kerangka tubuh yang terdeteksi kemudian disaring melalui protokol kontrol kualitas bertingkat (*two-layer quality control*):
* **Stereo-QC:** Memastikan tingkat nilai *confidence score* seluruh titik kunci sendi inti memenuhi ambang batas kelayakan $c_i \ge 0{,}5$ serta konsistensi tumpang-tindih kotak pembatas stereo.
* **Feature-QC:** Mewajibkan keterbacaan deteksi sendi panggul kanan (`right_hip`) pada kedua kamera tanpa pengisian nilai nol (*zero imputation*) guna menjaga integritas topologi ruang spasial.

Penerapan protokol kontrol kualitas bertingkat ini menyaring 727 tangkapan enam kelas menjadi **403 sampel pasangan citra valid** yang berasal dari **18 subjek partisipan** (`S001`–`S006`, `S011`–`S019`, `S021`, `S022`, dan `S024`). Sebanyak 6 subjek lainnya tereliminasi secara otomatis karena mengalami kendala oklusi sendi panggul kanan pada kamera samping akibat sudut pandang paha saat duduk. Pasangan citra yang memenuhi kedua kriteria kelayakan tersebut dikelompokkan sebagai **subset irisan valid (*intersection subset*)** sebagai dasar perbandingan terkontrol yang setara (*apple-to-apple comparison*) antara representasi 2D dan 3D.

---

### D. Rekayasa Fitur Geometri (*Feature Engineering*)

Tahap ini bertujuan mentransformasikan koordinat titik kunci piksel mentah hasil estimasi pose menjadi representasi geometris yang bermakna secara anatomis. Koordinat piksel mentah tidak dapat digunakan secara langsung karena nilainya sangat sensitif terhadap variasi jarak subjek ke kamera, pergeseran posisi duduk, serta perbedaan dimensi fisik antar-individu. Oleh karena itu, dirancang fitur-fitur invarian berbasis sudut kemiringan anatomis dan rasio jarak ternormalisasi yang merepresentasikan postur tubuh secara objektif. Rekayasa fitur ini dipecah ke dalam dua pendekatan komparatif: representasi *Multi-View 2D* dan representasi spasial *Stereo 3D*.

#### 1. Rekayasa Fitur *Multi-View* 2D
Pada pendekatan ini, informasi spasial diekstraksi secara terpisah dari bidang proyeksi citra dua dimensi Kamera Depan (*frontal view*, CAM01) dan Kamera Samping (*lateral view*, CAM02) sebelum digabungkan ke dalam satu vektor fitur tabular, seperti diilustrasikan pada Gambar 2.3.

![Gambar 2.3 Skema Ekstraksi Fitur Geometri 2D](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_3_fitur_geometri_2d.png)
**Gambar 2.3** Skema ekstraksi fitur geometris 2D multi-sudut pandang: (a) Proyeksi frontal CAM01, dan (b) Proyeksi lateral CAM02.

Untuk mencapai invarian terhadap skala (*scale-invariance*), seluruh jarak piksel dinormalisasi terhadap panjang vektor torso pada bidang citra masing-masing kamera:

$$
d_{\text{norm}} = \frac{d_{\text{piksel}}}{\|\mathbf{v}_{\text{torso}}\|} \qquad (2.5)
$$

di mana vektor torso $\mathbf{v}_{\text{torso}}$ dihitung dari titik tengah kedua bahu ke titik tengah kedua panggul:

$$
\mathbf{v}_{\text{torso}} = \mathbf{p}_{\text{mid-shoulder}} - \mathbf{p}_{\text{mid-hip}} \qquad (2.6)
$$

Sudut kemiringan garis bahu (*shoulder slope*) dan panggul (*hip slope*) dihitung terhadap sumbu horizontal citra:

$$
\theta_{\text{slope}} = \arctan\left(\frac{y_{\text{kanan}} - y_{\text{kiri}}}{x_{\text{kanan}} - x_{\text{kiri}}}\right) \times \frac{180^\circ}{\pi} \qquad (2.7)
$$

Sedangkan sudut inklinasi torso dihitung terhadap garis vertikal gravitasi citra ($\mathbf{v}_{\text{vertikal}} = [0, -1]^T$):

$$
\theta_{\text{torso}} = \arccos\left(\frac{\mathbf{v}_{\text{torso}} \cdot \mathbf{v}_{\text{vertikal}}}{\|\mathbf{v}_{\text{torso}}\| \|\mathbf{v}_{\text{vertikal}}\|}\right) \times \frac{180^\circ}{\pi} \qquad (2.8)
$$

Ekstraksi fitur 2D ini dievaluasi ke dalam dua varian komparatif:
* **Varian 2D Core Baseline (36 Fitur):** Merupakan representasi dasar multi-sudut pandang yang menggabungkan 18 fitur dari CAM01 dan 18 fitur dari CAM02. Fitur inti ini mencakup sudut kemiringan garis bahu, sudut kemiringan panggul, sudut inklinasi torso terhadap garis vertikal citra, sudut ruang kepala-torso, serta rasio jarak relatif horizontal dan vertikal sendi utama yang dinormalisasi terhadap panjang torso.
* **Varian 2D Best Practical (42 Fitur):** Mengembangkan 36 fitur inti dengan menambahkan 6 fitur geometri telinga terhadap bahu pada proyeksi sagital yang diekstraksi khusus dari Kamera Samping (CAM02). Fitur tambahan ini mencakup jarak relatif horizontal telinga ke bahu ($\Delta x / \|\mathbf{v}_{\text{torso}}\|$), jarak relatif vertikal telinga ke bahu ($\Delta y / \|\mathbf{v}_{\text{torso}}\|$), jarak Euclidean ternormalisasi telinga-bahu, serta sudut inklinasi leher/telinga sagital. Penambahan ini dirancang khusus untuk menangkap deviasi kepala yang condong ke depan (*forward head translation*) pada postur merosot (*slouching*) yang sering kali luput jika hanya mengandalkan sudut inklinasi torso global.

#### 2. Rekonstruksi dan Rekayasa Fitur Stereo 3D
Pendekatan kedua memanfaatkan hubungan geometris antarkamera untuk merekonstruksi titik kunci ke dalam ruang metrik nyata tiga dimensi. Pasangan titik kunci 2D dari Kamera Depan $(\mathbf{x}_1)$ dan Kamera Samping $(\mathbf{x}_2)$ ditriangulasi menggunakan metode *Direct Linear Transform* (DLT) berbasis matriks proyeksi kamera terkalibrasi $P_1 = K_1 [I \mid \mathbf{0}]$ dan $P_2 = K_2 [R_{\text{stereo}} \mid T_{\text{stereo}}]$:

$$
\begin{bmatrix} 
u_1 P_{1,3}^T - P_{1,1}^T \\
v_1 P_{1,3}^T - P_{1,2}^T \\
u_2 P_{2,3}^T - P_{2,1}^T \\
v_2 P_{2,3}^T - P_{2,2}^T 
\end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \mathbf{0} \qquad (2.9)
$$

Sistem persamaan homogen $A \mathbf{X} = \mathbf{0}$ diselesaikan menggunakan *Singular Value Decomposition* (SVD) untuk menghasilkan koordinat metrik Euklides 3D $\mathbf{X} = [X, Y, Z]^T$ dalam satuan milimeter. Fitur spasial 3D ini kemudian dimodelkan ke dalam dua varian sistem koordinat:

* **3D Camera Frame (25 Fitur):** Fitur dihitung langsung berdasarkan orientasi sistem koordinat optik kamera stereo dengan menempatkan titik tengah panggul ($\mathbf{p}_{\text{mid-hip}}$) sebagai titik acuan origin $(0, 0, 0)$. Varian ini terdiri dari:
  1. 15 koordinat metrik Euklides 3D $(X, Y, Z)$ dari lima sendi inti tubuh atas (hidung, bahu kiri, bahu kanan, panggul kiri, dan panggul kanan);
  2. 2 sudut putar (*roll*) garis bahu dan panggul pada bidang koronal kamera (dalam derajat);
  3. 3 sudut kemiringan torso yang mencakup kemiringan lateral (*lateral lean*), kemiringan sagital (*sagittal lean*), dan sudut inklinasi 3D ruang total terhadap sumbu vertikal (dalam derajat);
  4. 1 sudut ruang kepala-torso 3D antara vektor kepala-leher dan vektor torso (dalam derajat);
  5. 4 rasio deviasi dan asimetri kedalaman tanpa dimensi (*dimensionless normalized ratios*), yaitu rasio *offset* kedalaman kepala, rasio *offset* lateral kepala, rasio asimetri kedalaman antarbahu, dan rasio asimetri kedalaman antarpanggul yang dinormalisasi terhadap panjang vektor torso 3D.
* **3D Canonical Body Frame (22 Fitur):** Untuk mengatasi ketergantungan model terhadap orientasi fisik rig kamera, koordinat 3D ditransformasikan ke kerangka acuan tubuh anatomis lokal menggunakan ortogonalisasi Gram-Schmidt:

$$
\mathbf{Y}_{\text{can}} = \frac{\mathbf{p}_{\text{mid-shoulder}} - \mathbf{p}_{\text{mid-hip}}}{\|\mathbf{p}_{\text{mid-shoulder}} - \mathbf{p}_{\text{mid-hip}}\|}, \quad \mathbf{X}_{\text{can}} = \frac{\mathbf{v}_{\text{hip}} - (\mathbf{v}_{\text{hip}} \cdot \mathbf{Y}_{\text{can}})\mathbf{Y}_{\text{can}}}{\|\mathbf{v}_{\text{hip}} - (\mathbf{v}_{\text{hip}} \cdot \mathbf{Y}_{\text{can}})\mathbf{Y}_{\text{can}}\|}, \quad \mathbf{Z}_{\text{can}} = \mathbf{X}_{\text{can}} \times \mathbf{Y}_{\text{can}} \qquad (2.10)
$$

Akibat penyelarasan koordinat ke tubuh ini, tiga fitur geometris secara matematis mengalami degenerasi nilai konstan nol dengan varians nol ($\sigma = 0$):
1. Sudut kemiringan lateral torso bernilai identik $0^\circ$ karena sumbu longitudinal tubuh dijadikan basis sumbu vertikal lokal $Y_{\text{can}}$;
2. Sudut kemiringan sagital torso bernilai identik $0^\circ$ karena vektor torso berimpit penuh pada sumbu $Y_{\text{can}}$;
3. Rasio asimetri kedalaman panggul (*hip depth asymmetry*) bernilai identik 0 tanpa dimensi karena kedua sendi panggul dikonstruksikan sejajar pada bidang $X_{\text{can}}$ tanpa komponen kedalaman pada sumbu $Z_{\text{can}}$.

Ketiga fitur ber-varians nol tersebut dieliminasi sehingga menghasilkan 22 fitur ruang murni yang sepenuhnya berorientasi pada tubuh subjek.

---

### E. Pelatihan dan Evaluasi Model Terkontrol (*Model Training & Controlled Evaluation*)

Tahap klasifikasi menggunakan algoritma **XGBoost Classifier** (*Extreme Gradient Boosting*) yang dirancang untuk mengklasifikasikan enam kelas postur duduk secara probabilistik. Konfigurasi hiperparameter dikunci secara terstandarisasi untuk seluruh varian fitur: fungsi objektif multikelas `multi:softprob`, kedalaman pohon `max_depth = 6`, jumlah pohon `n_estimators = 300`, laju pembelajaran `learning_rate = 0.1`, dan penetapan *random seed* = 42.

Untuk menjamin pengujian bebas kebocoran data (*zero-leakage*), diterapkan protokol **5-Fold Subject-Stratified Cross-Validation**, di mana partisi data latih dan uji diisolasi sepenuhnya pada tingkat subjek partisipan sehingga model benar-benar mengevaluasi generalisasi terhadap individu baru (*unseen subjects*), sebagaimana diilustrasikan pada Gambar 2.4.

![Gambar 2.4 Skema Partisi 5-Fold Subject-Stratified CV](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_4_skema_5fold_cv.png)
**Gambar 2.4** Skema pembagian lipatan *5-Fold Subject-Stratified Cross-Validation* yang menjamin ketiadaan kebocoran data antarsubjek.

Transformasi standarisasi skala fitur (`StandardScaler`) di-*fit* secara eksklusif pada data latih di setiap lipatan (*fold*) sebelum diterapkan (*transform*) ke data pengujian:

$$
z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}} \qquad (2.11)
$$

Keempat varian representasi fitur (2D Core, 2D Best, 3D Camera Frame, dan 3D Canonical Frame) dievaluasi pada subset irisan sampel yang sama persis (403 sampel dari 18 subjek) untuk menjamin perbandingan yang setara dan objektif (*fair comparison*). **Macro F1-score pooled out-of-fold (OOF)** ditetapkan sebagai **metrik evaluasi utama** (*primary performance metric*) guna memberikan bobot evaluasi yang seimbang pada setiap kelas postur tanpa terdistorsi oleh perbedaan jumlah sampel:

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c} \qquad (2.12)
$$

$$
F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} \qquad (2.13)
$$

$$
\text{Macro } F1_{\text{pooled}} = \frac{1}{C} \sum_{c=1}^{C} F1_c \qquad (2.14)
$$

$$
\text{Akurasi} = \frac{\sum_{c=1}^C TP_c}{N_{\text{total}}} \qquad (2.15)
$$

Selain metrik pooled OOF, stabilitas dan konsistensi kinerja klasifikasi postur antarsubjek diukur menggunakan simpangan baku antarlipatan (*Fold Standard Deviation*, $\sigma_{\text{fold}}$) yang dihitung dari Macro F1-score pada masing-masing dari lima fold pengujian ($K=5$):

$$
\sigma_{\text{fold}} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (F1_k - \overline{F1})^2} \qquad (2.16)
$$

---

### F. Pengujian Operasional Inferensi dan Latensi *Real-Time* (*Operational Latency & Deployment Protocol*)

Sebagai bagian integral dari perancangan sistem pemantauan postur duduk preventif, tahap akhir metodologi merancang arsitektur implementasi sistem ke lingkungan operasional nyata:
1. **Serialisasi Pipeline Model:** Seluruh komponen inferensi terlatih diekspor ke dalam berkas artefak deployment mandiri, yang mencakup model XGBoost terbaik, skaler normalisasi (`scaler.pkl`), skema urutan fitur (`feature_schema.json`), dan kamus pemetaan kelas target (`class_map.json`).
2. **Prosedur Pengukuran Latensi Komputasi:** Pengujian kecepatan eksekusi diukur secara terisolasi pada siklus inferensi *end-to-end* (akuisisi frame stereo sinkron $\to$ estimasi pose kerangka YOLOv8n $\to$ ekstraksi fitur geometri $\to$ prediksi kelas XGBoost) pada lingkungan CPU lokal standar tanpa akselerasi kartu grafis eksternal (GPU).
3. **Parameter Kinerja Operasional:** Efisiensi komputasi diukur melalui rata-rata durasi per siklus inferensi (dalam milidetik, ms) dan estimasi *throughput* pemrosesan dalam *Frames Per Second* (FPS):

$$
\text{FPS} = \frac{1000}{\text{Latensi}_{\text{end-to-end}}\text{ (ms)}} \qquad (2.17)
$$

Prosedur pengukuran ini dirancang pada Bab Metode guna memverifikasi apakah pipeline yang diusulkan mampu memenuhi ambang batas waktu-nyata (*real-time processing* $\ge 5\text{ FPS}$) sebelum hasil pengukuran kuantitatif empirisnya dilaporkan secara rinci pada Bab Hasil dan Pembahasan.
