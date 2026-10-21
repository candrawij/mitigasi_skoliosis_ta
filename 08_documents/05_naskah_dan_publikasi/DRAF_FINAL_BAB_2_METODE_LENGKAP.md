# BAB II METODOLOGI PENELITIAN

Penelitian ini bertujuan untuk mengembangkan dan mengevaluasi efektivitas komparatif representasi *Multi-View 2D Keypoint* dan *Stereo 3D Reconstruction* dalam mengklasifikasikan enam kelas postur duduk berisiko gangguan muskuloskeletal dan tulang belakang (*skoliosis*). Tahapan penelitian dirancang secara sistematis yang meliputi kalibrasi kamera stereo, akuisisi dataset privat, estimasi pose dan seleksi subjek sasaran menggunakan YOLOv8n-Pose, rekayasa fitur geometri invarian, pelatihan dan evaluasi model XGBoost menggunakan protokol *5-Fold Subject-Stratified Cross-Validation*, hingga pengujian inferensi *real-time*. Seluruh alur kerja penelitian diilustrasikan secara komprehensif pada Gambar 1.

![Gambar 1. Alur Kerja Metodologi Penelitian (Research Pipeline)](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_1_pipeline_penelitian.png)
**Gambar 1.** Alur kerja metodologi penelitian deteksi dan klasifikasi postur duduk berisiko skoliosis, membandingkan representasi *Multi-View 2D* dan *Stereo 3D* menggunakan model klasifikasi XGBoost.

---

### A. Kalibrasi Kamera Stereo (*Stereo Camera Calibration*)

Tahap awal penelitian dimulai dengan perancangan dan kalibrasi sistem penangkap visual ganda (*dual-camera setup*). Sistem menggunakan dua kamera web USB (*dual USB webcams*) beresolusi HD ($1280 \times 720$ piksel) yang dipasang pada rig kaku dengan sudut konvergen menghadap area kerja meja kerja (*workstation*): satu kamera diposisikan dari arah depan (*frontal view*, CAM01) dan kamera kedua dari arah samping (*lateral view*, CAM02) seperti diilustrasikan pada Gambar 2.

![Gambar 2. Konfigurasi Penempatan Kamera Ganda](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_2_setup_kamera.png)
**Gambar 2.** Diagram tata letak sistem kamera ganda (*wide-baseline setup*) menghadap meja kerja dan subjek pada kursi tanpa sandaran.

Konfigurasi sudut lebar (*wide-baseline* $\sim 90^\circ$) ini bertujuan agar kedua kamera memiliki bidang pandang tumpang tindih (*overlapping field of view*) yang mencakup tubuh bagian atas subjek tanpa terhalang (*occlusion*). Kalibrasi geometris optik dilakukan menggunakan metode estimasi bidang Zhang dengan bantuan pola papan catur (*checkerboard*) berdimensi $9 \times 6$ titik sudut interior ($54$ titik sudut) dengan panjang sisi kotak $25\text{ mm}$ yang direkam secara simultan pada berbagai orientasi ($\ge 30$ pasangan citra).

Proses kalibrasi memodelkan hubungan matematis antara koordinat ruang 3D dunia $\mathbf{X}_w = [X, Y, Z, 1]^T$ dengan koordinat proyeksi piksel 2D pada citra $\mathbf{x} = [u, v, 1]^T$ berdasarkan model kamera pinhole:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} R & T \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

di mana $s$ adalah faktor skala proyeksi, $K$ adalah matriks parameter intrinsik kamera, sedangkan $R$ dan $T$ adalah parameter ekstrinsik:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

dengan $f_x, f_y$ menyatakan panjang fokus (*focal length*) dalam satuan piksel, dan $c_x, c_y$ menyatakan koordinat titik utama (*principal point*). Efek kelengkungan optik lensa dikoreksi menggunakan vektor koefisien distorsi radial ($k_1, k_2$) dan tangensial ($p_1, p_2$):

$$D = \begin{bmatrix} k_1 & k_2 & p_1 & p_2 \end{bmatrix}$$

Hubungan spasial relatif antara Kamera Depan (CAM01) dan Kamera Samping (CAM02) dimodelkan melalui matriks rotasi stereo $R_{stereo}$ dan vektor translasi stereo $T_{stereo}$ yang memenuhi persamaan:

$$\mathbf{X}_{c2} = R_{stereo} \mathbf{X}_{c1} + T_{stereo}$$

Kriteria keberhasilan kalibrasi divalidasi dengan batas ambang kesalahan proyeksi ulang (*mean reprojection error*) di bawah $1{,}5\text{ piksel}$ guna menjamin akurasi geometri triangulasi stereo.

---

### B. Pengumpulan Data (*Dataset Acquisition*)

Pengumpulan dataset primer (*private dataset*) dirancang khusus untuk merefleksikan postur duduk kerja pada skenario operasional nyata (*in-the-wild*):
1. **Partisipan Penelitian:** Melibatkan 24 subjek sukarelawan (`S001` hingga `S024`) yang mencakup keragaman karakteristik antropometri tubuh (variasi tinggi badan, rentang lebar bahu, dan proporsi panjang torso).
2. **Lingkungan dan Fasilitas Duduk:** Berbeda dengan pengujian laboratorium tertutup yang terkontrol secara artifisial, akuisisi data sengaja dilakukan pada lingkungan semi-terbuka alami (seperti area belajar bersama dan kafetaria) guna menguji ketahanan model terhadap fluktuasi pencahayaan alami serta kerumunan latar belakang dinamis. Fasilitas duduk yang digunakan adalah kursi tanpa sandaran (*backless stool*) agar profil kelengkungan tulang belakang dan pergeseran panggul dapat teramati secara utuh tanpa terhalang (*occlusion*) oleh sandaran kursi dari sudut pandang lateral.
3. **Protokol Perekaman dan Kelas Postur:** Perekaman citra stereo dilakukan secara simultan antarkamera. Partisipan diarahkan untuk memperagakan variasi postur duduk kerja alami yang terbagi ke dalam enam kelas target ergonomis:
   - `upright`: postur duduk tegak netral ergonomis;
   - `leaning_forward`: condong ke arah depan (indikasi fleksi torso berlebih / risiko kifosis);
   - `leaning_backward`: bersandar atau condong ke arah belakang;
   - `leaning_left`: deviasi lateral tulang belakang ke sisi kiri (indikasi kecenderungan skoliosis kiri);
   - `leaning_right`: deviasi lateral tulang belakang ke sisi kanan (indikasi kecenderungan skoliosis kanan);
   - `slouching`: postur merosot dengan panggul meluncur ke depan dan bahu membulat (*kyphotic slump*).
4. **Volume Data Mentah (*Raw Captures*):** Sesi akuisisi menghasilkan total 885 rekaman pasangan citra mentah dari 24 subjek. Dari jumlah tersebut, sebanyak 727 rekaman pasangan citra merepresentasikan enam kelas postur target utama, setelah menyaring 121 rekaman kelas di luar cakupan utama (seperti *forward head posture* terisolasi) serta 37 rekaman data gagal/anomali (*reject*). Seluruh rekaman disimpan berpasangan disertai pencatatan metadata relasional terstruktur (identitas subjek, sesi perekaman, rig kalibrasi, jenis kursi, dan label postur).

---

### C. Estimasi Pose dan Seleksi Subjek Sasaran (*Pose Estimation & Target Selection*)

Pasangan citra dari kedua kamera diproses secara berurutan melalui tiga tingkatan: estimasi titik kerangka tubuh, penyeleksian individu sasaran, dan penapisan kualitas data (*quality control*).

#### 1. Estimasi Pose 2D (YOLOv8n-Pose)
Model berbasis jaringan saraf tiruan konvolusional ringan **YOLOv8n-Pose** (*pretrained* COCO) digunakan untuk mengekstraksi 17 koordinat titik kunci anatomis tubuh $\mathbf{k}_i = (x_i, y_i, c_i)$ beserta nilai tingkat keyakinannya (*confidence score* $c_i \in [0, 1]$) pada citra Kamera Depan (CAM01) dan Kamera Samping (CAM02) secara simultan.

#### 2. Seleksi Subjek Sasaran (*Target Person Selection*)
Untuk mengantisipasi kemunculan individu lain di latar belakang (*bystanders*) pada ruang semi-terbuka, diterapkan algoritma penyeleksi deterministik berbasis fungsi penalti jarak dan area:
- Memprioritaskan subjek dengan luas kotak pembatas (*bounding box*) terbesar pada area kerja utama (*Region of Interest* / ROI);
- Memverifikasi kelengkapan deteksi sendi tubuh bagian atas (kedua bahu dan kedua panggul);
- Menjaga konsistensi identitas antarmuka stereo (*stereo cross-view consistency*).

#### 3. Protokol Kontrol Kualitas Bertingkat (*Two-Layer Quality Control*) dan Ukuran Data Final
Data koordinat kerangka tubuh yang terdeteksi disaring melalui dua lapis aturan kelayakan ketat sebelum masuk ke tahap ekstraksi fitur:
- **Layer 1 (Stereo-QC):** Mensyaratkan nilai ambang batas keypoint confidence $c_i \ge 0{,}5$ untuk seluruh sendi inti serta konsistensi tumpang-tindih kotak pembatas stereo antarkamera.
- **Layer 2 (Feature-QC):** Mewajibkan kelengkapan deteksi sendi panggul kanan (`right_hip`) pada kedua kamera sebagai titik jangkar pembentukan vektor torso. Imputasi nilai nol (*zero imputation*) dihindari karena berpotensi merusak integritas topologi ruang metrik 3D.

Penerapan protokol kontrol kualitas bertingkat ini menyaring 727 tangkapan enam kelas menjadi **403 sampel pasangan citra valid** yang berasal dari **18 subjek partisipan** (`S001`, `S002`, `S003`, `S004`, `S005`, `S006`, `S011`, `S012`, `S013`, `S014`, `S015`, `S016`, `S017`, `S018`, `S019`, `S021`, `S022`, dan `S024`). Sebanyak 6 subjek lainnya tereliminasi secara otomatis karena mengalami oklusi sendi panggul kanan pada sudut pandang lateral akibat keterbatasan pencahayaan atau terhalang paha saat duduk. Sebanyak 403 sampel dari 18 subjek inilah yang membentuk **subset irisan valid (*intersection subset*)** sebagai basis pengujian yang setara (*apple-to-apple comparison*) untuk membandingkan performa representasi fitur 2D dan 3D.

---

### D. Rekayasa Fitur Geometri (*Feature Engineering*)

Koordinat titik piksel mentah tidak dapat digunakan secara langsung karena nilainya sangat bergantung pada jarak subjek ke kamera. Dilakukan rekayasa fitur geometris invarian skala (*scale-invariant*) yang terbagi ke dalam dua pendekatan komparatif:

#### 1. Ekstraksi Fitur *Multi-View* 2D
Fitur diekstraksi dari proyeksi citra Kamera Depan dan Kamera Samping secara terpisah, lalu digabungkan (*concatenated*). Anatomi sudut dan rasio normalisasi diilustrasikan pada Gambar 3.

![Gambar 3. Anatomi Fitur Geometri 2D](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_3_fitur_geometri_2d.png)
**Gambar 3.** Anatomi ekstraksi fitur geometris 2D: (a) Proyeksi frontal CAM01 untuk deteksi kemiringan lateral, dan (b) Proyeksi lateral CAM02 untuk deteksi kemiringan sagital dan pergeseran telinga.

- **Normalisasi Skala:** Seluruh jarak piksel dinormalisasi terhadap panjang vektor torso pada bidang citra masing-masing kamera:
  $$d_{\text{norm}} = \frac{d_{\text{piksel}}}{\|\mathbf{v}_{\text{torso}}\|}$$
  di mana $\mathbf{v}_{\text{torso}} = \mathbf{p}_{\text{mid-shoulder}} - \mathbf{p}_{\text{mid-hip}}$.
- **Sudut Kemiringan Garis Bahu dan Panggul:**
  $$\theta_{\text{slope}} = \arctan\left(\frac{y_{\text{kanan}} - y_{\text{kiri}}}{x_{\text{kanan}} - x_{\text{kiri}}}\right) \times \frac{180^\circ}{\pi}$$
- **Sudut Inklinasi Torso terhadap Garis Vertikal:**
  $$\theta_{\text{torso}} = \arccos\left(\frac{\mathbf{v}_{\text{torso}} \cdot \mathbf{v}_{\text{vertikal}}}{\|\mathbf{v}_{\text{torso}}\| \|\mathbf{v}_{\text{vertikal}}\|}\right) \times \frac{180^\circ}{\pi}$$
  di mana $\mathbf{v}_{\text{vertikal}} = [0, -1]^T$.

Ekstraksi fitur 2D dievaluasi ke dalam dua varian komparatif:
- **Varian 2D Core Baseline (36 Fitur):** Merupakan representasi standar multi-sudut pandang yang menggabungkan 18 fitur dari CAM01 (tampak depan) dan 18 fitur dari CAM02 (tampak samping). Fitur ini mencakup sudut kemiringan garis bahu, sudut kemiringan garis panggul, sudut inklinasi torso terhadap sumbu vertikal citra, sudut ruang kepala-torso, serta rasio jarak relatif horizontal dan vertikal sendi utama yang dinormalisasi terhadap panjang torso.
- **Varian 2D Best Practical (42 Fitur):** Mengembangkan 36 fitur inti dengan menambahkan 6 fitur telinga sagital (*sagittal ear/cervical features*) yang diekstraksi khusus dari Kamera Samping (CAM02), meliputi jarak horizontal telinga ke bahu ($\Delta x / \|\mathbf{v}_{\text{torso}}\|$), jarak vertikal telinga ke bahu ($\Delta y / \|\mathbf{v}_{\text{torso}}\|$), jarak Euclidean ternormalisasi telinga-bahu, serta sudut inklinasi leher/telinga sagital. Penambahan ini merupakan kontribusi eksperimental penting karena sudut inklinasi torso global saja terbukti kurang sensitif dalam membedakan duduk tegak (*upright*) dan duduk merosot (*slouching*); pada postur merosot, translasi anterior kepala (*forward head translation*) dan penurunan posisi telinga relatif terhadap gelang bahu menjadi pembeda biomekanis yang paling diskriminatif.

#### 2. Rekonstruksi dan Ekstraksi Fitur Stereo 3D
Pasangan titik kunci 2D dari Kamera Depan $(\mathbf{x}_1)$ dan Kamera Samping $(\mathbf{x}_2)$ ditriangulasi ke dalam ruang koordinat metrik 3D $\mathbf{X} = [X, Y, Z]^T$ bersatuan milimeter menggunakan metode *Direct Linear Transform* (DLT). Hubungan proyeksi linier dinyatakan sebagai:

$$\begin{bmatrix} 
u_1 P_{1,3}^T - P_{1,1}^T \\
v_1 P_{1,3}^T - P_{1,2}^T \\
u_2 P_{2,3}^T - P_{2,1}^T \\
v_2 P_{2,3}^T - P_{2,2}^T 
\end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \mathbf{0}$$

di mana $P_1 = K_1 [I \mid \mathbf{0}]$ dan $P_2 = K_2 [R_{stereo} \mid T_{stereo}]$ adalah matriks proyeksi kamera stereo. Sistem persamaan homogen $A \mathbf{X} = \mathbf{0}$ diselesaikan menggunakan *Singular Value Decomposition* (SVD) dengan mengambil vektor eigen yang bersesuaian dengan nilai singular terkecil.

Fitur metrik 3D dimodelkan ke dalam dua sistem koordinat ruang:
- **3D Camera Frame (25 Fitur):** Diekstraksi langsung dalam sistem koordinat optik stereo kamera. Representasi ini terdiri dari 25 fitur terstruktur:
  1. *15 Koordinat Spasial Metrik 3D:* Koordinat $(X, Y, Z)$ dari lima sendi kunci tubuh atas (hidung, bahu kiri, bahu kanan, panggul kiri, dan panggul kanan) yang dinormalisasi dengan menempatkan titik tengah kedua panggul ($\mathbf{p}_{\text{mid-hip}}$) sebagai titik acuan nol origin $(0, 0, 0)$ dalam satuan milimeter;
  2. *2 Sudut Roll:* Sudut kemiringan rotasi garis bahu (*shoulder roll*) dan garis panggul (*hip roll*) pada bidang koronal kamera (dalam derajat);
  3. *3 Sudut Kemiringan Torso:* Sudut kemiringan lateral torso (*torso lateral lean*), sudut kemiringan sagital torso (*torso sagittal lean*), dan sudut inklinasi 3D ruang total (*torso 3D inclination*) terhadap sumbu vertikal ruang (dalam derajat);
  4. *1 Sudut Kepala-Torso 3D:* Sudut orientasi ruang 3D antara vektor kepala-leher dengan vektor torso (dalam derajat);
  5. *4 Rasio Asimetri dan Offset Kedalaman Tanpa Dimensi (*Dimensionless Normalized Ratios*):* Rasio offset deviasi kedalaman kepala (*head depth offset*), rasio offset deviasi lateral kepala (*head lateral offset*), rasio asimetri kedalaman antarbahu (*shoulder depth asymmetry*), dan rasio asimetri kedalaman antarpanggul (*hip depth asymmetry*), di mana seluruh selisih jarak kedalaman ($Z$) dinormalisasi terhadap panjang vektor torso 3D metrik.
- **3D Canonical Body Frame (22 Fitur):** Mentransformasikan koordinat metrik 3D ke kerangka acuan anatomis lokal tubuh (*subject-centered canonical frame*) menggunakan prosedur ortogonalisasi Gram-Schmidt:
  $$\mathbf{Y}_{\text{can}} = \frac{\mathbf{p}_{\text{mid-shoulder}} - \mathbf{p}_{\text{mid-hip}}}{\|\mathbf{p}_{\text{mid-shoulder}} - \mathbf{p}_{\text{mid-hip}}\|}, \quad \mathbf{X}_{\text{can}} = \frac{\mathbf{v}_{\text{hip}} - (\mathbf{v}_{\text{hip}} \cdot \mathbf{Y}_{\text{can}})\mathbf{Y}_{\text{can}}}{\|\mathbf{v}_{\text{hip}} - (\mathbf{v}_{\text{hip}} \cdot \mathbf{Y}_{\text{can}})\mathbf{Y}_{\text{can}}\|}, \quad \mathbf{Z}_{\text{can}} = \mathbf{X}_{\text{can}} \times \mathbf{Y}_{\text{can}}$$
  Secara formulasi matematis dan terverifikasi secara empiris pada data eksperimen, penyelarasan koordinat ke tubuh lokal menyebabkan tiga fitur geometris mengalami degenerasi nilai konstan nol dengan varians nol ($\sigma = 0$):
  1. *Sudut kemiringan lateral torso:* identik bernilai $0^\circ$ karena sumbu longitudinal tubuh ditetapkan sebagai sumbu $Y_{\text{can}}$ lokal;
  2. *Sudut kemiringan sagital torso:* identik bernilai $0^\circ$ karena orientasi torso berimpit penuh pada sumbu $Y_{\text{can}}$ tanpa deviasi sudut terhadap bidang sagital lokal;
  3. *Rasio asimetri kedalaman panggul (*hip depth asymmetry*):* identik bernilai $0$ tanpa dimensi karena kedua sendi panggul secara konstruksi membentuk bidang koronal $X_{\text{can}}$ lokal dengan komponen kedalaman $Z_{\text{can}} = 0$.
  Ketiga parameter konstan tersebut dieliminasi karena tidak memberikan informasi diskriminatif (*zero mutual information*), menghasilkan 22 fitur representasi kanonikal tubuh murni.

---

### E. Pelatihan dan Evaluasi Model Terkontrol (*Model Training & Controlled Evaluation*)

Evaluasi komparatif dirancang secara terstandarisasi untuk menjamin perbandingan performa yang objektif (*fair comparison*):

#### 1. Algoritma Klasifikasi (XGBoost)
Model **XGBoost Classifier** (*Extreme Gradient Boosting*) digunakan dengan konfigurasi hiperparameter yang dikunci untuk seluruh varian fitur: fungsi objektif multikelas `multi:softprob`, kedalaman pohon `max_depth = 6`, jumlah estimator `n_estimators = 300`, laju pembelajaran `learning_rate = 0.1`, dan penetapan *random seed* = 42.

#### 2. Protokol Validasi Bebas Kebocoran (*Zero-Leakage 5-Fold Subject-Stratified CV*)
Untuk menguji generalisasi kinerja klasifikasi postur pada individu baru secara kredibel (*unseen subjects*), diterapkan skema **5-Fold Subject-Stratified Cross-Validation**. Pembagian data latih dan uji diisolasi sepenuhnya pada tingkat subjek partisipan sehingga data dari satu subjek tidak pernah muncul bersamaan di set pelatihan dan pengujian, sebagaimana diilustrasikan pada Gambar 4.

![Gambar 4. Partisi 5-Fold Subject-Stratified CV](file:///d:/.Candra/Project/TA/08_documents/figures/gambar_4_skema_5fold_cv.png)
**Gambar 4.** Skema pembagian lipatan *5-Fold Subject-Stratified Cross-Validation* yang menjamin ketiadaan kebocoran data antar-subjek (*zero subject leakage*).

Transformasi standarisasi skala fitur (`StandardScaler`) diestimasi (*fit*) secara ketat hanya pada data pelatihan di setiap fold sebelum diterapkan (*transform*) ke data pengujian:

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

#### 3. Metrik Evaluasi Kinerja Klasifikasi Postur
Keempat varian representasi fitur (2D Core, 2D Best, 3D Camera, dan 3D Canonical) dievaluasi pada subset irisan data valid yang identik (*intersection subset* 403 sampel dari 18 subjek). **Macro F1-score pooled out-of-fold (OOF)** ditetapkan sebagai **metrik evaluasi utama** (*primary performance metric*) guna memberikan bobot evaluasi yang seimbang pada setiap kelas tanpa terdistorsi oleh distribusi jumlah sampel:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{Macro } F1_{\text{pooled}} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

$$\text{Akurasi} = \frac{\sum_{c=1}^C TP_c}{N_{\text{total}}}$$

Stabilitas dan konsistensi kinerja klasifikasi postur antarsubjek diukur melalui simpangan baku antarlipatan (*Fold Standard Deviation*, $\sigma_{\text{fold}}$) yang dihitung dari Macro F1-score masing-masing fold pengujian ($K=5$):

$$\sigma_{\text{fold}} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (F1_k - \overline{F1})^2}$$

---

### F. Pengujian Operasional Inferensi dan Latensi *Real-Time* (*Operational Latency & Deployment Protocol*)

Sebagai bagian integral dari evaluasi sistem pemantauan postur duduk preventif, dirancang protokol pengujian operasional di lingkungan nyata:
1. **Serialisasi Pipeline Model:** Seluruh komponen inferensi yang telah terlatih—mencakup model klasifikasi XGBoost, skaler normalisasi (`scaler.pkl`), skema urutan fitur (`feature_schema.json`), serta pemetaan label kelas (`class_map.json`)—diserialisasi ke dalam format artefak deployment mandiri.
2. **Prosedur Pengukuran Latensi Komputasi:** Pengujian kecepatan eksekusi dilakukan secara terisolasi pada siklus inferensi *end-to-end* (akuisisi frame stereo $\to$ estimasi pose kerangka YOLOv8n $\to$ ekstraksi fitur geometri $\to$ prediksi kelas XGBoost) pada lingkungan CPU lokal standar tanpa akselerasi kartu grafis tambahan.
3. **Parameter Kinerja Operasional:** Diukur melalui rata-rata durasi komputasi per siklus inferensi (dalam milidetik, ms) dan throughput pemrosesan dalam *Frames Per Second* (FPS):
   $$\text{FPS} = \frac{1000}{\text{Latensi}_{\text{end-to-end}}\text{ (ms)}}$$
   Protokol pengukuran ini dirancang di Bab Metode untuk memverifikasi apakah pipeline yang diusulkan memenuhi ambang batas waktu-nyata (*real-time processing* $\ge 5\text{ FPS}$) sebelum angka capaian empiris dilaporkan secara rinci pada Bab Hasil dan Pembahasan.
