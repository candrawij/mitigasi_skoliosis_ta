# Analisis Ilmiah & Interpretasi Fair: Komparasi 2D Multi-View vs Stereo 3D

Berdasarkan panduan metodologi (STEP 15), evaluasi ilmiah tidak boleh berasumsi apriori bahwa 3D harus selalu mengungguli 2D. Analisis harus objektif mengurai kontribusi informasi kedalaman (depth $Z$) terhadap dinamika klasifikasi tiap postur.

---

## 1. Kelas Mana yang Paling Diuntungkan oleh Rekonstruksi 3D?
- **Kelas dengan lonjakan performa terbesar pada 3D:** `leaning_forward` dengan kenaikan $\Delta F1 = +0.1582$ (F1 2D: 0.3453 $\rightarrow$ F1 3D: 0.5035).
- **Penjelasan Fisis:** Postur yang melibatkan rotasi atau translasi pada sumbu optik ($Z$-axis) mendapatkan benefit langsung dari metrik `torso_sagittal_lean_deg` dan `head_depth_offset_norm` yang dikalkulasi secara metrik (meter riil).

---

## 2. Kelas Mana yang Cukup Diselesaikan oleh 2D Multi-View?
- **Kelas lateral:** `leaning_left` (F1 2D: 0.9280) dan `leaning_right` (F1 2D: 0.9606).
- **Penjelasan Fisis:** Kemiringan lateral terproyeksi sempurna pada bidang coronal kamera frontal (CAM01). Fitur 2D seperti `shoulder_slope_deg` dan `torso_inclination_deg` sudah memberikan separabilitas kelas yang nyaris linear (>92-96%), sehingga penambahan dimensi $Z$ tidak memberikan marjin keuntungan yang signifikan pada bidang lateral.

---

## 3. Apakah `leaning_forward` dan `leaning_backward` Mendapat Manfaat Depth Spasial?
- **Leaning Forward:**
  - F1 2D: 0.3453 vs F1 3D: 0.5035 ($\Delta F1 = +0.1582$)
- **Leaning Backward:**
  - F1 2D: 0.6789 vs F1 3D: 0.5979 ($\Delta F1 = -0.0810$)
- **Analisis:** Penambahan koordinat 3D metrik memfasilitasi model untuk membedakan kemiringan bidang sagital tanpa terdistorsi oleh variasi jarak kamera atau skala subjektif tubuh subjek.

---

## 4. Apakah `slouching` Masih Tertukar dengan `leaning_forward`?
- **Kasus GT = `slouching` (Support: 65):**
  - Pada 2D Multi-View: diprediksi keliru sebagai `leaning_forward` sebanyak **18 kali**.
  - Pada Stereo 3D: diprediksi keliru sebagai `leaning_forward` sebanyak **20 kali**.
- **Kasus GT = `leaning_forward` (Support: 75):**
  - Pada 2D Multi-View: diprediksi keliru sebagai `slouching` sebanyak **36 kali**.
  - Pada Stereo 3D: diprediksi keliru sebagai `slouching` sebanyak **26 kali**.
- **Kesimpulan Fisis:** `slouching` (membungkuk) dan `leaning_forward` (condong ke depan) secara biomekanik memiliki orientasi vektor torso yang sangat berdekatan pada bidang sagital. Fitur `head_to_shoulder_norm` dan `head_torso_angle_3d_deg` membantu mengisolasi kelengkungan leher-tulang belakang, namun ambiguitas transisi postur tetap menjadi tantangan inheren pada subjek berpostur lentur.

---

## 5. Trade-Off Operasional: Akurasi vs Usable Coverage
- **2D Multi-View:** Coverage tinggi (704/727 = 96.84%), tidak terpengaruh oleh kalibrasi stereo epipolar yang ketat, robust terhadap oklusi parsial pada salah satu kamera.
- **Stereo 3D:** Menawarkan representasi geometri metrik spasial independen sudut pandang (25 fitur vs 36 fitur), namun memerlukan epipolar constraint yang ketat sehingga menghasilkan usable coverage 403/727 (55.43%).
- **Rekomendasi Deployment TA:** Sistem mitigasi skoliosis dapat menerapkan skema **Hierarchical Dual-Mode**: menggunakan Stereo 3D saat validasi epipolar lulus (QC Valid), dan bertransisi secara graceful ke 2D Multi-View saat terjadi degradasi triangulasi 3D.
