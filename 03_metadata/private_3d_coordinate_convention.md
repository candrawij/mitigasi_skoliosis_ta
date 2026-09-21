# 3D Coordinate Convention — Private Dataset

## Coordinate Frame

Sistem koordinat 3D yang digunakan adalah **CAM01 (frontal camera) camera coordinate frame**, setelah transformasi dari rectified stereo frame.

```
R1^T @ X_rectified -> X_cam01
```

Pipeline lengkap (dari kode `triangulate_stereo_pair` di `private_inference_common.py`):

```
2D keypoints (pixels, capture resolution)
    |
    v  scale to calibration resolution (640x480)
    |
    v  cv2.undistortPoints (K1/D1, R=R1, P=P1 dan K2/D2, R=R2, P=P2)
    |
    v  cv2.triangulatePoints(P1, P2, u1_rect.T, u2_rect.T)  -> X_rectified
    |
    v  R1.T @ X_rectified  -> X_cam01
```

---

## Definisi Sumbu

| Sumbu | Arah | Unit | Keterangan |
|-------|------|------|-----------|
| **X** | lateral (kiri → kanan dari perspektif kamera) | meter | Positif = ke kanan kamera (= sisi kiri subjek) |
| **Y** | vertikal (atas → bawah dari perspektif kamera) | meter | Negatif = ke atas (kepala lebih negatif daripada pinggul) |
| **Z** | depth (jauh dari kamera) | meter | Positif = jauh dari kamera; range ~0.7–1.1m pada dataset ini |

---

## Origin

- **Origin**: optical center CAM01 (kamera frontal)
- **Unit**: meter

---

## Sanity Check dari Data Nyata

Nilai diambil dari capture FULL 3D yang valid.

### Vertical (Y axis)

- Shoulder center Y lebih negatif dari hip center Y — artinya kepala/bahu lebih tinggi (lebih negatif) dari pinggul. Konsisten dengan Y-down convention camera space:

```
upright [CAP000001]:  sh_c_Y = -0.785  hip_c_Y = -0.409  -> sh lebih tinggi (negatif lebih besar)
```

### Depth (Z axis)

- `leaning_forward` memiliki Z lebih kecil daripada `leaning_backward`, artinya condong ke depan = mendekati kamera = Z lebih kecil:

```
leaning_forward  [CAP000006]: sh_c_Z = 0.848m
leaning_backward [CAP000021]: sh_c_Z = 0.929m
```

Selisih ~8cm sesuai dengan harapan anatomis.

### Lateral (X axis)

- Left shoulder memiliki X lebih besar dari right shoulder (dari perspektif kamera, subjek menghadap kamera), karena CAM01 adalah kamera frontal dan COCO kiri/kanan relatif terhadap tubuh subjek:

```
leaning_right [CAP000031]:
  L_sh X = 0.205m,  R_sh X = 0.044m
  Selisih ~16cm (plausible shoulder width)
```

### Depth Z positif check

- Kode memeriksa `X_orig_cam1[j, 2] > 0.2` sebelum menerima joint.
- Semua joint valid memiliki Z > 0.2m (cam in front of subject).

---

## Catatan Penting

### Rig-degenerate (CAL_006, CAL_010)

- **CAL_006** dan **CAL_010** seluruh capture-nya masuk `EXCLUDE_3D_DEGENERATE_RIG`.
- Kedua rig ini menghasilkan triangulasi yang geometrically invalid.
- **Tidak digunakan** dalam feature extraction maupun model training.
- Subjek yang terdampak: **S007** (CAL_006) dan **S023** (CAL_010).

### NaN convention

- Joint yang tidak memenuhi syarat (confidence < 0.25 pada salah satu kamera, atau Z <= 0.2m) disimpan sebagai `NaN`.
- Jangan replace NaN dengan 0. Nilai `(0,0,0)` di frame kamera berarti tepat di optical center.

### Scale

- `pose_scale` untuk normalization akan dihitung sebagai jarak Euclidean antara `shoulder_center` dan `hip_center` (torso length).
- Range torso yang valid dari data: **0.39–0.44m** untuk dataset ini.

---

## Referensi Kode

- `04_scripts/inference/private_inference_common.py`, fungsi `triangulate_stereo_pair` (baris ~247–333)
- `04_scripts/preprocessing/private_feature_common.py`, fungsi `extract_3d_features`
- Kalibrasi: `02_data/calibration/stereo/`

---

*Dibuat: 2026-09-22*
*Dibuat berdasarkan inspeksi kode dan data nyata. Bukan asumsi.*
