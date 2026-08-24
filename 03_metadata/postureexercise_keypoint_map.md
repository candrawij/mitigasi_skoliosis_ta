# Postureexercise — Keypoint & Class Mapping

## 1. Class Mapping (Confirmed)

| Class ID | Label Asli | Arti | Kategori |
|---|---|---|---|
| 0 | `nga_phai` | Condong/jatuh ke kanan | Lean right |
| 1 | `nga_trai` | Condong/jatuh ke kiri | Lean left |
| 2 | `nghieng_phai` | Miring ke kanan | Tilt right |
| 3 | `nghieng_trai` | Miring ke kiri | Tilt left |
| 4 | `thang` | Tegak/lurus | Upright |

**Catatan bahasa:** Label menggunakan bahasa Vietnam.
- "ngã" (nga) = jatuh/condong (deviasi lebih besar)
- "nghiêng" (nghieng) = miring (deviasi lebih kecil)
- "phải" (phai) = kanan, "trái" (trai) = kiri
- "thẳng" (thang) = lurus/tegak

---

## 2. Keypoint Spatial Analysis

Dataset: `Sitting_posture.v17i.yolov8` — `kpt_shape: [7, 3]`

Semua 1696 anotasi memiliki 7 keypoint dengan visibility = 2 (terlihat jelas, 100%).

### Posisi rata-rata tiap keypoint — kelas `thang` (upright, n=566)

| KP Index | Mean X | Mean Y | Std X | Std Y |
|---|---|---|---|---|
| 0 | 0.3670 | 0.7596 | 0.0357 | 0.1224 |
| 1 | 0.7139 | 0.7525 | 0.0364 | 0.1283 |
| 2 | 0.4479 | 0.4588 | 0.0365 | 0.0952 |
| 3 | 0.4939 | 0.4057 | 0.0423 | 0.0888 |
| 4 | 0.5340 | 0.4746 | 0.0435 | 0.0851 |
| 5 | 0.5738 | 0.4046 | 0.0437 | 0.0879 |
| 6 | 0.6163 | 0.4498 | 0.0442 | 0.0927 |

### Posisi diurutkan dari atas ke bawah (semua kelas, n=1696)

| Ranking | KP Index | Mean Y | Mean X | Posisi |
|---|---|---|---|---|
| 1 (atas) | **5** | 0.4332 | 0.5841 | middle-right |
| 2 | **3** | 0.4408 | 0.5052 | middle-center |
| 3 | **2** | 0.4659 | 0.4550 | middle-center |
| 4 | **4** | 0.4851 | 0.5397 | middle-center |
| 5 | **6** | 0.4950 | 0.5858 | middle-right |
| 6 | **1** | 0.7538 | 0.7104 | lower-right |
| 7 (bawah) | **0** | 0.7633 | 0.3715 | lower-left |

### Observasi kunci

1. **KP 0 dan KP 1** jauh di bawah (Y ≈ 0.75), berseberangan kiri-kanan (X: 0.37 vs 0.71). Jarak horizontal ≈ 0.35. → **Bahu (shoulders)**

2. **KP 3 dan KP 5** paling tinggi (Y ≈ 0.40–0.43), simetris kiri-kanan dari center. → **Telinga (ears)**

3. **KP 2 dan KP 6** di tengah vertikal (Y ≈ 0.45–0.50), simetris. → **Mata (eyes)**

4. **KP 4** di tengah, antara KP 2–6 secara horizontal, antara ear dan eye secara vertikal. → **Hidung (nose)**

### Proposed Mapping (COCO Upper-Body Convention)

Berdasarkan pola spasial yang konsisten dengan 7 keypoint teratas format COCO:

| KP Index | Proposed Body Part | COCO Equivalent | Bukti Spasial |
|---|---|---|---|
| **0** | `left_shoulder` | COCO KP 5 | Paling bawah, paling kiri (X≈0.37, Y≈0.76) |
| **1** | `right_shoulder` | COCO KP 6 | Paling bawah, paling kanan (X≈0.71, Y≈0.75) |
| **2** | `left_eye` | COCO KP 1 | Tengah-kiri (X≈0.45, Y≈0.46) |
| **3** | `left_ear` | COCO KP 3 | Atas-kiri (X≈0.49, Y≈0.41) |
| **4** | `nose` | COCO KP 0 | Tengah-center (X≈0.53, Y≈0.47) |
| **5** | `right_ear` | COCO KP 4 | Atas-kanan (X≈0.57, Y≈0.40) |
| **6** | `right_eye` | COCO KP 2 | Tengah-kanan (X≈0.62, Y≈0.45) |

> **✅ STATUS: CONFIRMED**
>
> Mapping diverifikasi secara visual menggunakan overlay images pada 15 sampel (3 per kelas, 5 kelas).
> Overlay tersimpan di `07_results/dataset_audit/postureexercise_semantics/keypoint_overlay_verification/`.
> Semua titik keypoint sesuai dengan proposed mapping.

### Catatan penting tentang `flip_idx`

Dari `data.yaml`:
```yaml
flip_idx: [0, 1, 2, 3, 4, 5, 6]
```

`flip_idx` ini adalah **identity mapping** (tidak ada pertukaran), yang artinya:
- Dataset ini **tidak menggunakan horizontal flip augmentation yang benar** pada level keypoint
- Jika kita melakukan horizontal flip, pasangan keypoint kiri-kanan harus ditukar
- Setelah mapping dikonfirmasi, `flip_idx` yang benar seharusnya: `[1, 0, 6, 5, 4, 3, 2]` (tukar shoulder L↔R, eye L↔R, ear L↔R)

---

## 3. Langkah Verifikasi yang Diperlukan

- [x] Overlay keypoint dengan nomor indeks pada 15 gambar sampel (3 per kelas) — ✅ Selesai
- [x] Konfirmasi KP 2/6 = mata, KP 3/5 = telinga — ✅ Terkonfirmasi
- [x] `flip_idx` yang benar: `[1, 0, 6, 5, 4, 3, 2]` — ✅ Ditentukan
- [x] Update dokumen ini dengan status CONFIRMED — ✅

---

## 4. File Terkait

- Analisis spasial: `07_results/dataset_audit/postureexercise_semantics/keypoint_spatial_analysis.csv`
- Contact sheets per kelas: `07_results/dataset_audit/postureexercise_semantics/indexed_keypoint_sheets/`
- Verifikasi semantik: `07_results/dataset_audit/postureexercise_semantics/semantic_verification_summary.md`
- Data YAML: `02_data/raw/Sitting_posture.v17i.yolov8/data.yaml`
