# Ringkasan Audit Dataset COCO Pose

- Folder dataset: `D:\.Candra\Project\TA\02_data\raw\sitting posture.v4-sitting_posture_4keypoint.coco`
- Jumlah keypoint: **4** (bottom, shoulder, head, back)
- Total citra: **655**
- Citra valid: **655**
- Citra rusak: **0**
- Citra tanpa anotasi: **0**
- Total anotasi: **655**
- Anotasi bbox tidak valid: **0**
- Anotasi dengan semua keypoint vis=0: **0**
- Grup exact duplicate: **0**
- Pasangan near duplicate: **3551**
- Potensi leakage lintas split: **520**

## Distribusi Kelas (Total Anotasi)

| Kelas | Jumlah Anotasi |
|---|---:|
| Bad | 420 |
| Good | 235 |

## Distribusi Kelas per Split

| Split | Kelas | Jumlah |
|---|---|---:|
| test | Bad | 17 |
| test | Good | 10 |
| train | Bad | 366 |
| train | Good | 207 |
| valid | Bad | 37 |
| valid | Good | 18 |

## Distribusi Split (Jumlah Gambar)

| Split | Jumlah Gambar |
|---|---:|
| test | 27 |
| train | 573 |
| valid | 55 |

## Visibilitas Keypoint

| # | Nama | vis=0 | vis=1 | vis=2 | Rate Terlihat (%) |
|---|---|---:|---:|---:|---:|
| 0 | bottom | 0 | 0 | 655 | 100.0 |
| 1 | shoulder | 0 | 0 | 655 | 100.0 |
| 2 | head | 0 | 0 | 655 | 100.0 |
| 3 | back | 0 | 0 | 655 | 100.0 |

## Pemeriksaan Kualitas

- Citra gelap: **0** (threshold brightness < 45.0)
- Citra berpotensi blur: **0** (threshold edge variance < 120.0)
- Citra di bawah resolusi minimum: **0**

## Peringatan

- Terdapat exact/near duplicate pada split berbeda; ini berpotensi data leakage.

## File yang Harus Diperiksa

1. `class_distribution.csv`
2. `keypoint_visibility.csv`
3. `annotation_issues.csv` (bbox tidak valid atau kp vis=0 semua)
4. `split_leakage.csv`
5. `contact_sheets/` — termasuk `_keypoint_legend.jpg`

## Catatan

vis=0: keypoint tidak berlabel. vis=1: terhalang/tidak jelas. vis=2: terlihat jelas.
Nilai blur, brightness, dan near duplicate adalah indikator awal.
Keputusan akhir harus melalui pemeriksaan visual.