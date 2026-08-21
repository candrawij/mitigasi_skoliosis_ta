# Ringkasan Audit Dataset

- Folder dataset: `D:\.Candra\Project\TA\Sitting Posture Classification.v5i.folder`
- Total file citra terdeteksi: **4813**
- Citra valid: **4813**
- Citra rusak: **0**
- Jumlah kelas: **5**
- Citra Unlabeled/Unknown: **0**
- Grup exact duplicate: **0**
- Pasangan near duplicate: **65237**
- Potensi leakage lintas split: **10755**

## Distribusi Kelas

| Kelas | Jumlah |
|---|---:|
| leaning_backward | 717 |
| leaning_forward | 1212 |
| leaning_left | 887 |
| leaning_right | 761 |
| upright | 1236 |

## Distribusi Split

| Split | Jumlah |
|---|---:|
| test | 484 |
| train | 3364 |
| valid | 965 |

## Pemeriksaan Kualitas

- Citra gelap: **0** (threshold brightness < 45.0)
- Citra berpotensi blur: **2** (threshold edge variance < 120.0)
- Citra di bawah resolusi minimum: **0**

## Peringatan

- Terdapat exact/near duplicate pada split berbeda; ini berpotensi data leakage.

## Catatan

Hasil blur, brightness, dan near duplicate adalah indikator awal. Keputusan penghapusan gambar tetap harus melalui pemeriksaan visual.