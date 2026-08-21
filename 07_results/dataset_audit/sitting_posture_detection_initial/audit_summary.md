# Ringkasan Audit Dataset COCO Detection

- Folder dataset: `D:\.Candra\Project\TA\02_data\raw\Sitting Posture Detection.v2i.coco`
- Total citra: **490**
- Citra valid: **490**
- Citra rusak: **0**
- Citra tanpa anotasi: **1**
- Total anotasi: **560**
- Anotasi bbox tidak valid: **0**
- Grup exact duplicate: **0**
- Pasangan near duplicate: **40**
- Potensi leakage lintas split: **17**

## Distribusi Kelas (Total Anotasi)

| Kelas | Jumlah Anotasi |
|---|---:|
| good_posture | 156 |
| leaning_backward | 154 |
| leaning_forward | 150 |
| slouch | 100 |

## Distribusi Kelas per Split

| Split | Kelas | Jumlah Anotasi |
|---|---|---:|
| test | good_posture | 17 |
| test | leaning_backward | 30 |
| test | leaning_forward | 38 |
| test | slouch | 18 |
| train | good_posture | 74 |
| train | leaning_backward | 91 |
| train | leaning_forward | 67 |
| train | slouch | 30 |
| valid | good_posture | 65 |
| valid | leaning_backward | 33 |
| valid | leaning_forward | 45 |
| valid | slouch | 52 |

## Distribusi Split (Jumlah Gambar)

| Split | Jumlah Gambar |
|---|---:|
| test | 92 |
| train | 214 |
| valid | 184 |

## Pemeriksaan Kualitas

- Citra gelap: **2** (threshold brightness < 45.0)
- Citra berpotensi blur: **0** (threshold edge variance < 120.0)
- Citra di bawah resolusi minimum: **0**

## Peringatan

- Terdapat 1 gambar tanpa anotasi.
- Terdapat exact/near duplicate pada split berbeda; ini berpotensi data leakage.

## Catatan

Nilai blur, brightness, dan near duplicate adalah indikator awal. Keputusan akhir harus melalui pemeriksaan visual (lihat contact_sheets/).