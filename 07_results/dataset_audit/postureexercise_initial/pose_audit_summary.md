# Ringkasan Audit Postureexercise

- Dataset: `D:\.Candra\Project\TA\02_data\raw\Sitting_posture.v17i.yolov8`
- YAML: `D:\.Candra\Project\TA\02_data\raw\Sitting_posture.v17i.yolov8\data.yaml`
- kpt_shape: **[7, 3]**
- Kelas: **{0: 'nga_phai', 1: 'nga_trai', 2: 'nghieng_phai', 3: 'nghieng_trai', 4: 'thang'}**
- Total citra: **1694**
- Citra rusak: **0**
- Gambar tanpa label: **0**
- Label kosong: **0**
- Total objek anotasi: **1696**
- Objek anotasi valid: **1696**
- Objek anotasi tidak valid: **0**
- Exact duplicate lintas split: **0**
- Near duplicate lintas split: **0**

## Distribusi Split

| Split | Gambar |
|---|---:|
| test | 316 |
| train | 1158 |
| valid | 220 |

## Peringatan

- Tidak ada peringatan otomatis utama.

## File yang Harus Diperiksa

1. `class_distribution.csv`
2. `keypoint_visibility.csv`
3. `annotation_issues.csv`
4. `cross_split_duplicates.csv`
5. `annotated_contact_sheets/`

## Catatan

Keypoint pada format YOLO-Pose tetap merupakan koordinat 2D pada citra. Dimensi ketiga pada `kpt_shape: [K, 3]` adalah visibility, bukan koordinat kedalaman z.