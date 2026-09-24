# Format & Manajemen Label Postur Dataset Privat

Direktori ini (`02_data/private_annotations/posture_labels/`) dicadangkan secara arsitektural untuk dokumentasi label kelas postur subjek penelitian.

---

## Manajemen Terpusat (Centralized Labels)

Demi menjamin integritas data (*single source of truth*) dan mencegah ketidakkonsistenan antarsubjek, anotasi label kelas postur pada proyek ini **tidak disimpan sebagai file teks individual**, melainkan dikelola secara terpusat melalui file metadata relasional berikut:

1. **Definisi Label Mentah & Anotasi Sesi:**  
   [`03_metadata/private_templates/captures.csv`](file:///03_metadata/private_templates/captures.csv)  
   Setiap baris mencatat `capture_id`, `session_id`, `subject_id`, dan kolom target `posture_class` (misal: `ideal`, `kyphosis`, `slouching`, `scoliosis_left`, `scoliosis_right`, `lordosis`, `forward_head`).

2. **Taksonomi & Kodifikasi Kelas:**  
   [`03_metadata/taxonomy_kelas.md`](file:///03_metadata/taxonomy_kelas.md)  
   Menjelaskan definisi klinis/ergonomis dari 6 kelas postur utama yang diklasifikasikan pada penelitian ini.

3. **Partisi Cross-Validation (5-Fold Stratified by Subject):**  
   [`03_metadata/private_final_split/private_stratified_group_5fold.csv`](file:///03_metadata/private_final_split/private_stratified_group_5fold.csv) dan  
   [`03_metadata/private_final_split/private_2d3d_intersection_5fold.csv`](file:///03_metadata/private_final_split/private_2d3d_intersection_5fold.csv)

4. **Pemetaan Numerik Model Deployment:**  
   [`06_models/keypoint_2d/private_final/class_map.json`](file:///06_models/keypoint_2d/private_final/class_map.json)
