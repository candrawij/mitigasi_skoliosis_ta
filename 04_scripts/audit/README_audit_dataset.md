Audit Dataset Project Design 20242025

1. Siapkan Python

Disarankan Python 3.10 atau lebih baru.

python --version

2. Buat virtual environment

python -m venv .venv
.venv\Scripts\activate

3. Instal dependensi

pip install -r requirements_audit.txt

4. Jalankan audit

Contoh:

python audit_posture_dataset.py "D:\Dataset\Project-Design-20242025"

Menentukan folder output sendiri:

python audit_posture_dataset.py "D:\Dataset\Project-Design-20242025" --output-dir "D:\Dataset\audit_project_design"

5. File hasil utama

audit_summary.md: ringkasan utama.

class_distribution.csv: distribusi kelas pada train/valid/test.

images_audit.csv: metadata seluruh citra.

exact_duplicates.csv: gambar identik.

near_duplicates.csv: gambar yang sangat mirip.

split_leakage.csv: gambar identik/mirip pada split berbeda.

contact_sheets/: contoh visual setiap kelas.

6. Interpretasi awal

Dataset belum langsung dinyatakan layak hanya karena skrip berhasil.

Periksa terutama:

Apakah Unlabeled masih ada?

Apakah kelas sangat tidak seimbang?

Apakah exact/near duplicate muncul di train dan test?

Apakah gambar tiap kelas benar-benar sesuai label?

Apakah tubuh, bahu, dan pinggul terlihat?

Apakah gambar tampak sebagai frame berurutan dari video?

Apakah subjek yang sama muncul pada train dan test?

7. Catatan penting

Skrip tidak menghapus file. Nilai blur, brightness, dan near duplicate hanya indikator awal; keputusan akhir harus melalui pemeriksaan visual.