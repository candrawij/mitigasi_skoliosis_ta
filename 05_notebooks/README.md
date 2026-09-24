# Eksplorasi & Notebooks (05_notebooks)

Direktori ini dicadangkan untuk notebook interaktif (Jupyter Notebook / `.ipynb`).

---

## Standar Rekayasa Perangkat Lunak Riset

Untuk menjamin **reprodusibilitas penuh**, konsistensi eksekusi *cross-validation* tanpa intervensi manual, serta kemudahan eksekusi terdistribusi pada PC GPU (*headless training*), seluruh implementasi eksperimen pada penelitian ini diorganisir sebagai **modul Python mandiri (`.py`)** di bawah direktori:  
👉 [`04_scripts/`](file:///04_scripts/)

Kelebihan pendekatan skrip modular:
1. **Reproducibility:** Seed acak, split subjek, dan pipeline feature engineering terkunci rapi tanpa risiko *out-of-order execution* antarsel notebook.
2. **Batch Logging:** Seluruh log metrik, matriks kebingungan, dan file serialisasi model (`.pkl`, `.json`) otomatis tersimpan ke folder `07_results/` dan `06_models/`.
3. **Command-Line Interface (CLI):** Dapat dieksekusi secara otomatis atau dijalankan via skrip batch (`.bat`) untuk pengujian kamera langsung.
