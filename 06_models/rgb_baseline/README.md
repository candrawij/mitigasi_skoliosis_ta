# Model Baseline Citra RGB (06_models/rgb_baseline)

Direktori ini mencatat model klasifikasi berbasis citra RGB langsung (tanpa ekstraksi keypoint / tanpa pose estimation).

---

## Status & Lokasi Bobot Model

Pada fase awal penelitian (*Exploratory Baseline*), model CNN deep learning (EfficientNet-B0) dilatih langsung pada citra dataset publik untuk membandingkan pendekatan citra penuh (*end-to-end image-based*) melawan pendekatan berbasis geometri keypoint (*pose-based*).

Hasil temuan menunjukkan bahwa model citra penuh sangat rentan terhadap *background overfitting* (menghafal corak dinding, kursi, atau pakaian subjek) dibandingkan model berbasis fitur sudut skeleton tubuh.

Bobot model (*checkpoints*) dan riwayat pelatihan baseline RGB ini disimpan sebagai artefak eksperimen di bawah:
1. **Dataset Project Design:**  
   [`07_results/experiments/EXP-PD-CNN/best_model.pt`](file:///07_results/experiments/EXP-PD-CNN/best_model.pt)  
   Riwayat pelatihan: `training_history.csv`, Hasil: `results.json`
2. **Dataset Sitting Posture Detection:**  
   [`07_results/experiments/EXP-SPD-CNN/best_model.pt`](file:///07_results/experiments/EXP-SPD-CNN/best_model.pt)  
   Riwayat pelatihan: `training_history.csv`, Hasil: `results.json`
