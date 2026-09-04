## posture_correction

Status: Tidak diperlakukan sebagai kandidat dataset independen.

Temuan:
- Gambar yang digunakan sama dengan Project Design 20242025.
- Perbedaannya terutama berada pada skema label/rekomendasi koreksi.
- Berpotensi merupakan turunan atau hasil relabeling dari sumber visual yang sama.

Keputusan:
- Tidak digunakan sebagai dataset utama terpisah.
- Dapat digunakan sebagai referensi perancangan pesan feedback.
- Tidak dihitung sebagai dataset eksternal untuk validasi.

---

## Final Supervisor Decision — 6-Class Private Experiment

- Dataset acquisition stopped at 24 subjects (`S001`–`S024`, total 885 captures / 1.770 images).
- `forward_head` removed from main experiment because the class definition was considered insufficiently verified for the final study. Raw data is preserved safely.
- `reject` retained as negative / invalid-input / QC data and not trained as a seventh posture class.
- XGBoost selected as the sole final classifier for private dataset evaluation and deployment.
- Final scientific comparison is 2D multi-view pose representation vs stereo 3D representation on the exact paired intersection dataset using subject-aware grouped 5-fold cross-validation.