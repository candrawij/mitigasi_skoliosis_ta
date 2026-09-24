# 📝 Direktori 05: Naskah Publikasi & Bahan Artikel Ilmiah

Direktori ini memuat seluruh materi, rancangan struktur (outline), dan bahan penulisan untuk publikasi ilmiah (jurnal terakreditasi / konferensi internasional) serta Bab Pembahasan Skripsi.

---

## 📑 Daftar Dokumen Publikasi

| No | Dokumen | Target | Deskripsi Singkat |
|:---:|---|:---:|---|
| **1** | [OUTLINE_ARTIKEL_ILMIAH_POSTUR.md](file:///d:/.Candra/Project/TA/08_documents/05_naskah_dan_publikasi/OUTLINE_ARTIKEL_ILMIAH_POSTUR.md) | ⭐ **Publikasi Utama** | Outline komprehensif berstandar internasional (format IMRaD). Berisi: judul, latar belakang ergonomis/medis, tinjauan pustaka (CVA, Plumb Line, XGBoost), metodologi ekstraksi 42-fitur 2D & triangulasi stereo 3D, hasil eksperimen tri-model fair comparison, kegagalan canonicalization, profil latency 191.5 ms (5.2 FPS), serta implikasi klinis. |

---

## 🎯 Poin-Poin Narasi Kunci untuk Publikasi

1. **Dual-View Advantage:** Mengapa dua kamera (frontal CAM01 + lateral CAM02) melenyapkan ambiguitas sagital yang dialami kamera tunggal.
2. **Sagittal Ear Features:** Bukti empiris bahwa penambahan 3 fitur telinga lateral menekan konfusi *slouching vs leaning forward* dari 26.1% menjadi 3.6%.
3. **The 2D vs 3D Reality:** Temuan kritis bahwa 3D pose rentan terhadap variasi kalibrasi rig ekstrinsik antar-subjek (Fold 3 collapse), sementara 2D Multi-View mempertahankan kestabilan performa yang tinggi (F1 0.7039 vs 0.6182).
4. **Canonicalization Insight:** Pembuktian eksperimental bahwa pergeseran ke *body-aligned canonical frame* (33.0% akurasi, F1 0.3259) menghilangkan sinyal diskriminatif sudut pandang kamera, membuktikan batasan intrinsik rekonstruksi stereo pasif pada domain ini.
