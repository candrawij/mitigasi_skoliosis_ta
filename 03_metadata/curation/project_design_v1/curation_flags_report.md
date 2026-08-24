# Laporan Curation Flags Semi-Otomatis — Project Design 20242025

- Total gambar: **4813**
- Auto KEEP: **912** (18.9%)
- Auto REVIEW: **2419** (50.3%)
- Auto EXCLUDE: **1482** (30.8%)

## Rincian Flags

| Flag | Jumlah | Keterangan |
|---|---:|---|
| Singleton (unique) | 597 | Gambar tanpa near-duplicate → KEEP |
| P1: Mixed + Cross-split | 2151 | Label campur DAN tersebar antar split → REVIEW |
| P2: Cross-split, representative | 41 | Wakil cluster cross-split → KEEP |
| P2: Cross-split, non-rep | 225 | Non-wakil di cluster cross-split → REVIEW |
| P3: Mixed label | 43 | Label campur dalam satu cluster → REVIEW |
| P4: Same, representative | 274 | Wakil cluster bersih → KEEP |
| P4: Large cluster, non-rep | 652 | Non-wakil di cluster besar (>20) → EXCLUDE |
| P4: Small cluster, non-rep | 830 | Non-wakil di cluster kecil → EXCLUDE |

## Urutan Audit Manual

1. **REVIEW** items terlebih dahulu (mixed/cross-split clusters)
2. Verifikasi **KEEP** untuk representatives: pastikan label benar
3. Konfirmasi **EXCLUDE** pada non-representatives: pastikan tidak ada variasi penting yang hilang

## Catatan Penting

- Semua keputusan auto bersifat **saran**, bukan final
- Gambar **tidak dihapus** oleh script ini
- Kolom `auto_flag`, `auto_decision`, `auto_notes` ditambahkan
- Kolom `decision` asli tidak diubah — isi secara manual setelah review
- File output: `curation_master_flagged.csv`