# Taxonomy Kelas — Unified Label Mapping

## 1. Kelas per Dataset

### Project Design 20242025 (5 kelas)
| Label | Deskripsi |
|---|---|
| `upright` | Duduk tegak |
| `leaning_forward` | Condong ke depan |
| `leaning_backward` | Condong ke belakang |
| `leaning_left` | Condong ke kiri |
| `leaning_right` | Condong ke kanan |

### Postureexercise (5 kelas)
| Label | Deskripsi |
|---|---|
| `thang` | Tegak/lurus |
| `nga_phai` | Condong ke kanan (besar) |
| `nga_trai` | Condong ke kiri (besar) |
| `nghieng_phai` | Miring ke kanan (kecil) |
| `nghieng_trai` | Miring ke kiri (kecil) |

### Sitting Posture Detection (4 kelas)
| Label | Deskripsi |
|---|---|
| `good_posture` | Postur baik/tegak |
| `leaning_forward` | Condong ke depan |
| `leaning_backward` | Condong ke belakang |
| `slouch` | Bungkuk |

### IKORN 4-Keypoint (2 kelas)
| Label | Deskripsi |
|---|---|
| `Good` | Postur baik |
| `Bad` | Postur buruk |

---

## 2. Strategi Mapping

### Untuk training per-dataset (EXP-01, EXP-02, EXP-03)
Setiap dataset ditraining dengan label aslinya. Tidak perlu mapping.

### Untuk cross-dataset evaluation (EXP-05)
Diperlukan mapping ke taxonomy bersama. Dua level mapping:

#### Level A: Binary (Good/Bad)
| Dataset | Good | Bad |
|---|---|---|
| Project Design | upright | leaning_forward, leaning_backward, leaning_left, leaning_right |
| Postureexercise | thang | nga_phai, nga_trai, nghieng_phai, nghieng_trai |
| SPD | good_posture | leaning_forward, leaning_backward, slouch |
| IKORN | Good | Bad |

#### Level B: Multi-class (5-way simplified)
| Unified | Project Design | Postureexercise | SPD | IKORN |
|---|---|---|---|---|
| upright | upright | thang | good_posture | Good |
| forward | leaning_forward | — | leaning_forward | Bad* |
| backward | leaning_backward | — | leaning_backward | — |
| lean_left | leaning_left | nghieng_trai, nga_trai | — | — |
| lean_right | leaning_right | nghieng_phai, nga_phai | — | — |
| slouch | — | — | slouch | Bad* |

*IKORN "Bad" tidak bisa dipetakan ke satu kelas spesifik tanpa inspeksi visual.

---

## 3. Keputusan Praktis

1. **Fase Training (EXP-01–03):** Gunakan label asli per dataset
2. **Fase Cross-dataset (EXP-05):** Gunakan Level A (Binary) sebagai baseline cross-dataset
3. **Level B** hanya feasible untuk subset dataset yang memiliki kelas yang sama
4. Taxonomy final akan di-review setelah hasil eksperimen awal tersedia
