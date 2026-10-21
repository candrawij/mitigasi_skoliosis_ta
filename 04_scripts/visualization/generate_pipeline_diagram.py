import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set font family and clean styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(16, 10.5), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10.5)
ax.axis('off')

# Color palette (Modern academic aesthetic)
BORDER_BLUE = "#1E3A8A"
BOX_BLUE = "#F0F7FF"
BORDER_GREEN = "#166534"
BOX_GREEN = "#F0FDF4"
BORDER_ORANGE = "#C2410C"
BOX_ORANGE = "#FFF7ED"
BORDER_PURPLE = "#6B21A8"
BOX_PURPLE = "#FAF5FF"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#334155"
ARROW_COLOR = "#475569"

fig.patch.set_facecolor('white')

def draw_box(ax, x, y, w, h, title, items, box_color, border_color, title_color=None):
    if title_color is None:
        title_color = border_color
    rect = patches.FancyBboxPatch((x, y), w, h,
                                  boxstyle="round,pad=0.15,rounding_size=0.18",
                                  facecolor=box_color, edgecolor=border_color,
                                  linewidth=1.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
            fontsize=11.2, fontweight='bold', color=title_color, zorder=3)
    ax.plot([x + 0.2, x + w - 0.2], [y + h - 0.62, y + h - 0.62],
            color=border_color, linewidth=0.8, alpha=0.5, zorder=3)
    
    y_text = y + h - 0.88
    for item in items:
        ax.text(x + 0.25, y_text, item, ha='left', va='center',
                fontsize=9.0, color=TEXT_MUTED, zorder=3)
        y_text -= 0.31

def draw_arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0, 0.15)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=ARROW_COLOR, edgecolor=ARROW_COLOR,
                                width=1.5, headwidth=6, headlength=6, shrink=0.05),
                zorder=1)
    if label:
        ax.text((x1 + x2)/2 + label_offset[0], (y1 + y2)/2 + label_offset[1],
                label, ha='center', va='center', fontsize=8.2, fontweight='bold',
                color=TEXT_DARK, zorder=4,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))

# Main Title & Subtitle
ax.text(8.0, 10.1, "Alur Kerja Metodologi Penelitian (Research Pipeline)",
        ha='center', va='center', fontsize=16, fontweight='heavy', color=BORDER_BLUE)
ax.text(8.0, 9.7, "Deteksi & Klasifikasi Postur Duduk Berisiko Skoliosis: Multi-View 2D Keypoint vs Stereo 3D Reconstruction",
        ha='center', va='center', fontsize=10.8, fontstyle='italic', color=TEXT_MUTED)

# Level 1: Persiapan & Input (Top Row)
# Box 1: Kalibrasi Kamera Stereo
draw_box(ax, 0.8, 7.0, 4.0, 2.3,
         "1. Kalibrasi Kamera Stereo",
         ["• Dual USB Webcam (Dual USB Webcams (HD 720p))",
          "• Pola Papan Catur (Checkerboard 9x6)",
          "• Kalibrasi Intrinsik Kamera (K1, K2, D1, D2)",
          "• Kalibrasi Ekstrinsik Stereo (Matriks R, T)",
          "• Validasi Reprojection Error Kalibrasi < 1.5 px"],
         BOX_BLUE, BORDER_BLUE)

# Box 2: Akuisisi Dataset Privat
draw_box(ax, 6.0, 7.0, 4.0, 2.3,
         "2. Akuisisi Dataset Privat",
         ["• 24 Subjek Partisipan Riil (S001 - S024)",
          "• 885 Pasangan Stereo / 1.770 Citra Mentah",
          "• 6 Kelas Label Utama Postur Duduk Ergonomis",
          "• Anotasi Relasional (participants, captures.csv)",
          "• Eksklusi Kelas Tambahan (forward_head)"],
         BOX_BLUE, BORDER_BLUE)

# Box 3: Estimasi Pose & Seleksi Target
draw_box(ax, 11.2, 7.0, 4.0, 2.3,
         "3. Pose & Seleksi Target",
         ["• Model Deteksi Pose: YOLOv8n-Pose (COCO-17)",
          "• Deteksi Keypoint 2D Sinkron CAM01 & CAM02",
          "• Target Person Selector (Filter Latar Belakang)",
          "• Layer 1 QC: Stereo-QC Bounding Box & Keypoints",
          "• Layer 2 QC: Feature-QC (Validitas Right-Hip)"],
         BOX_BLUE, BORDER_BLUE)

# Horizontal Arrows Level 1
draw_arrow(ax, 4.8, 8.15, 6.0, 8.15, label="Parameter (K, R, T)")
draw_arrow(ax, 10.0, 8.15, 11.2, 8.15, label="Pasangan Citra")

# Vertical Connector from Level 1 to Level 2 (Branching)
ax.plot([13.2, 13.2], [7.0, 5.85], color=ARROW_COLOR, linewidth=1.5, zorder=1)
ax.plot([4.5, 13.2], [5.85, 5.85], color=ARROW_COLOR, linewidth=1.5, zorder=1)
ax.text(8.85, 6.1, "Filter Irisan Valid (Intersection Subset: N=403 Model-Ready Captures)",
        ha='center', va='center', fontsize=8.8, fontweight='bold', color=BORDER_PURPLE,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FAF5FF', edgecolor=BORDER_PURPLE, linewidth=1))

# Level 2: Percabangan Rekayasa Fitur (Comparative Feature Engineering)
draw_arrow(ax, 4.5, 5.85, 4.5, 5.25)
draw_arrow(ax, 11.5, 5.85, 11.5, 5.25)

# Box 4A: Fitur Multi-View 2D
draw_box(ax, 1.6, 3.2, 5.8, 2.05,
         "4A. Rekayasa Fitur Multi-View 2D",
         ["• Normalisasi Koordinat Piksel terhadap Ukuran Torso",
          "• 2D Core (36 Fitur): 18 Fitur CAM01 + 18 Fitur CAM02",
          "  (Sudut Kemiringan Bahu, Pinggul, & Torso dari Vertikal)",
          "• 2D Best (42 Fitur): 36 Core + 6 Fitur Sagital Telinga-Bahu",
          "  (Offset Horizontal/Vertikal Telinga ke Bahu & Sudut Leher)"],
         BOX_GREEN, BORDER_GREEN)

# Box 4B: Fitur Stereo 3D
draw_box(ax, 8.6, 3.2, 5.8, 2.05,
         "4B. Rekayasa Fitur Stereo 3D",
         ["• Triangulasi Stereo DLT: Pasangan Keypoint 2D -> (X, Y, Z)",
          "• 3D Camera Frame (25 Fitur): 15 Koordinat Euklides 3D,",
          "  Roll Bahu/Pinggul, Torso Lean, Inklinasi, Asimetri Kedalaman",
          "• 3D Canonical Body Frame (22 Fitur): Rotasi ke Sumbu Tubuh,",
          "  Mengeliminasi 3 Fitur Degenerate (Torso Lean & Asimetri Hip)"],
         BOX_ORANGE, BORDER_ORANGE)

# Level 3: Pelatihan Model & Evaluasi Terkontrol (Bottom Center)
ax.plot([4.5, 4.5], [3.2, 2.5], color=ARROW_COLOR, linewidth=1.5, zorder=1)
ax.plot([11.5, 11.5], [3.2, 2.5], color=ARROW_COLOR, linewidth=1.5, zorder=1)
ax.plot([4.5, 11.5], [2.5, 2.5], color=ARROW_COLOR, linewidth=1.5, zorder=1)
draw_arrow(ax, 8.0, 2.5, 8.0, 2.1)

# Box 5: Pelatihan & Evaluasi Terkontrol
draw_box(ax, 2.0, 0.35, 12.0, 1.75,
         "5. Pelatihan Klasifikasi & Evaluasi Terkontrol (Fair Comparison: 4 Model Utama)",
         ["• Model Klasifikasi: XGBoost Classifier (multi:softprob, max_depth=6, n_estimators=300, seed=42)",
          "• Protokol Evaluasi: 5-Fold Subject-Stratified Cross-Validation (Isolasi Subjek Tanpa Kebocoran Data)",
          "• Evaluasi Komparatif 4 Model: 2D Core (36f), 2D Best (42f), 3D Camera (25f), 3D Canonical (22f)",
          "• Target 6 Kelas Postur: upright, leaning_forward, leaning_backward, leaning_left, leaning_right, slouching",
          "• Metrik: Pooled OOF Macro F1, Akurasi, Standar Deviasi Antar-Fold, dan Benchmark Latensi Real-Time"],
         BOX_PURPLE, BORDER_PURPLE)

plt.tight_layout()
out_dir = r"08_documents\figures"
os.makedirs(out_dir, exist_ok=True)
out_path_png = os.path.join(out_dir, "gambar_1_pipeline_penelitian.png")
fig.savefig(out_path_png, dpi=300, bbox_inches='tight')
print(f"Updated diagram successfully saved at: {out_path_png}")
