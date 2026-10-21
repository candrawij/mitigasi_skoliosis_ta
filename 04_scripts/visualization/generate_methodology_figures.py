import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

out_dir = r"08_documents\figures"
os.makedirs(out_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------------
# GAMBAR 2: DIAGRAM SETUP KAMERA STEREO WIDE-BASELINE
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.set_xlim(-1, 9)
ax.set_ylim(-1, 8)
ax.axis('off')
fig.patch.set_facecolor('white')

# Title
ax.text(4.0, 7.5, "Gambar 2. Konfigurasi Penempatan Kamera Ganda (Wide-Baseline Setup)",
        ha='center', va='center', fontsize=13, fontweight='bold', color="#1E3A8A")

# Desk (Meja Kerja)
desk = patches.Rectangle((2.0, 3.2), 4.0, 1.8, facecolor='#E2E8F0', edgecolor='#475569', linewidth=2, zorder=2)
ax.add_patch(desk)
ax.text(4.0, 4.1, "Meja Kerja (Workstation)", ha='center', va='center', fontsize=11, fontweight='bold', color="#334155")

# Laptop on desk
laptop = patches.Rectangle((3.3, 3.7), 1.4, 0.9, facecolor='#94A3B8', edgecolor='#334155', linewidth=1.5, zorder=3)
ax.add_patch(laptop)
ax.text(4.0, 4.15, "Laptop", ha='center', va='center', fontsize=8, color="#0F172A", zorder=4)

# Backless Stool (Kursi Tanpa Sandaran)
stool = patches.Circle((4.0, 2.0), 0.65, facecolor='#FEF08A', edgecolor='#CA8A04', linewidth=2, zorder=2)
ax.add_patch(stool)
ax.text(4.0, 2.0, "Kursi Tanpa\nSandaran", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#854D0E", zorder=3)

# Subjek (Duduk di Stool)
subject_head = patches.Circle((4.0, 2.0), 0.35, facecolor='#FCA5A5', edgecolor='#DC2626', linewidth=1.8, zorder=4)
ax.add_patch(subject_head)
ax.text(4.0, 2.0, "Subjek", ha='center', va='center', fontsize=8, fontweight='bold', color="#7F1D1D", zorder=5)

# Camera 1 (Frontal View - Depan)
cam1_x, cam1_y = 4.0, 6.2
cam1 = patches.Rectangle((cam1_x - 0.45, cam1_y - 0.25), 0.9, 0.5, facecolor='#3B82F6', edgecolor='#1D4ED8', linewidth=2, zorder=3)
ax.add_patch(cam1)
ax.text(cam1_x, cam1_y, "CAM 01", ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=4)
ax.text(cam1_x, cam1_y + 0.45, "Kamera Depan (Frontal View)", ha='center', va='center', fontsize=10, fontweight='bold', color="#1E40AF")

# FOV cone Cam 1
fov1 = patches.Polygon([[cam1_x, cam1_y], [2.2, 1.5], [5.8, 1.5]], closed=True,
                       facecolor='#60A5FA', edgecolor='#2563EB', linestyle='--', alpha=0.18, zorder=1)
ax.add_patch(fov1)

# Camera 2 (Lateral View - Samping Kanan)
cam2_x, cam2_y = 7.6, 2.0
cam2 = patches.Rectangle((cam2_x - 0.25, cam2_y - 0.45), 0.5, 0.9, facecolor='#10B981', edgecolor='#047857', linewidth=2, zorder=3)
ax.add_patch(cam2)
ax.text(cam2_x, cam2_y, "CAM\n02", ha='center', va='center', fontsize=8.5, fontweight='bold', color='white', zorder=4)
ax.text(cam2_x, cam2_y + 0.7, "Kamera Samping\n(Lateral View)", ha='center', va='center', fontsize=10, fontweight='bold', color="#065F46")

# FOV cone Cam 2
fov2 = patches.Polygon([[cam2_x, cam2_y], [3.2, 0.3], [3.2, 3.7]], closed=True,
                       facecolor='#34D399', edgecolor='#059669', linestyle='--', alpha=0.18, zorder=1)
ax.add_patch(fov2)

# Angle Arc between Cam 1 and Cam 2
arc = patches.Arc((4.0, 2.0), 3.0, 3.0, angle=0, theta1=0, theta2=90, color='#DC2626', linewidth=2, linestyle=':')
ax.add_patch(arc)
ax.text(5.5, 3.5, "Sudut Konvergen ~90°\n(Wide-Baseline)", ha='center', va='center', fontsize=9, fontweight='bold', color='#B91C1C')

# Overlapping FOV indicator
ax.text(4.0, 0.6, "• Area Tumpang Tindih (Overlapping FOV) Mencakup Tubuh Bagian Atas Subjek\n• Kamera Samping Mengamati Deviasi Sagital Tanpa Terhalang Sandaran Kursi",
        ha='center', va='center', fontsize=9.5, color='#334155',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2))

plt.tight_layout()
p2 = os.path.join(out_dir, "gambar_2_setup_kamera.png")
fig.savefig(p2, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {p2}")


# -------------------------------------------------------------
# GAMBAR 3: ANATOMI FITUR GEOMETRIS 2D (FRONTAL & LATERAL)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5), dpi=300)
fig.patch.set_facecolor('white')

# 3A: Frontal View
ax1.set_xlim(-1, 5)
ax1.set_ylim(-0.5, 7)
ax1.axis('off')
ax1.text(2.0, 6.7, "(a) Sudut Pandang Depan (CAM01)\nDeteksi Deviasi Lateral (Skoliosis)",
         ha='center', va='center', fontsize=11, fontweight='bold', color='#1E40AF')

# Points Frontal (Slight leaning right)
head = (2.2, 5.5)
ls = (1.2, 4.4)
rs = (3.2, 4.1) # right shoulder slightly lower
ms = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
lh = (1.4, 1.8)
rh = (3.0, 1.7)
mh = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)

# Head circle
ax1.add_patch(patches.Circle(head, 0.4, facecolor='#E0F2FE', edgecolor='#0284C7', linewidth=2))
ax1.text(head[0], head[1], "Nose", ha='center', va='center', fontsize=8, fontweight='bold')

# Neck line
ax1.plot([head[0], ms[0]], [head[1], ms[1]], color='#0284C7', linewidth=2)

# Shoulder line & points
ax1.plot([ls[0], rs[0]], [ls[1], rs[1]], color='#16A34A', linewidth=2.5)
ax1.plot(ls[0], ls[1], 'o', color='#15803D', markersize=8)
ax1.plot(rs[0], rs[1], 'o', color='#15803D', markersize=8)
ax1.text(ls[0]-0.2, ls[1]+0.2, "Left Shoulder", ha='right', fontsize=8)
ax1.text(rs[0]+0.2, rs[1]+0.2, "Right Shoulder", ha='left', fontsize=8)

# Shoulder slope angle arc
ax1.plot([ls[0], rs[0]+0.8], [ls[1], ls[1]], '--', color='#94A3B8', linewidth=1.2)
ax1.text(rs[0]+0.3, rs[1]-0.25, "θ shoulder slope", fontsize=8.5, fontweight='bold', color='#16A34A')

# Hip line & points
ax1.plot([lh[0], rh[0]], [lh[1], rh[1]], color='#D97706', linewidth=2.5)
ax1.plot(lh[0], lh[1], 'o', color='#B45309', markersize=8)
ax1.plot(rh[0], rh[1], 'o', color='#B45309', markersize=8)
ax1.text(lh[0]-0.2, lh[1]-0.2, "Left Hip", ha='right', fontsize=8)
ax1.text(rh[0]+0.2, rh[1]-0.2, "Right Hip", ha='left', fontsize=8)

# Torso Vector
ax1.plot([mh[0], ms[0]], [mh[1], ms[1]], color='#DC2626', linewidth=3)
ax1.text(ms[0]+0.15, (ms[1]+mh[1])/2, "Vektor Torso (v_torso)", fontsize=9, fontweight='bold', color='#DC2626')

# Vertical Ref Line
ax1.plot([mh[0], mh[0]], [mh[1], ms[1]+0.5], ':', color='#475569', linewidth=1.5)
ax1.text(mh[0]+0.1, ms[1]+0.3, "Garis Vertikal", fontsize=8, color='#475569')
ax1.text(mh[0]+0.1, mh[1]+1.0, "θ torso\ninclination", fontsize=8.5, fontweight='bold', color='#DC2626')


# 3B: Lateral View
ax2.set_xlim(-1, 5)
ax2.set_ylim(-0.5, 7)
ax2.axis('off')
ax2.text(2.0, 6.7, "(b) Sudut Pandang Samping (CAM02)\nDeteksi Deviasi Sagital (Slouching / Kyphosis)",
         ha='center', va='center', fontsize=11, fontweight='bold', color='#047857')

# Points Lateral (Slouching forward)
lat_ear = (1.5, 5.4)
lat_shoulder = (2.2, 4.2)
lat_hip = (2.0, 1.8)

# Head & Ear
ax2.add_patch(patches.Circle((1.8, 5.5), 0.45, facecolor='#FEF3C7', edgecolor='#D97706', linewidth=2))
ax2.plot(lat_ear[0], lat_ear[1], 'o', color='#B45309', markersize=8)
ax2.text(lat_ear[0]-0.2, lat_ear[1], "Ear (Telinga)", ha='right', fontsize=8.5, fontweight='bold', color='#B45309')

# Shoulder
ax2.plot(lat_shoulder[0], lat_shoulder[1], 'o', color='#15803D', markersize=8)
ax2.text(lat_shoulder[0]+0.2, lat_shoulder[1], "Shoulder (Bahu)", ha='left', fontsize=8.5, fontweight='bold', color='#15803D')

# Hip
ax2.plot(lat_hip[0], lat_hip[1], 'o', color='#4338CA', markersize=8)
ax2.text(lat_hip[0]+0.2, lat_hip[1], "Hip (Panggul)", ha='left', fontsize=8.5, fontweight='bold', color='#4338CA')

# Torso line
ax2.plot([lat_hip[0], lat_shoulder[0]], [lat_hip[1], lat_shoulder[1]], color='#DC2626', linewidth=3)
ax2.text((lat_hip[0]+lat_shoulder[0])/2+0.15, (lat_hip[1]+lat_shoulder[1])/2, "Torso Sagital", fontsize=9, fontweight='bold', color='#DC2626')

# Neck line (Ear to Shoulder)
ax2.plot([lat_ear[0], lat_shoulder[0]], [lat_ear[1], lat_shoulder[1]], color='#8B5CF6', linewidth=2.5)
ax2.text((lat_ear[0]+lat_shoulder[0])/2-0.2, (lat_ear[1]+lat_shoulder[1])/2, "Garis Leher", fontsize=8.5, fontweight='bold', color='#6D28D9')

# Sagittal Ear-Shoulder Offset (Delta X & Delta Y)
ax2.plot([lat_ear[0], lat_shoulder[0]], [lat_ear[1], lat_ear[1]], '--', color='#9333EA', linewidth=1.2)
ax2.text((lat_ear[0]+lat_shoulder[0])/2, lat_ear[1]+0.2, "Δx (Offset Horizontal Telinga)", fontsize=8, color='#6B21A8')

ax2.plot([lat_shoulder[0], lat_shoulder[0]], [lat_ear[1], lat_shoulder[1]], '--', color='#9333EA', linewidth=1.2)
ax2.text(lat_shoulder[0]+0.2, (lat_ear[1]+lat_shoulder[1])/2, "Δy (Tinggi Telinga-Bahu)", fontsize=8, color='#6B21A8')

plt.tight_layout()
p3 = os.path.join(out_dir, "gambar_3_fitur_geometri_2d.png")
fig.savefig(p3, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {p3}")


# -------------------------------------------------------------
# GAMBAR 4: SKEMA 5-FOLD SUBJECT-STRATIFIED CV (ZERO LEAKAGE)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(6.0, 6.5, "Gambar 4. Partisi 5-Fold Subject-Stratified Cross-Validation (Zero Subject Leakage)",
        ha='center', va='center', fontsize=13, fontweight='bold', color='#1E3A8A')
ax.text(6.0, 6.1, "18 Subjek Partisipan Terdistribusi Bersih Antar-Fold: Uji Generalisasi pada Subjek Baru",
        ha='center', va='center', fontsize=10, fontstyle='italic', color='#475569')

folds_data = [
    ("Fold 0", "306 Sampel (15 Subjek)", "97 Sampel", "S001, S013, S018"),
    ("Fold 1", "326 Sampel (15 Subjek)", "77 Sampel", "S002, S012, S021"),
    ("Fold 2", "335 Sampel (13 Subjek)", "68 Sampel", "S006, S015, S016, S017, S019"),
    ("Fold 3", "313 Sampel (14 Subjek)", "90 Sampel", "S003, S004, S022, S024"),
    ("Fold 4", "332 Sampel (15 Subjek)", "71 Sampel", "S005, S011, S014"),
]

y_start = 4.8
row_h = 0.75

for i, (fold_name, train_desc, test_desc, test_subs) in enumerate(folds_data):
    y = y_start - i * row_h
    
    # Fold label
    ax.text(0.8, y + 0.22, fold_name, ha='left', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    
    # Train Bar (Blue)
    rect_train = patches.Rectangle((2.0, y), 6.5, 0.45, facecolor='#3B82F6', edgecolor='#1D4ED8', linewidth=1.5, zorder=2)
    ax.add_patch(rect_train)
    ax.text(5.25, y + 0.22, f"Training Set: {train_desc}", ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=3)
    
    # Test Bar (Orange/Red)
    rect_test = patches.Rectangle((8.6, y), 2.8, 0.45, facecolor='#F97316', edgecolor='#C2410C', linewidth=1.5, zorder=2)
    ax.add_patch(rect_test)
    ax.text(10.0, y + 0.22, f"Test: {test_desc}", ha='center', va='center', fontsize=8.5, fontweight='bold', color='white', zorder=3)
    
    # Subjects annotation
    ax.text(8.6, y - 0.18, f"Subjek Uji: {test_subs}", ha='left', va='center', fontsize=7.5, fontstyle='italic', color='#9A3412')

# Legend & note at bottom
ax.text(6.0, 0.6, "Aturan Zero-Leakage: Subjek pada Test Set sama sekali TIDAK PERNAH muncul di Training Set pada fold yang sama.\nStandardScaler hanya di-fit pada Training Set, lalu ditransformasikan ke Test Set.",
        ha='center', va='center', fontsize=9.5, fontweight='bold', color='#334155',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF2F2', edgecolor='#EF4444', linewidth=1.2))

plt.tight_layout()
p4 = os.path.join(out_dir, "gambar_4_skema_5fold_cv.png")
fig.savefig(p4, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {p4}")
