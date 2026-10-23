"""
generate_final_four_model_plot.py
Generates publication-ready Figure 3.5: Final Four Model Comparison (Accuracy, Macro F1, and Stability)
Saved as 300 DPI image: 08_documents/figures/gambar_5_komparasi_empat_model.png
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "07_results" / "experiments" / "private_final" / "FINAL_FOUR_MODEL" / "final_four_model_comparison.csv"
OUTPUT_PATH = PROJECT_ROOT / "08_documents" / "figures" / "gambar_5_komparasi_empat_model.png"

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found!")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    
    # Clean model labels for plot
    model_labels = [
        "2D Core Baseline\n(36 Fitur)",
        "2D Best Practical\n(42 Fitur - Telinga)",
        "3D Camera Frame\n(25 Fitur)",
        "3D Canonical Body\n(22 Fitur)"
    ]
    
    accuracies = df["pooled_accuracy"].values * 100
    f1_scores = df["pooled_macro_f1"].values * 100
    fold_means = df["fold_mean_f1"].values * 100
    fold_stds = df["fold_std_f1"].values * 100

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})

    x = np.arange(len(model_labels))
    width = 0.35

    # Subplot 1: Pooled Accuracy vs Pooled Macro F1
    colors_acc = ["#2b5c8f", "#1b7837", "#b2182b", "#762a83"]
    rects1 = ax1.bar(x - width/2, accuracies, width, label="Akurasi Pooled (%)", color="#4575b4", edgecolor="black", linewidth=1, alpha=0.9)
    rects2 = ax1.bar(x + width/2, f1_scores, width, label="Macro F1-Score Pooled (%)", color="#2ca02c", edgecolor="black", linewidth=1, alpha=0.9)

    ax1.set_ylabel("Persentase Kinerja (%)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Perbandingan Akurasi dan Macro F1-Score (Pooled OOF)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 85)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1b7837")

    # Add badge for 2D Best
    ax1.text(1, 77, "MODEL TERBAIK\n(F1 = 70.39%)", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", fc="#d9f0d3", ec="#1b7837", lw=1.5),
             fontsize=9, fontweight="bold", color="#1b7837")

    # Subplot 2: Fold-by-Fold Stability (Mean F1 with Error Bars = Fold Std)
    colors_fold = ["#4575b4", "#2ca02c", "#d73027", "#9970ab"]
    bars2 = ax2.bar(x, fold_means, yerr=fold_stds, capsize=6, color=colors_fold, edgecolor="black", linewidth=1, alpha=0.85, width=0.55)
    
    ax2.set_ylabel("Macro F1 Antarlipatan (Mean ± Std, %)", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Stabilitas Generalisasi Lintas Lipatan (5-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 95)

    for i, (m, s) in enumerate(zip(fold_means, fold_stds)):
        ax2.text(i, m + s + 3, f"{m:.1f} ± {s:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Evaluasi Komparatif Kinerja Klasifikasi Postur Duduk (Subset Irisan Valid N = 403, 18 Subjek)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Figure saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
