"""
Generate visualizations and publication-ready figures for all experimental results:
1. Model Comparison Bar Chart (Accuracy & Macro F1)
2. Confusion Matrices (EXP-01, EXP-02, EXP-03)
3. Training Convergence Curves (EfficientNet-B0 on PD & SPD)
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PROJECT_ROOT / "07_results" / "experiments"
PLOTS_DIR = PROJECT_ROOT / "07_results" / "visualizations"


def setup_style():
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def plot_model_comparison():
    """Create a unified bar chart comparing all models across datasets."""
    csv_file = EXP_DIR / "master_summary.csv"
    if not csv_file.exists():
        print("master_summary.csv not found!")
        return

    df = pd.read_csv(csv_file)
    
    # Clean up labels for display
    dataset_clean = {
        "project_design": "Project Design (5-class)",
        "sitting_posture_detection": "SPD (4-class)",
        "Postureexercise (7-KP, 5-class)": "Postureexercise (5-class)",
        "IKORN (4-KP, 2-class)": "IKORN (2-class)"
    }
    df["Dataset_Clean"] = df["Dataset"].map(lambda x: dataset_clean.get(str(x), str(x)))
    df["Pipeline_Model"] = df["Experiment"].str.split(" ").str[0] + " - " + df["Model"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Palette
    colors = sns.color_palette("deep", len(df["Pipeline_Model"].unique()))
    model_color_map = dict(zip(df["Pipeline_Model"].unique(), colors))

    # Bar chart 1: Accuracy
    sns.barplot(
        data=df,
        x="Dataset_Clean",
        y="Accuracy",
        hue="Pipeline_Model",
        palette=model_color_map,
        ax=ax1
    )
    ax1.set_title("Test Accuracy Comparison across Datasets & Models")
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Accuracy (0 - 1.0)")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="x", rotation=15)
    ax1.legend(title="Method", loc="lower right", fontsize=8)
    for p in ax1.patches:
        height = p.get_height()
        if height > 0.05:
            ax1.annotate(f"{height:.2f}",
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 3),
                         textcoords='offset points')

    # Bar chart 2: Macro F1
    sns.barplot(
        data=df,
        x="Dataset_Clean",
        y="F1_Macro",
        hue="Pipeline_Model",
        palette=model_color_map,
        ax=ax2
    )
    ax2.set_title("Macro F1-Score Comparison across Datasets & Models")
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Macro F1-Score")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(title="Method", loc="lower right", fontsize=8)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0.05:
            ax2.annotate(f"{height:.2f}",
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 3),
                         textcoords='offset points')

    plt.tight_layout()
    out_path = PLOTS_DIR / "model_comparison_benchmark.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved comparison benchmark plot to: {out_path}")


def plot_confusion_matrices():
    """Plot confusion matrices for key experiments."""
    # 1. EXP-01 Project Design & SPD
    exp01_file = EXP_DIR / "exp01_all_results.json"
    if exp01_file.exists():
        with open(exp01_file) as f:
            exp01_data = json.load(f)

        for item in exp01_data:
            dname = item["dataset"]
            cm = np.array(item["confusion_matrix"])
            classes = item["class_names"]

            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes, cbar=False)
            plt.title(f"Confusion Matrix: EfficientNet-B0\nDataset: {dname} (Test Acc: {item['accuracy']*100:.2f}%)")
            plt.ylabel("Ground Truth")
            plt.xlabel("Predicted Class")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            out_path = PLOTS_DIR / f"cm_efficientnet_{dname}.png"
            plt.savefig(out_path)
            plt.close()
            print(f"[OK] Saved confusion matrix to: {out_path}")

    # 2. EXP-02 Postureexercise & IKORN
    exp02_file = EXP_DIR / "exp02_all_results.json"
    if exp02_file.exists():
        with open(exp02_file) as f:
            exp02_data = json.load(f)

        for item in exp02_data:
            dname = item["dataset"].split(" ")[0].lower()
            mname = item["model"]
            cm = np.array(item["confusion_matrix"])
            classes = item["class_names"]

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                        xticklabels=classes, yticklabels=classes, cbar=False)
            plt.title(f"Confusion Matrix: {mname}\nDataset: {item['dataset']} (Test Acc: {item['accuracy']*100:.2f}%)")
            plt.ylabel("Ground Truth")
            plt.xlabel("Predicted Class")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            out_path = PLOTS_DIR / f"cm_keypoint_{dname}_{mname}.png"
            plt.savefig(out_path)
            plt.close()
            print(f"[OK] Saved confusion matrix to: {out_path}")


def plot_cnn_training_history():
    """Plot loss & accuracy training curves for EfficientNet-B0."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    datasets = [
        ("EXP-PD-CNN", "Project Design (5-Class)", axes[0]),
        ("EXP-SPD-CNN", "Sitting Posture Detection (4-Class)", axes[1]),
    ]

    for exp_id, title_name, (ax_loss, ax_acc) in datasets:
        hist_file = EXP_DIR / exp_id / "training_history.csv"
        if not hist_file.exists():
            continue

        df_hist = pd.read_csv(hist_file)

        # Loss curves
        ax_loss.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", color="royalblue", lw=2)
        ax_loss.plot(df_hist["epoch"], df_hist["valid_loss"], label="Validation Loss", color="crimson", lw=2, linestyle="--")
        ax_loss.set_title(f"{title_name} - Loss Convergence")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Cross Entropy Loss")
        ax_loss.legend()
        ax_loss.grid(True, linestyle=":", alpha=0.6)

        # Acc curves
        ax_acc.plot(df_hist["epoch"], df_hist["train_acc"] * 100, label="Train Acc", color="royalblue", lw=2)
        ax_acc.plot(df_hist["epoch"], df_hist["valid_acc"] * 100, label="Validation Acc", color="crimson", lw=2, linestyle="--")
        ax_acc.set_title(f"{title_name} - Accuracy Curves")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend()
        ax_acc.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = PLOTS_DIR / "cnn_training_convergence_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved CNN training curves to: {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    print("Generating comprehensive visualization figures...")
    plot_model_comparison()
    plot_confusion_matrices()
    plot_cnn_training_history()
    print("All visualizations successfully generated!")


if __name__ == "__main__":
    main()
