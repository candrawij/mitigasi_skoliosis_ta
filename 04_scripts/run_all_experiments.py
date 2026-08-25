"""
Master Runner Script: Runs all experiments sequentially or selectively.

Usage:
  python 04_scripts/run_all_experiments.py --all
  python 04_scripts/run_all_experiments.py --exp 1 2 3 5
"""
import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd, desc):
    print("\n" + "="*70)
    print(f"  [RUNNING] {desc}")
    print(f"  Command: {cmd}")
    print("="*70)
    ret = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        print(f"⚠️ Warning: {desc} exited with code {ret.returncode}")
    else:
        print(f"✅ Success: {desc}")
    return ret.returncode


def generate_master_summary():
    """Consolidate all experiment result JSON files into a master summary."""
    exp_dir = PROJECT_ROOT / "07_results" / "experiments"
    summary_rows = []

    # 1. EXP-01 EfficientNet
    exp01_file = exp_dir / "exp01_all_results.json"
    if exp01_file.exists():
        with open(exp01_file) as f:
            for item in json.load(f):
                summary_rows.append({
                    "Experiment": "EXP-01 (CNN Baseline)",
                    "Dataset": item.get("dataset"),
                    "Model": item.get("model", "EfficientNet-B0"),
                    "Accuracy": item.get("accuracy"),
                    "F1_Macro": item.get("f1_macro"),
                })

    # 2. EXP-02 Keypoints
    exp02_file = exp_dir / "exp02_all_results.json"
    if exp02_file.exists():
        with open(exp02_file) as f:
            for item in json.load(f):
                summary_rows.append({
                    "Experiment": "EXP-02 (Keypoint Classifiers)",
                    "Dataset": item.get("dataset"),
                    "Model": item.get("model"),
                    "Accuracy": item.get("accuracy"),
                    "F1_Macro": item.get("f1_macro"),
                })

    # 3. EXP-03 YOLO Pose
    for dname in ["PROJECT_DESIGN", "SITTING_POSTURE_DETECTION"]:
        yolo_file = exp_dir / f"EXP-YOLO-POSE-{dname}" / "results.json"
        if yolo_file.exists():
            with open(yolo_file) as f:
                for item in json.load(f):
                    summary_rows.append({
                        "Experiment": "EXP-03 (YOLO Pose + Clf)",
                        "Dataset": dname.lower(),
                        "Model": item.get("model"),
                        "Accuracy": item.get("accuracy"),
                        "F1_Macro": item.get("f1_macro"),
                    })

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        csv_path = exp_dir / "master_summary.csv"
        md_path = exp_dir / "master_summary.md"
        df_sum.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Master Summary of All Experiments\n\n")
            f.write(df_sum.to_markdown(index=False))
            f.write("\n")
        print(f"\n📊 Master Summary generated at:\n  - {csv_path}\n  - {md_path}")
        print("\n" + df_sum.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run posture detection experiments")
    parser.add_argument("--all", action="store_true", help="Run all preprocessing and experiments")
    parser.add_argument("--exp", nargs="+", type=int, choices=[1, 2, 3, 5], help="Specific experiments to run (1, 2, 3, 5)")
    args = parser.parse_args()

    run_all = args.all or (args.exp is None)
    selected_exps = set(args.exp) if args.exp else {1, 2, 3, 5}

    print("="*70)
    print("  POSTURE DETECTION RESEARCH — MASTER EXPERIMENT RUNNER")
    print(f"  Selected Experiments: {selected_exps if not run_all else 'ALL (1, 2, 3, 5)'}")
    print("="*70)

    # Preprocessing check
    proc_pe = PROJECT_ROOT / "02_data" / "processed" / "postureexercise" / "X_train.npy"
    if not proc_pe.exists() or run_all:
        run_cmd(f'python "{PROJECT_ROOT / "04_scripts" / "preprocessing" / "preprocess_postureexercise.py"}"', "Preprocessing Postureexercise")
        run_cmd(f'python "{PROJECT_ROOT / "04_scripts" / "preprocessing" / "preprocess_ikorn_4kp.py"}"', "Preprocessing IKORN 4-KP")

    # EXP-02: Keypoint Classifiers (MLP / XGBoost)
    if run_all or (2 in selected_exps):
        run_cmd(f'python "{PROJECT_ROOT / "04_scripts" / "training" / "train_keypoint_classifiers.py"}"', "EXP-02: Keypoint Classifiers")

    # EXP-01: EfficientNet-B0 Image Baseline
    if run_all or (1 in selected_exps):
        run_cmd(f'python "{PROJECT_ROOT / "04_scripts" / "training" / "train_efficientnet_baseline.py"}"', "EXP-01: EfficientNet-B0 Baseline")

    # EXP-03: YOLO Pose + Classifiers
    if run_all or (3 in selected_exps):
        run_cmd(f'python "{PROJECT_ROOT / "04_scripts" / "training" / "train_yolo_pose_classifier.py"}"', "EXP-03: YOLO Pose + Classifiers")

    # EXP-05: Cross Dataset Evaluation
    if run_all or (5 in selected_exps):
        run_cmd(f'python "{PROJECT_ROOT / "04_scripts" / "evaluation" / "evaluate_cross_dataset.py"}"', "EXP-05: Cross Dataset Evaluation")

    # Summary
    generate_master_summary()


if __name__ == "__main__":
    main()
