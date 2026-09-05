"""
test_private_single_capture.py — Evaluate Deployment Model on Private Dataset Captures
Supports single capture verification and full subject batch functional testing.

Usage:
  # Single capture test (2D):
  python 04_scripts/evaluation/test_private_single_capture.py --capture-id S024_SES1_R_upright_001 --mode 2d

  # Single capture test (3D):
  python 04_scripts/evaluation/test_private_single_capture.py --capture-id S024_SES1_R_leaning_forward_001 --mode 3d

  # Batch functional test on all captures of a subject:
  python 04_scripts/evaluation/test_private_single_capture.py --subject S024 --mode 2d
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
from private_feature_common import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    MAIN_CLASSES,
    NUM_CLASSES,
    FEATURE_NAMES_2D,
    FEATURE_NAMES_3D
)

DATA_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
MANIFEST_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "manifests"
MODELS_DIR = PROJECT_ROOT / "06_models"
EXP_DIR = PROJECT_ROOT / "07_results" / "experiments" / "private_final"


def load_deployment_model(mode: str):
    m_dir = MODELS_DIR / f"keypoint_{mode.lower()}" / "private_final"
    model_path = m_dir / "model.pkl"
    scaler_path = m_dir / "scaler.pkl"
    meta_path = m_dir / "model_metadata.json"
    class_map_path = m_dir / "class_map.json"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Deployment model artifacts not found in {m_dir}. Run fit_private_deployment_models.py first!")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    return model, scaler, meta, class_map


def test_single_capture(capture_id: str, mode: str, use_oof: bool = False):
    print("=" * 80)
    print(f"  SINGLE CAPTURE TEST: {capture_id} (Mode: {mode.upper()})")
    print("=" * 80)

    # Load feature dataset
    feat_file = DATA_DIR / f"private_features_{mode.lower()}.csv"
    if not feat_file.exists():
        raise FileNotFoundError(f"Features file {feat_file} not found.")

    df_feat = pd.read_csv(feat_file)
    row = df_feat[df_feat["capture_id"] == capture_id]

    if row.empty:
        # Try matching substring or case-insensitive
        matches = df_feat[df_feat["capture_id"].str.contains(capture_id, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[[0]]
            capture_id = row["capture_id"].values[0]
        else:
            print(f"Error: Capture ID '{capture_id}' not found in {feat_file.name}.")
            return

    row = row.iloc[0]
    sub_id = row["subject_id"]
    gt_label = row["label"]
    gt_class_id = int(row["class_id"])

    # Check status
    status_col = f"status_{mode.lower()}"
    status_val = row.get(status_col, "UNKNOWN")
    if status_val != "USABLE":
        print(f"Capture ID   : {capture_id}")
        print(f"Subject      : {sub_id}")
        print(f"Ground Truth : {gt_label}")
        print(f"Prediction   : REJECT / INVALID (QC Status: {status_val})")
        print(f"Status       : REJECTED_BY_GATE")
        return

    features = FEATURE_NAMES_2D if mode == "2d" else FEATURE_NAMES_3D
    x_raw = row[features].values.astype(np.float64).reshape(1, -1)

    if use_oof:
        oof_file = EXP_DIR / mode.lower() / "oof_predictions.csv"
        if oof_file.exists():
            df_oof = pd.read_csv(oof_file)
            oof_row = df_oof[df_oof["capture_id"] == capture_id]
            if not oof_row.empty:
                r = oof_row.iloc[0]
                pred_label = r["label_pred"]
                pred_class_id = int(r["y_pred"])
                prob = float(r[f"prob_{pred_label}"])
                is_correct = bool(r["is_correct"])
                print(f"[NOTE] Using unbiased OOF model prediction (Fold {r['fold_id']})")
                print(f"Capture ID   : {capture_id}")
                print(f"Subject      : {sub_id}")
                print(f"Ground Truth : {gt_label}")
                print(f"Prediction   : {pred_label}")
                print(f"Confidence   : {prob:.4f} ({prob*100:.1f}%)")
                print(f"Status       : {'CORRECT' if is_correct else 'INCORRECT'}")
                return

    # Deployment model prediction
    model, scaler, meta, class_map = load_deployment_model(mode)
    x_scaled = scaler.transform(x_raw)

    y_pred = model.predict(x_scaled)[0]
    y_prob = model.predict_proba(x_scaled)[0]

    pred_label = ID_TO_CLASS[int(y_pred)]
    conf = float(y_prob[int(y_pred)])
    is_correct = (gt_class_id == int(y_pred))

    print(f"Capture ID   : {capture_id}")
    print(f"Subject      : {sub_id}")
    print(f"Ground Truth : {gt_label}")
    print(f"Prediction   : {pred_label}")
    print(f"Confidence   : {conf:.4f} ({conf*100:.1f}%)")
    print(f"Status       : {'CORRECT' if is_correct else 'INCORRECT'}")

    print("\nClass Probabilities:")
    for c_id, c_name in enumerate(MAIN_CLASSES):
        bar = "█" * int(y_prob[c_id] * 20)
        marker = " <-- PREDICTION" if c_id == int(y_pred) else ""
        print(f"  {c_name:<18} : {y_prob[c_id]:.4f} ({y_prob[c_id]*100:5.1f}%) {bar}{marker}")


def test_subject_batch(subject_id: str, mode: str):
    print("=" * 80)
    print(f"  BATCH FUNCTIONAL TEST: SUBJECT {subject_id} (Mode: {mode.upper()})")
    print("=" * 80)

    feat_file = DATA_DIR / f"private_features_{mode.lower()}.csv"
    if not feat_file.exists():
        raise FileNotFoundError(f"Features file {feat_file} not found.")

    df_feat = pd.read_csv(feat_file)
    df_sub = df_feat[df_feat["subject_id"] == subject_id].copy().reset_index(drop=True)

    if df_sub.empty:
        print(f"Error: Subject '{subject_id}' not found in {feat_file.name}.")
        return

    model, scaler, meta, class_map = load_deployment_model(mode)
    features = FEATURE_NAMES_2D if mode == "2d" else FEATURE_NAMES_3D

    status_col = f"status_{mode.lower()}"

    total_caps = len(df_sub)
    usable_caps = len(df_sub[df_sub[status_col] == "USABLE"])
    rejected_caps = total_caps - usable_caps

    print(f"Total Captures for {subject_id}: {total_caps}")
    print(f"Usable for Evaluation: {usable_caps} | Rejected by Gate: {rejected_caps}\n")

    class_counts = {c: {"total": 0, "correct": 0, "usable": 0} for c in MAIN_CLASSES}

    correct_total = 0
    for idx, row in df_sub.iterrows():
        gt_label = row["label"]
        class_counts[gt_label]["total"] += 1

        if row[status_col] != "USABLE":
            continue

        class_counts[gt_label]["usable"] += 1
        x_raw = row[features].values.astype(np.float64).reshape(1, -1)
        x_scaled = scaler.transform(x_raw)
        y_pred = int(model.predict(x_scaled)[0])
        pred_label = ID_TO_CLASS[y_pred]

        if pred_label == gt_label:
            class_counts[gt_label]["correct"] += 1
            correct_total += 1

    print(f"{'Posture Class':<20} {'Correct / Usable':<18} {'Accuracy (%)':<15} {'Total Captures'}")
    print("-" * 65)
    for c_name in MAIN_CLASSES:
        corr = class_counts[c_name]["correct"]
        usab = class_counts[c_name]["usable"]
        tot = class_counts[c_name]["total"]
        acc_str = f"{(corr/usab*100):.1f}%" if usab > 0 else "N/A (Rejected)"
        print(f"{c_name:<20} {corr:>3} / {usab:<12} {acc_str:<15} {tot:>3}")

    print("-" * 65)
    overall_acc = (correct_total / usable_caps * 100) if usable_caps > 0 else 0.0
    print(f"{'OVERALL':<20} {correct_total:>3} / {usable_caps:<12} {overall_acc:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Test Private Deployment Models on Captures")
    parser.add_argument("--capture-id", type=str, default=None, help="Capture ID to test")
    parser.add_argument("--subject", type=str, default=None, help="Subject ID for batch testing (e.g. S024)")
    parser.add_argument("--mode", type=str, default="2d", choices=["2d", "3d"], help="Representation mode (2d or 3d)")
    parser.add_argument("--use-oof", action="store_true", help="Cross-reference with OOF prediction for unbiased evaluation demo")

    args = parser.parse_args()

    if args.capture_id:
        test_single_capture(args.capture_id, mode=args.mode, use_oof=args.use_oof)
    elif args.subject:
        test_subject_batch(args.subject, mode=args.mode)
    else:
        # Default test demonstration
        print("No --capture-id or --subject specified. Running default demonstration on S024...")
        test_subject_batch("S024", mode=args.mode)


if __name__ == "__main__":
    main()
