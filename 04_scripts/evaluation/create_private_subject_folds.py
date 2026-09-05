"""
create_private_subject_folds.py — Generate Subject-Aware Stratified Grouped 5-Fold Splits.
Ensures zero data leakage between training and testing subjects in every fold.

Input:
  02_data/private_processed/manifests/private_6class_intersection.csv

Output:
  03_metadata/private_final_split/private_stratified_group_5fold.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
from private_feature_common import MAIN_CLASSES

MANIFESTS_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "manifests"
SPLIT_DIR = PROJECT_ROOT / "03_metadata" / "private_final_split"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def create_subject_folds():
    print("=" * 80)
    print("  STEP 9: GENERATE SUBJECT-AWARE STRATIFIED GROUPED 5-FOLD SPLITS")
    print("=" * 80)

    inter_file = MANIFESTS_DIR / "private_6class_intersection.csv"
    if not inter_file.exists():
        raise FileNotFoundError(f"Intersection manifest missing: {inter_file}. Run build_private_intersection.py first!")

    df_inter = pd.read_csv(inter_file)
    n_samples = len(df_inter)
    n_subjects = df_inter["subject_id"].nunique()
    print(f"Loaded intersection dataset: {n_samples} captures across {n_subjects} subjects")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    X = df_inter
    y = df_inter["class_id"]
    groups = df_inter["subject_id"]

    fold_assignments = np.zeros(n_samples, dtype=int)

    fold_summary = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        fold_assignments[test_idx] = fold_idx

        train_subs = set(df_inter.iloc[train_idx]["subject_id"])
        test_subs = set(df_inter.iloc[test_idx]["subject_id"])
        overlap = train_subs.intersection(test_subs)

        # CRITICAL ASSERTION: Zero Subject Leakage
        assert len(overlap) == 0, f"STOP: Subject leakage detected in Fold {fold_idx}: {overlap}"

        test_labels = df_inter.iloc[test_idx]["label"].value_counts().to_dict()

        fold_summary.append({
            "fold_id": fold_idx,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_subjects_count": len(train_subs),
            "test_subjects_count": len(test_subs),
            "test_subjects": sorted(list(test_subs)),
            "test_class_distribution": test_labels
        })

    df_inter["fold_id"] = fold_assignments

    # Print summary table
    print("\n" + "=" * 80)
    print("  5-FOLD PARTITION SUMMARY & ANTI-LEAKAGE VERIFICATION")
    print("=" * 80)
    for s in fold_summary:
        f_id = s["fold_id"]
        n_tr = s["train_samples"]
        n_te = s["test_samples"]
        subs = ", ".join(s["test_subjects"])
        print(f"Fold {f_id}: Train = {n_tr:3d} ({s['train_subjects_count']:2d} subs) | "
              f"Test = {n_te:3d} ({s['test_subjects_count']:2d} subs: [{subs}]) | Overlap = 0")

    # Verification of class representation in all folds
    print("\nTest Class Distribution per Fold:")
    for f_id in range(5):
        f_df = df_inter[df_inter["fold_id"] == f_id]
        dist_str = ", ".join([f"{c}: {f_df['label'].value_counts().get(c, 0)}" for c in MAIN_CLASSES])
        print(f"  Fold {f_id}: {dist_str}")

    # Prepare final output dataframe
    cols_to_save = ["capture_id", "subject_id", "session_id", "class_id", "label",
                    "calibration_id", "lateral_side", "subset", "fold_id"]
    df_split = df_inter[cols_to_save].copy()

    out_split_csv = SPLIT_DIR / "private_stratified_group_5fold.csv"
    df_split.to_csv(out_split_csv, index=False)

    print(f"\n[SAVED] Stratified Group 5-Fold split saved to: {out_split_csv}")
    print("  ✓ CRITICAL ASSERTION PASSED: train_subjects ∩ test_subjects == empty in all 5 folds!")

    return df_split


if __name__ == "__main__":
    create_subject_folds()
