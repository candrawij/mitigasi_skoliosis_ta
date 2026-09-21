"""
3D-08 + 3D-09 -- Build 2D-3D Intersection and Common Subject-Aware 5-Fold
==========================================================================
Outputs:
    02_data/private_processed/manifests/private_6class_intersection.csv
    03_metadata/private_final_split/private_2d3d_intersection_5fold.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(r"d:\.Candra\Project\TA")
FEAT2D_CSV  = ROOT / "02_data/private_processed/features/private_features_2d_v2.csv"
FEAT3D_CSV  = ROOT / "02_data/private_processed/features/private_features_3d.csv"
OUT_ISECT   = ROOT / "02_data/private_processed/manifests/private_6class_intersection.csv"
OUT_FOLDS   = ROOT / "03_metadata/private_final_split/private_2d3d_intersection_5fold.csv"

MAIN_CLASSES = sorted(["upright","leaning_forward","leaning_backward","leaning_left","leaning_right","slouching"])
N_FOLDS = 5
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("3D-08+09 -- Build 2D-3D Intersection + Common 5-Fold")
    print("=" * 60)

    feat2d = pd.read_csv(FEAT2D_CSV)
    feat3d = pd.read_csv(FEAT3D_CSV)
    print(f"\n2D features: {len(feat2d)} captures, {(feat2d['status_2d']=='USABLE').sum()} usable")
    print(f"3D features: {len(feat3d)} captures, {(feat3d['status_3d']=='USABLE').sum()} usable")

    # Usable sets
    cap2d_usable = set(feat2d[feat2d["status_2d"] == "USABLE"]["capture_id"])
    cap3d_usable = set(feat3d[feat3d["status_3d"] == "USABLE"]["capture_id"])

    intersection_caps = cap2d_usable & cap3d_usable

    print(f"\n2D usable  : {len(cap2d_usable)}")
    print(f"3D usable  : {len(cap3d_usable)}")
    print(f"Intersection: {len(intersection_caps)}")

    # Build intersection manifest
    df_3d_usable = feat3d[feat3d["capture_id"].isin(intersection_caps)][
        ["capture_id","subject_id","label","class_id","calibration_id","lateral_side","subset"]
    ].copy()
    df_3d_usable = df_3d_usable.sort_values("capture_id").reset_index(drop=True)

    print(f"\nIntersection label distribution:")
    for lbl, cnt in df_3d_usable["label"].value_counts().sort_index().items():
        print(f"  {lbl:<22} : {cnt}")

    print(f"\nIntersection subject distribution:")
    for sid, cnt in df_3d_usable["subject_id"].value_counts().sort_index().items():
        print(f"  {sid} : {cnt}")

    # Assertion: no subject in both train and test (checked at fold level)
    assert not df_3d_usable["capture_id"].duplicated().any(), "Duplicate capture_id!"
    assert len(df_3d_usable) == len(intersection_caps), "Size mismatch!"

    OUT_ISECT.parent.mkdir(parents=True, exist_ok=True)
    df_3d_usable.to_csv(OUT_ISECT, index=False)
    print(f"\n[DONE] Intersection saved -> {OUT_ISECT}")

    # ---- 3D-09: Common Subject-Aware 5-Fold ----
    print("\n" + "=" * 60)
    print("3D-09 -- Generate Common Subject-Aware 5-Fold")
    print("=" * 60)

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    X_idx = np.arange(len(df_3d_usable))
    y = df_3d_usable["class_id"].values
    groups = df_3d_usable["subject_id"].values

    fold_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_idx, y, groups)):
        train_subjects = set(groups[train_idx])
        test_subjects  = set(groups[test_idx])
        overlap = train_subjects & test_subjects
        assert len(overlap) == 0, f"Fold {fold_idx}: Subject overlap! {overlap}"

        for i in train_idx:
            fold_rows.append({
                "capture_id": df_3d_usable.iloc[i]["capture_id"],
                "subject_id": df_3d_usable.iloc[i]["subject_id"],
                "label": df_3d_usable.iloc[i]["label"],
                "class_id": df_3d_usable.iloc[i]["class_id"],
                "fold": fold_idx,
                "split": "train"
            })
        for i in test_idx:
            fold_rows.append({
                "capture_id": df_3d_usable.iloc[i]["capture_id"],
                "subject_id": df_3d_usable.iloc[i]["subject_id"],
                "label": df_3d_usable.iloc[i]["label"],
                "class_id": df_3d_usable.iloc[i]["class_id"],
                "fold": fold_idx,
                "split": "test"
            })

        train_labs = df_3d_usable.iloc[train_idx]["label"].value_counts().sort_index()
        test_labs  = df_3d_usable.iloc[test_idx]["label"].value_counts().sort_index()
        print(f"\n  Fold {fold_idx}: train={len(train_idx)} test={len(test_idx)}")
        print(f"    train subjects: {sorted(train_subjects)}")
        print(f"    test  subjects: {sorted(test_subjects)}")
        print(f"    train labels: {train_labs.to_dict()}")
        print(f"    test  labels: {test_labs.to_dict()}")
        print(f"    [OK] No subject overlap.")

    df_folds = pd.DataFrame(fold_rows)

    OUT_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(OUT_FOLDS, index=False)
    print(f"\n[DONE] Fold file saved -> {OUT_FOLDS}")
    print(f"  Total rows: {len(df_folds)}  (each capture appears {N_FOLDS}x: once per fold)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  2D usable        : {len(cap2d_usable)}")
    print(f"  3D usable        : {len(cap3d_usable)}")
    print(f"  Intersection     : {len(intersection_caps)}")
    n_subjects = df_3d_usable["subject_id"].nunique()
    print(f"  Subjects covered : {n_subjects}")
    print(f"  Folds created    : {N_FOLDS}")
    print(f"  Subject-aware    : YES (verified no overlap)")

if __name__ == "__main__":
    main()
