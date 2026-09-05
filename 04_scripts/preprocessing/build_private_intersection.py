"""
build_private_intersection.py — Generate Manifests and Feature Sets for 2D, 3D, and Intersection.

Outputs:
  - 02_data/private_processed/manifests/private_6class_2d.csv
  - 02_data/private_processed/manifests/private_6class_3d.csv
  - 02_data/private_processed/manifests/private_6class_intersection.csv
  - 02_data/private_processed/features/private_features_2d_intersection.csv
  - 02_data/private_processed/features/private_features_3d_intersection.csv
  - 02_data/private_processed/audit/intersection_audit.csv
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "preprocessing"))
from private_feature_common import MAIN_CLASSES

MANIFESTS_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "manifests"
FEATURES_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "features"
AUDIT_DIR = PROJECT_ROOT / "02_data" / "private_processed" / "audit"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def build_intersection():
    print("=" * 80)
    print("  STEP 8: BUILD 2D / 3D / INTERSECTION DATASETS & MANIFESTS")
    print("=" * 80)

    manifest_all_p = MANIFESTS_DIR / "private_6class_all.csv"
    feat_2d_p = FEATURES_DIR / "private_features_2d.csv"
    feat_3d_p = FEATURES_DIR / "private_features_3d.csv"

    if not manifest_all_p.exists() or not feat_2d_p.exists() or not feat_3d_p.exists():
        raise FileNotFoundError("Prerequisite files missing. Ensure build_private_6class_manifest.py, "
                                "extract_private_2d_features.py, and extract_private_3d_features.py have run!")

    df_manifest_all = pd.read_csv(manifest_all_p)
    df_feat_2d = pd.read_csv(feat_2d_p)
    df_feat_3d = pd.read_csv(feat_3d_p)

    raw_total = len(df_manifest_all)
    assert raw_total == 727, f"Expected 727 raw captures, found {raw_total}"

    # 1. 2D Usable Manifest & Features
    valid_2d_cids = set(df_feat_2d[df_feat_2d["status_2d"] == "USABLE"]["capture_id"])
    df_manifest_2d = df_manifest_all[df_manifest_all["capture_id"].isin(valid_2d_cids)].copy()
    df_features_2d_usable = df_feat_2d[df_feat_2d["capture_id"].isin(valid_2d_cids)].copy()

    # 2. 3D Usable Manifest & Features
    valid_3d_cids = set(df_feat_3d[df_feat_3d["status_3d"] == "USABLE"]["capture_id"])
    df_manifest_3d = df_manifest_all[df_manifest_all["capture_id"].isin(valid_3d_cids)].copy()
    df_features_3d_usable = df_feat_3d[df_feat_3d["capture_id"].isin(valid_3d_cids)].copy()

    # 3. Intersection Manifest & Features
    intersection_cids = sorted(list(valid_2d_cids.intersection(valid_3d_cids)))
    df_manifest_inter = df_manifest_all[df_manifest_all["capture_id"].isin(intersection_cids)].copy()

    df_feat_2d_inter = df_feat_2d[df_feat_2d["capture_id"].isin(intersection_cids)].copy()
    df_feat_3d_inter = df_feat_3d[df_feat_3d["capture_id"].isin(intersection_cids)].copy()

    # Sort consistently by capture_id
    df_manifest_2d = df_manifest_2d.sort_values("capture_id").reset_index(drop=True)
    df_manifest_3d = df_manifest_3d.sort_values("capture_id").reset_index(drop=True)
    df_manifest_inter = df_manifest_inter.sort_values("capture_id").reset_index(drop=True)
    df_feat_2d_inter = df_feat_2d_inter.sort_values("capture_id").reset_index(drop=True)
    df_feat_3d_inter = df_feat_3d_inter.sort_values("capture_id").reset_index(drop=True)

    # Verify that intersection 2D and 3D features have identical capture order
    assert (df_feat_2d_inter["capture_id"].values == df_feat_3d_inter["capture_id"].values).all(), \
        "Order mismatch between 2D and 3D intersection features!"

    print(f"Raw 6-Class Total:     {raw_total} captures")
    print(f"2D Usable Total:       {len(df_manifest_2d)} captures ({len(df_manifest_2d)/raw_total*100:.2f}%)")
    print(f"3D Usable Total:       {len(df_manifest_3d)} captures ({len(df_manifest_3d)/raw_total*100:.2f}%)")
    print(f"Intersection Usable:   {len(df_manifest_inter)} captures ({len(df_manifest_inter)/raw_total*100:.2f}%)")
    print(f"Subjects in Inter:     {df_manifest_inter['subject_id'].nunique()} subjects")

    # Class distribution breakdown table
    print("\n" + "=" * 80)
    print("  COVERAGE & CLASS DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(f"{'Class Name':20s} | {'Raw 6-Class':12s} | {'2D Usable':10s} | {'3D Usable':10s} | {'Intersection':12s}")
    print("-" * 75)

    audit_rows = []
    for cls in MAIN_CLASSES:
        raw_cnt = (df_manifest_all["label"] == cls).sum()
        c2d_cnt = (df_manifest_2d["label"] == cls).sum()
        c3d_cnt = (df_manifest_3d["label"] == cls).sum()
        inter_cnt = (df_manifest_inter["label"] == cls).sum()

        print(f"{cls:20s} | {raw_cnt:12d} | {c2d_cnt:10d} | {c3d_cnt:10d} | {inter_cnt:12d}")

        audit_rows.append({
            "class_name": cls,
            "raw_6class_count": raw_cnt,
            "usable_2d_count": c2d_cnt,
            "usable_3d_count": c3d_cnt,
            "intersection_count": inter_cnt,
            "coverage_2d_pct": round(c2d_cnt / raw_cnt * 100, 2),
            "coverage_3d_pct": round(c3d_cnt / raw_cnt * 100, 2),
            "coverage_intersection_pct": round(inter_cnt / raw_cnt * 100, 2)
        })

    print("-" * 75)
    print(f"{'TOTAL':20s} | {raw_total:12d} | {len(df_manifest_2d):10d} | {len(df_manifest_3d):10d} | {len(df_manifest_inter):12d}")

    # Total audit row
    audit_rows.append({
        "class_name": "TOTAL",
        "raw_6class_count": raw_total,
        "usable_2d_count": len(df_manifest_2d),
        "usable_3d_count": len(df_manifest_3d),
        "intersection_count": len(df_manifest_inter),
        "coverage_2d_pct": round(len(df_manifest_2d) / raw_total * 100, 2),
        "coverage_3d_pct": round(len(df_manifest_3d) / raw_total * 100, 2),
        "coverage_intersection_pct": round(len(df_manifest_inter) / raw_total * 100, 2)
    })
    df_audit = pd.DataFrame(audit_rows)

    # Save all generated files
    f_m2d = MANIFESTS_DIR / "private_6class_2d.csv"
    f_m3d = MANIFESTS_DIR / "private_6class_3d.csv"
    f_minter = MANIFESTS_DIR / "private_6class_intersection.csv"
    f_feat2d_inter = FEATURES_DIR / "private_features_2d_intersection.csv"
    f_feat3d_inter = FEATURES_DIR / "private_features_3d_intersection.csv"
    f_audit = AUDIT_DIR / "intersection_audit.csv"

    df_manifest_2d.to_csv(f_m2d, index=False)
    df_manifest_3d.to_csv(f_m3d, index=False)
    df_manifest_inter.to_csv(f_minter, index=False)
    df_feat_2d_inter.to_csv(f_feat2d_inter, index=False)
    df_feat_3d_inter.to_csv(f_feat3d_inter, index=False)
    df_audit.to_csv(f_audit, index=False)

    print(f"\n[SAVED] 2D Manifest:           {f_m2d}")
    print(f"[SAVED] 3D Manifest:           {f_m3d}")
    print(f"[SAVED] Intersection Manifest: {f_minter}")
    print(f"[SAVED] 2D Inter Features:     {f_feat2d_inter}")
    print(f"[SAVED] 3D Inter Features:     {f_feat3d_inter}")
    print(f"[SAVED] Intersection Audit:    {f_audit}")

    return df_manifest_inter


if __name__ == "__main__":
    build_intersection()
