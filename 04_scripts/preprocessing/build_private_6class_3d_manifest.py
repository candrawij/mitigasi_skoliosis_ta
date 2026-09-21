import sys
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\.Candra\Project\TA")
CAPTURES_CSV = ROOT / "03_metadata/private_templates/captures.csv"
QC_3D_CSV    = ROOT / "07_results/private_audit/private_3d_qc_final.csv"
CALMAP_CSV   = ROOT / "03_metadata/private_templates/calibration_map.csv"
OUTPUT_CSV   = ROOT / "02_data/private_processed/manifests/private_6class_3d_manifest.csv"

MAIN_CLASSES = ["upright","leaning_forward","leaning_backward","leaning_left","leaning_right","slouching"]
EXCLUDE_LABELS = {"forward_head", "reject"}
EXPECTED_TOTAL = 727

def stop(msg):
    print(f"[STOP] {msg}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("3D-02 -- Build 6-Class 3D Manifest")
    print("=" * 60)

    captures = pd.read_csv(CAPTURES_CSV)
    print(f"\nLoaded captures.csv: {len(captures)} rows")
    print(captures["primary_posture"].value_counts().to_string())

    mask_6class = captures["primary_posture"].isin(MAIN_CLASSES)
    df_6class = captures[mask_6class].copy()
    excluded = captures.loc[~mask_6class, "primary_posture"].value_counts().to_dict()
    print(f"\nExcluded: {excluded}")
    print(f"6-class subset: {len(df_6class)} captures")

    if len(df_6class) != EXPECTED_TOTAL:
        stop(f"Expected {EXPECTED_TOTAL} captures, got {len(df_6class)}")
    remaining_excluded = set(df_6class["primary_posture"].unique()) & EXCLUDE_LABELS
    if remaining_excluded:
        stop(f"Excluded labels still present: {remaining_excluded}")
    dupes = df_6class["capture_id"].duplicated().sum()
    if dupes:
        stop(f"Found {dupes} duplicate capture_ids")
    missing_subj = df_6class["subject_id"].isna().sum()
    if missing_subj:
        stop(f"Found {missing_subj} captures with missing subject_id")
    print("[OK] All stop conditions passed")

    qc3d = pd.read_csv(QC_3D_CSV)
    calmap = pd.read_csv(CALMAP_CSV)

    # Exclude calibration_id from qc3d_slim — already in df_out from captures
    qc_cols = ["capture_id","decision","valid_3d_joints_core","valid_3d_joints_total","core_reproj_error_640p_px","shoulder_width_m","torso_length_m","depth_z_m","reason"]
    qc3d_slim = qc3d[qc_cols].copy()

    calmap_slim = calmap[["calibration_id","qc_status","mean_reprojection_error_px"]].copy()
    calmap_slim.rename(columns={"qc_status":"rig_qc_status","mean_reprojection_error_px":"rig_reproj_err_px"}, inplace=True)

    df_out = df_6class[["capture_id","subject_id","session_id","calibration_id","primary_posture","repetition","subset","lateral_side"]].copy()
    df_out.rename(columns={"primary_posture":"label"}, inplace=True)
    label_to_id = {lbl: i for i, lbl in enumerate(sorted(MAIN_CLASSES))}
    df_out["class_id"] = df_out["label"].map(label_to_id)

    df_out = df_out.merge(qc3d_slim, on="capture_id", how="left")
    df_out = df_out.merge(calmap_slim, on="calibration_id", how="left")

    missing_qc = df_out["decision"].isna().sum()
    if missing_qc:
        print(f"[WARNING] {missing_qc} captures have no 3D QC entry.")
        df_out["decision"] = df_out["decision"].fillna("NO_QC")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total 6-class captures : {len(df_out)}")
    print("\nLabel distribution:")
    for lbl, cnt in df_out["label"].value_counts().sort_index().items():
        print(f"  {lbl:<22} : {cnt}")
    print("\n3D Decision distribution:")
    for dec, cnt in df_out["decision"].value_counts().items():
        print(f"  {dec:<35} : {cnt}")

    usable = df_out["decision"].isin(["INCLUDE_3D_FULL","INCLUDE_3D_WITH_MASKING"]).sum()
    pct = usable / len(df_out) * 100
    full_only = (df_out["decision"] == "INCLUDE_3D_FULL").sum()
    print(f"\nUsable 3D (FULL+MASKING) : {usable} / {len(df_out)} ({pct:.1f}%)")
    print(f"FULL only                : {full_only}")
    print(f"Subjects                 : {df_out['subject_id'].nunique()}")
    print(f"Calibration rigs         : {df_out['calibration_id'].nunique()}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] Saved -> {OUTPUT_CSV}")
    print(f"Columns: {list(df_out.columns)}")

if __name__ == "__main__":
    main()
