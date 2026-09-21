"""
3D-03 -- Recalculate QC 3D for 6-class dataset
================================================
Reads the 6-class manifest and produces a detailed QC summary:
- FULL / MASKING / EXCLUDE counts per class, per subject, per rig
- Coverage percentages

Output:
    07_results/private_audit/private_3d_qc_6class_final.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\.Candra\Project\TA")
MANIFEST_CSV = ROOT / "02_data/private_processed/manifests/private_6class_3d_manifest.csv"
OUTPUT_CSV   = ROOT / "07_results/private_audit/private_3d_qc_6class_final.csv"

DECISION_ORDER = [
    "INCLUDE_3D_FULL",
    "INCLUDE_3D_WITH_MASKING",
    "EXCLUDE_3D_DEGENERATE_RIG",
    "EXCLUDE_3D",
    "NO_QC",
]

def main():
    print("=" * 60)
    print("3D-03 -- Recalculate QC 3D for 6-class dataset")
    print("=" * 60)

    df = pd.read_csv(MANIFEST_CSV)
    print(f"\nLoaded manifest: {len(df)} captures")

    MAIN_CLASSES = sorted(df["label"].unique().tolist())
    print(f"Classes: {MAIN_CLASSES}")

    # --- Helper maps ---
    df["is_full"]    = df["decision"] == "INCLUDE_3D_FULL"
    df["is_masking"] = df["decision"] == "INCLUDE_3D_WITH_MASKING"
    df["is_usable"]  = df["decision"].isin(["INCLUDE_3D_FULL", "INCLUDE_3D_WITH_MASKING"])
    df["is_exclude"] = ~df["is_usable"]

    # ── A. Overall Summary ─────────────────────────────────────────────────────
    total        = len(df)
    full_n       = int(df["is_full"].sum())
    masking_n    = int(df["is_masking"].sum())
    usable_n     = int(df["is_usable"].sum())
    exclude_n    = total - usable_n

    print(f"\n{'='*60}")
    print("A. OVERALL (6-class)")
    print(f"{'='*60}")
    print(f"  Raw 6-class captures   : {total}")
    print(f"  FULL                   : {full_n}  ({full_n/total*100:.1f}%)")
    print(f"  MASKING                : {masking_n}  ({masking_n/total*100:.1f}%)")
    print(f"  Usable (FULL+MASKING)  : {usable_n}  ({usable_n/total*100:.1f}%)")
    print(f"  EXCLUDE                : {exclude_n}  ({exclude_n/total*100:.1f}%)")

    # ── B. Per-class summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("B. PER CLASS")
    print(f"{'='*60}")
    class_rows = []
    for lbl in MAIN_CLASSES:
        sub = df[df["label"] == lbl]
        n   = len(sub)
        f   = int(sub["is_full"].sum())
        m   = int(sub["is_masking"].sum())
        u   = int(sub["is_usable"].sum())
        e   = n - u
        class_rows.append({
            "label": lbl,
            "raw": n, "full": f, "masking": m,
            "usable": u, "exclude": e,
            "usable_pct": round(u/n*100, 1)
        })
        print(f"  {lbl:<22} raw={n}  full={f}  mask={m}  usable={u}({u/n*100:.0f}%)  excl={e}")
    df_class = pd.DataFrame(class_rows)

    # ── C. Per-subject summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C. PER SUBJECT")
    print(f"{'='*60}")
    subj_rows = []
    for sid in sorted(df["subject_id"].unique()):
        sub = df[df["subject_id"] == sid]
        n   = len(sub)
        u   = int(sub["is_usable"].sum())
        e   = n - u
        subj_rows.append({
            "subject_id": sid, "raw": n,
            "usable": u, "exclude": e,
            "usable_pct": round(u/n*100, 1)
        })
        print(f"  {sid}  raw={n}  usable={u}({u/n*100:.0f}%)  excl={e}")
    df_subj = pd.DataFrame(subj_rows)

    # ── D. Per-rig summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("D. PER CALIBRATION RIG")
    print(f"{'='*60}")
    rig_rows = []
    for rig in sorted(df["calibration_id"].unique()):
        sub = df[df["calibration_id"] == rig]
        n   = len(sub)
        u   = int(sub["is_usable"].sum())
        e   = n - u
        rig_rows.append({
            "calibration_id": rig, "raw": n,
            "usable": u, "exclude": e,
            "usable_pct": round(u/n*100, 1)
        })
        print(f"  {rig}  raw={n}  usable={u}({u/n*100:.0f}%)  excl={e}")
    df_rig = pd.DataFrame(rig_rows)

    # ── E. Decision breakdown ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("E. DECISION BREAKDOWN")
    print(f"{'='*60}")
    for dec, cnt in df["decision"].value_counts().items():
        print(f"  {dec:<35} : {cnt}  ({cnt/total*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Main output: one row per capture with QC columns enriched
    df_qc = df[["capture_id","subject_id","label","calibration_id","decision",
                "is_full","is_masking","is_usable","is_exclude",
                "valid_3d_joints_core","valid_3d_joints_total",
                "core_reproj_error_640p_px","shoulder_width_m","torso_length_m",
                "depth_z_m","reason"]].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_qc.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] QC CSV saved -> {OUTPUT_CSV}")

    # Also save summary tables
    summary_dir = ROOT / "07_results/private_audit"
    df_class.to_csv(summary_dir / "private_3d_qc_6class_per_class.csv", index=False)
    df_subj.to_csv(summary_dir  / "private_3d_qc_6class_per_subject.csv", index=False)
    df_rig.to_csv(summary_dir   / "private_3d_qc_6class_per_rig.csv", index=False)
    print(f"Summaries saved (per_class, per_subject, per_rig).")

    print(f"\n{'='*60}")
    print(f"FINAL NUMBERS")
    print(f"{'='*60}")
    print(f"  6-class raw     : {total}")
    print(f"  Usable 3D       : {usable_n} ({usable_n/total*100:.1f}%)")
    print(f"  FULL            : {full_n}")
    print(f"  MASKING         : {masking_n}")
    print(f"  EXCLUDE         : {exclude_n}")

if __name__ == "__main__":
    main()
