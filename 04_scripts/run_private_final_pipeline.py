"""
run_private_final_pipeline.py — Master Orchestrator for Final 2D vs 3D XGBoost Pipeline
Executes or audits all 9 core pipeline stages from raw data manifests to deployment models:

  Step 1: Build 6-Class Manifest (private_6class_all.csv, N=727)
  Step 2: Extract 36 2D Multi-View Features (private_features_2d.csv, N=704)
  Step 3: Extract 25 Stereo 3D Features (private_features_3d.csv, N=403)
  Step 4: Build Intersection Manifest & Features (N=403, 18 subjects)
  Step 5: Subject-Aware Stratified Group 5-Fold Partitioning (Zero subject leakage)
  Step 6: Train & Evaluate 2D Multi-View XGBoost (Subject-Aware 5-Fold CV)
  Step 7: Train & Evaluate Stereo 3D XGBoost (Subject-Aware 5-Fold CV)
  Step 8: Formal Head-to-Head Comparison & Failure Analysis (2D vs 3D)
  Step 9: Fit & Serialize Final Deployment Models & Scalers

Usage:
  # Check status and run only missing steps:
  python 04_scripts/run_private_final_pipeline.py --skip-existing

  # Force re-run from step 6 onwards:
  python 04_scripts/run_private_final_pipeline.py --from-step 6

  # Audit / verification only (does not re-run):
  python 04_scripts/run_private_final_pipeline.py --audit-only
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_STEPS = [
    {
        "step": 1,
        "name": "Build 6-Class Manifest",
        "script": "04_scripts/preprocessing/build_private_6class_manifest.py",
        "expected_outputs": [
            "02_data/private_processed/manifests/private_6class_all.csv"
        ]
    },
    {
        "step": 2,
        "name": "Extract 36 2D Multi-View Features",
        "script": "04_scripts/preprocessing/extract_private_2d_features.py",
        "expected_outputs": [
            "02_data/private_processed/features/private_features_2d.csv",
            "02_data/private_processed/audit/feature_2d_audit.csv"
        ]
    },
    {
        "step": 3,
        "name": "Extract 25 Stereo 3D Features",
        "script": "04_scripts/preprocessing/extract_private_3d_features.py",
        "expected_outputs": [
            "02_data/private_processed/features/private_features_3d.csv",
            "02_data/private_processed/audit/feature_3d_audit.csv"
        ]
    },
    {
        "step": 4,
        "name": "Build Intersection Manifests & Features",
        "script": "04_scripts/preprocessing/build_private_intersection.py",
        "expected_outputs": [
            "02_data/private_processed/manifests/private_6class_intersection.csv",
            "02_data/private_processed/features/private_features_2d_intersection.csv",
            "02_data/private_processed/features/private_features_3d_intersection.csv"
        ]
    },
    {
        "step": 5,
        "name": "Subject-Aware Stratified Group 5-Fold Partitioning",
        "script": "04_scripts/evaluation/create_private_subject_folds.py",
        "expected_outputs": [
            "03_metadata/private_final_split/private_stratified_group_5fold.csv"
        ]
    },
    {
        "step": 6,
        "name": "Train & Evaluate XGBoost 2D Multi-View (5-Fold CV)",
        "script": "04_scripts/training/train_private_xgboost_2d.py",
        "expected_outputs": [
            "07_results/experiments/private_final/2d/oof_predictions.csv",
            "07_results/experiments/private_final/2d/summary_metrics.json"
        ]
    },
    {
        "step": 7,
        "name": "Train & Evaluate XGBoost Stereo 3D (5-Fold CV)",
        "script": "04_scripts/training/train_private_xgboost_3d.py",
        "expected_outputs": [
            "07_results/experiments/private_final/3d/oof_predictions.csv",
            "07_results/experiments/private_final/3d/summary_metrics.json"
        ]
    },
    {
        "step": 8,
        "name": "Formal Comparison & Failure Analysis (2D vs 3D)",
        "script": "04_scripts/evaluation/evaluate_private_2d_vs_3d.py",
        "expected_outputs": [
            "07_results/experiments/private_final/comparison/comparison_metrics.csv",
            "07_results/experiments/private_final/comparison/per_class_comparison.csv",
            "07_results/experiments/private_final/comparison/paired_capture_predictions.csv",
            "07_results/experiments/private_final/comparison/confusion_matrix_comparison.png"
        ]
    },
    {
        "step": 9,
        "name": "Fit & Serialize Final Deployment Models & Scalers",
        "script": "04_scripts/training/fit_private_deployment_models.py",
        "expected_outputs": [
            "06_models/keypoint_2d/private_final/model.pkl",
            "06_models/keypoint_2d/private_final/scaler.pkl",
            "06_models/keypoint_3d/private_final/model.pkl",
            "06_models/keypoint_3d/private_final/scaler.pkl"
        ]
    }
]


def check_step_outputs(step_info):
    """Check if all expected output files exist."""
    missing = []
    for rel_path in step_info["expected_outputs"]:
        p = PROJECT_ROOT / rel_path
        if not p.exists() or p.stat().st_size == 0:
            missing.append(rel_path)
    return len(missing) == 0, missing


def run_pipeline(from_step=1, to_step=9, skip_existing=False, audit_only=False):
    print("=" * 80)
    print("  MASTER ORCHESTRATOR: PRIVATE FINAL 2D–3D XGBOOST PIPELINE")
    print("=" * 80)
    print(f"Workspace Root : {PROJECT_ROOT}")
    print(f"Python Exec    : {sys.executable}")
    print(f"Step Range     : Step {from_step} -> Step {to_step}")
    print(f"Skip Existing  : {skip_existing}")
    print(f"Audit Only     : {audit_only}\n")

    overall_start = time.time()
    step_results = []

    for item in PIPELINE_STEPS:
        s_num = item["step"]
        s_name = item["name"]
        s_script = item["script"]

        if s_num < from_step or s_num > to_step:
            continue

        outputs_ok, missing_files = check_step_outputs(item)

        print("-" * 80)
        print(f"STEP {s_num}: {s_name}")
        print(f"Script: {s_script}")

        if audit_only:
            status_str = "[OK] Completed & Verified" if outputs_ok else f"[MISSING] {missing_files}"
            print(f"Status: {status_str}")
            step_results.append((s_num, s_name, "VERIFIED" if outputs_ok else "MISSING", 0.0))
            continue

        if skip_existing and outputs_ok:
            print("[SKIPPED] All target outputs exist and verified. Skipping execution.")
            step_results.append((s_num, s_name, "SKIPPED_EXISTS", 0.0))
            continue

        # Execute script
        script_path = PROJECT_ROOT / s_script
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            step_results.append((s_num, s_name, "FAILED_SCRIPT_NOT_FOUND", 0.0))
            break

        t0 = time.time()
        print(f">>> Executing: {sys.executable} {s_script} ...")
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
        dur = time.time() - t0

        if res.returncode != 0:
            print(f"\n[ERROR] Step {s_num} failed with returncode {res.returncode}!")
            step_results.append((s_num, s_name, f"FAILED_CODE_{res.returncode}", dur))
            break

        # Post-execution verification
        outputs_ok, missing_files = check_step_outputs(item)
        if not outputs_ok:
            print(f"\n[ERROR] Step {s_num} succeeded but missing outputs: {missing_files}!")
            step_results.append((s_num, s_name, "FAILED_MISSING_OUTPUTS", dur))
            break

        print(f"[SUCCESS] Step {s_num} completed in {dur:.1f}s.")
        step_results.append((s_num, s_name, "SUCCESS", dur))

    total_dur = time.time() - overall_start

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION SUMMARY DASHBOARD")
    print("=" * 80)
    print(f"{'Step':<6} {'Stage Name':<50} {'Status':<16} {'Duration'}")
    print("-" * 80)
    for s_num, s_name, status, dur in step_results:
        dur_str = f"{dur:5.1f}s" if dur > 0 else "-"
        print(f"{s_num:<6} {s_name:<50} {status:<16} {dur_str}")
    print("-" * 80)
    print(f"Total Execution Time: {total_dur:.1f}s")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Private Final 2D-3D XGBoost Pipeline")
    parser.add_argument("--from-step", type=int, default=1, help="Starting step (1-9)")
    parser.add_argument("--to-step", type=int, default=9, help="Ending step (1-9)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps if outputs already exist")
    parser.add_argument("--audit-only", action="store_true", help="Audit outputs without executing scripts")

    args = parser.parse_args()

    run_pipeline(
        from_step=args.from_step,
        to_step=args.to_step,
        skip_existing=args.skip_existing,
        audit_only=args.audit_only
    )


if __name__ == "__main__":
    main()
