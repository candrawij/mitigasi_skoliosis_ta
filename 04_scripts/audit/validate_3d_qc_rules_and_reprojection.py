"""
T5.E.2 FINAL: Stereo 3D QC Pipeline with Correspondence Check + Core/Masked Error Separation

Pipeline Structure:
  CAPTURE
    -> Person Selection (CAM01 & CAM02)
    -> Same-Person Correspondence Check
    -> Keypoint QC
    -> Stereo Triangulation (resolution-scaled)
    -> Reprojection QC (core vs masked/peripheral)
    -> Anatomical Sanity Check
    -> Decision: INCLUDE_3D_FULL / INCLUDE_3D_WITH_MASKING / EXCLUDE_3D_*

Output Columns:
  - core_reproj_error_640p_px: mean error on joints that passed quality threshold (used for geometry)
  - masked_joint_max_error_px: max error on outlier joints (diagnostic, preserved for audit)
  - correspondence_status: SAME_PERSON / LIKELY_SAME / SUSPICIOUS / WRONG_PERSON / NO_TARGET

Output:
  07_results/private_audit/private_3d_qc_final.csv
"""
import sys
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"
SELECTED_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "selected_person"
CALIB_DIR = PROJECT_ROOT / "02_data" / "private_calibration"
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"
ANNOT_3D_DIR = PROJECT_ROOT / "02_data" / "private_annotations" / "keypoints_3d"
ANNOT_3D_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_RIGS = ["CAL_001", "CAL_004", "CAL_005", "CAL_006", "CAL_008", "CAL_009", "CAL_010"]
DEGENERATE_RIGS = ["CAL_006", "CAL_010"]

# Joint-level error threshold for masking (in 640p space)
JOINT_CLEAN_THRESHOLD = 65.0


def check_correspondence(d1, d2):
    """Check if CAM01 and CAM02 selected the same person."""
    if not d1.get("has_target") or not d2.get("has_target"):
        return "NO_TARGET", 0

    b1, b2 = d1["bbox"], d2["bbox"]
    kp1 = np.array(d1["keypoints"])
    kp2 = np.array(d2["keypoints"])
    conf1 = np.array(d1["confidences"])
    conf2 = np.array(d2["confidences"])

    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    area1 = (w1 * h1) / (1920 * 1080)
    area2 = (w2 * h2) / (1920 * 1080)
    area_ratio = area2 / max(area1, 0.001)

    score = 0

    # Area ratio
    if area_ratio > 2.5 or area_ratio < 0.4:
        score += 2
    elif area_ratio > 2.0 or area_ratio < 0.5:
        score += 1

    # Nose relative position
    nose_y_rel_1 = (kp1[0, 1] - b1[1]) / max(h1, 1) if conf1[0] > 0.25 else None
    nose_y_rel_2 = (kp2[0, 1] - b2[1]) / max(h2, 1) if conf2[0] > 0.25 else None
    if nose_y_rel_1 is not None and nose_y_rel_2 is not None:
        if abs(nose_y_rel_1 - nose_y_rel_2) > 0.20:
            score += 2
        elif abs(nose_y_rel_1 - nose_y_rel_2) > 0.10:
            score += 1

    # Both-frontal shoulder span
    sh1 = abs(kp1[5, 0] - kp1[6, 0]) / max(w1, 1) if (conf1[5] > 0.25 and conf1[6] > 0.25) else None
    sh2 = abs(kp2[5, 0] - kp2[6, 0]) / max(w2, 1) if (conf2[5] > 0.25 and conf2[6] > 0.25) else None
    if sh1 is not None and sh2 is not None:
        if sh1 > 0.30 and sh2 > 0.30:
            score += 3

    if score == 0:
        return "SAME_PERSON", score
    elif score <= 2:
        return "LIKELY_SAME", score
    elif score <= 4:
        return "SUSPICIOUS", score
    else:
        return "WRONG_PERSON", score


def run_final_3d_qc():
    print("=" * 80)
    print("  T5.E.2 FINAL: 3D QC WITH CORRESPONDENCE + CORE/MASKED SEPARATION")
    print("=" * 80)

    df_cap = pd.read_csv(META_DIR / "captures.csv")
    df_img = pd.read_csv(META_DIR / "images.csv")

    calib_cache = {}
    for c_id in ACTIVE_RIGS:
        fpath = CALIB_DIR / "stereo" / (c_id + "_stereo.json")
        if fpath.exists():
            with open(fpath, "r") as fp:
                calib_cache[c_id] = json.load(fp)

    scale_x = 640.0 / 1920.0
    scale_y = 480.0 / 1080.0

    records = []
    counts = {"INCLUDE_3D_FULL": 0, "INCLUDE_3D_WITH_MASKING": 0}
    exclude_reasons = {}

    for idx, row in df_cap.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        posture = row["primary_posture"]
        cal_id = row["calibration_id"]

        img_c1 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")].iloc[0]["image_id"]
        img_c2 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")].iloc[0]["image_id"]

        f1 = SELECTED_DIR / (img_c1 + "_selected_person.json")
        f2 = SELECTED_DIR / (img_c2 + "_selected_person.json")

        # Default record
        rec = {
            "capture_id": cap_id, "subject_id": sub_id, "posture": posture,
            "calibration_id": cal_id,
            "correspondence_status": None, "correspondence_score": None,
            "status_3d": None, "decision": None,
            "valid_3d_joints_core": 0, "valid_3d_joints_total": 0,
            "core_reproj_error_640p_px": None, "core_reproj_error_1080p_px": None,
            "masked_joint_max_error_px": None,
            "shoulder_width_m": None, "torso_length_m": None, "depth_z_m": None,
            "reason": None,
        }

        if not f1.exists() or not f2.exists() or cal_id not in calib_cache:
            rec.update(status_3d="FAIL", decision="EXCLUDE_3D", reason="Missing annotation or calibration")
            exclude_reasons["Missing files"] = exclude_reasons.get("Missing files", 0) + 1
            records.append(rec)
            continue

        with open(f1, "r") as fp: d1 = json.load(fp)
        with open(f2, "r") as fp: d2 = json.load(fp)

        # STEP 1: Correspondence Check
        corr_status, corr_score = check_correspondence(d1, d2)
        rec["correspondence_status"] = corr_status
        rec["correspondence_score"] = corr_score

        if corr_status == "NO_TARGET":
            is_reject = (posture == "reject")
            rec.update(
                status_3d="PASS" if is_reject else "FAIL",
                decision="EXCLUDE_3D",
                reason="Clean negative sample (empty chair)" if is_reject else "Target lost"
            )
            exclude_reasons["No target"] = exclude_reasons.get("No target", 0) + 1
            records.append(rec)
            continue

        if corr_status == "WRONG_PERSON":
            rec.update(status_3d="FAIL", decision="EXCLUDE_3D_WRONG_PERSON",
                       reason="Strong evidence of cross-camera person mismatch (score={})".format(corr_score))
            exclude_reasons["Wrong person"] = exclude_reasons.get("Wrong person", 0) + 1
            records.append(rec)
            continue

        # STEP 2: Check for Degenerate Rig
        if cal_id in DEGENERATE_RIGS:
            baseline = calib_cache[cal_id]["stereo_quality"]["baseline_distance_m"]
            rec.update(status_3d="FAIL", decision="EXCLUDE_3D_DEGENERATE_RIG",
                       reason="Rig {} excluded (Baseline={:.1f}m)".format(cal_id, baseline))
            exclude_reasons["Degenerate rig " + cal_id] = exclude_reasons.get("Degenerate rig " + cal_id, 0) + 1
            records.append(rec)
            continue

        # STEP 3: Triangulation with Resolution Scaling
        pts1_1080 = np.array(d1["keypoints"], dtype=np.float64)
        pts2_1080 = np.array(d2["keypoints"], dtype=np.float64)
        conf1 = np.array(d1["confidences"], dtype=np.float64)
        conf2 = np.array(d2["confidences"], dtype=np.float64)

        pts1_640 = pts1_1080 * [scale_x, scale_y]
        pts2_640 = pts2_1080 * [scale_x, scale_y]

        cal = calib_cache[cal_id]
        K1 = np.array(cal["intrinsics_refined"]["K1"])
        D1 = np.array(cal["intrinsics_refined"]["D1"])
        K2 = np.array(cal["intrinsics_refined"]["K2"])
        D2 = np.array(cal["intrinsics_refined"]["D2"])
        R1 = np.array(cal["rectification"]["R1"])
        R2 = np.array(cal["rectification"]["R2"])
        P1 = np.array(cal["rectification"]["P1"])
        P2 = np.array(cal["rectification"]["P2"])

        u1 = cv2.undistortPoints(pts1_640.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
        u2 = cv2.undistortPoints(pts2_640.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
        p4 = cv2.triangulatePoints(P1, P2, u1.T, u2.T)
        p3_rect = (p4[:3] / p4[3]).T

        X1 = (R1.T @ p3_rect.T).T
        rep1, _ = cv2.projectPoints(X1, np.zeros(3), np.zeros(3), K1, D1)
        rep1 = rep1.reshape(-1, 2)

        R_st = np.array(cal["extrinsics"]["rotation_matrix_R"])
        t_raw = cal["extrinsics"].get("translation_vector_T", cal["extrinsics"].get("translation_vector_T_m"))
        T_st = np.array(t_raw).reshape(3, 1)
        X2 = (R_st @ X1.T + T_st).T
        rep2, _ = cv2.projectPoints(X2, np.zeros(3), np.zeros(3), K2, D2)
        rep2 = rep2.reshape(-1, 2)

        # STEP 4: Joint-Level Quality Classification
        clean_3d = [None] * 17
        core_errors = []
        all_errors = []
        masked_max = 0.0
        valid_core = 0
        valid_total = 0

        for j in range(17):
            if (conf1[j] < 0.25 or conf2[j] < 0.25 or
                    np.isinf(p3_rect[j]).any() or np.isnan(p3_rect[j]).any() or
                    p3_rect[j, 2] < 0.3):
                clean_3d[j] = [float("nan")] * 3
                continue

            e1 = np.linalg.norm(rep1[j] - pts1_640[j])
            e2 = np.linalg.norm(rep2[j] - pts2_640[j])
            mean_e = (e1 + e2) / 2.0
            all_errors.append(mean_e)

            if mean_e <= JOINT_CLEAN_THRESHOLD:
                clean_3d[j] = p3_rect[j].tolist()
                core_errors.append(mean_e)
                valid_core += 1
                valid_total += 1
            else:
                clean_3d[j] = [float("nan")] * 3
                masked_max = max(masked_max, mean_e)
                valid_total += 1  # detected but masked

        core_mean = float(np.mean(core_errors)) if core_errors else float("nan")
        core_mean_1080 = core_mean / scale_x if not np.isnan(core_mean) else float("nan")

        # STEP 5: Anatomical Sanity
        def joint_clean(j):
            return clean_3d[j] is not None and not np.isnan(clean_3d[j][0])

        if joint_clean(5) and joint_clean(6):
            sh_width = float(np.linalg.norm(np.array(clean_3d[5]) - np.array(clean_3d[6])))
        else:
            sh_width = float("nan")

        if joint_clean(5) and joint_clean(6) and joint_clean(11) and joint_clean(12):
            mid_sh = (np.array(clean_3d[5]) + np.array(clean_3d[6])) / 2.0
            mid_hip = (np.array(clean_3d[11]) + np.array(clean_3d[12])) / 2.0
            torso_len = float(np.linalg.norm(mid_sh - mid_hip))
        else:
            torso_len = float("nan")

        valid_z = [clean_3d[j][2] for j in range(17) if joint_clean(j) and 0.4 < clean_3d[j][2] < 8.0]
        mean_z = float(np.mean(valid_z)) if valid_z else float("nan")

        # Save clean 3D keypoints
        annot_file = ANNOT_3D_DIR / (cap_id + "_3d_keypoints.json")
        with open(annot_file, "w") as fp:
            json.dump({
                "capture_id": cap_id, "calibration_id": cal_id,
                "valid_core_joints": valid_core,
                "keypoints_3d_m": clean_3d,
            }, fp, indent=2)

        # STEP 6: Decision
        depth_ok = 0.50 <= mean_z <= 3.80 if not np.isnan(mean_z) else False
        sh_ok = np.isnan(sh_width) or (0.15 <= sh_width <= 0.65)
        torso_ok = np.isnan(torso_len) or (0.22 <= torso_len <= 0.85)

        if valid_core >= 8 and depth_ok and sh_ok and torso_ok and core_mean <= 45.0:
            status_3d = "PASS"
            decision = "INCLUDE_3D_FULL"
            reason = "Full 3D: {}/17 core joints, core_err={:.1f}px, Z={:.2f}m".format(valid_core, core_mean, mean_z)
            counts["INCLUDE_3D_FULL"] += 1
        elif valid_core >= 4 and depth_ok:
            status_3d = "REVIEW"
            decision = "INCLUDE_3D_WITH_MASKING"
            reason = "Partial 3D: {}/17 core joints - usable with NaN masking".format(valid_core)
            counts["INCLUDE_3D_WITH_MASKING"] += 1
        elif corr_status == "SUSPICIOUS":
            status_3d = "FAIL"
            decision = "EXCLUDE_3D_SUSPICIOUS_CORRESPONDENCE"
            reason = "Suspicious correspondence (score={}) + insufficient core joints ({})".format(corr_score, valid_core)
            exclude_reasons["Suspicious corr"] = exclude_reasons.get("Suspicious corr", 0) + 1
        else:
            status_3d = "FAIL"
            decision = "EXCLUDE_3D"
            reason = "Insufficient core joints ({}/17) or depth anomaly (Z={:.2f}m)".format(
                valid_core, mean_z if not np.isnan(mean_z) else 0)
            exclude_reasons["Insufficient joints/depth"] = exclude_reasons.get("Insufficient joints/depth", 0) + 1

        rec.update(
            status_3d=status_3d, decision=decision, reason=reason,
            valid_3d_joints_core=valid_core,
            valid_3d_joints_total=valid_total,
            core_reproj_error_640p_px=round(core_mean, 2) if not np.isnan(core_mean) else None,
            core_reproj_error_1080p_px=round(core_mean_1080, 2) if not np.isnan(core_mean_1080) else None,
            masked_joint_max_error_px=round(masked_max, 2) if masked_max > 0 else None,
            shoulder_width_m=round(sh_width, 3) if not np.isnan(sh_width) else None,
            torso_length_m=round(torso_len, 3) if not np.isnan(torso_len) else None,
            depth_z_m=round(mean_z, 3) if not np.isnan(mean_z) else None,
        )
        records.append(rec)

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "private_3d_qc_final.csv"
    df.to_csv(out_csv, index=False)
    print("\n[SAVED] {}".format(out_csv))

    total = len(df_cap)
    inc_full = counts["INCLUDE_3D_FULL"]
    inc_mask = counts["INCLUDE_3D_WITH_MASKING"]
    exc_total = total - inc_full - inc_mask
    print("\n" + "=" * 80)
    print("  RINGKASAN FINAL VALIDASI 3D STEREO (851 CAPTURES)")
    print("=" * 80)
    print("  INCLUDE_3D_FULL          : {} ({:.2f}%)".format(inc_full, inc_full / total * 100))
    print("  INCLUDE_3D_WITH_MASKING  : {} ({:.2f}%)".format(inc_mask, inc_mask / total * 100))
    print("  EXCLUDE (all reasons)    : {} ({:.2f}%)".format(exc_total, exc_total / total * 100))
    print("  ---")
    print("  TOTAL 3D USABLE          : {} ({:.2f}%)".format(inc_full + inc_mask, (inc_full + inc_mask) / total * 100))
    print("=" * 80)

    print("\nExclusion Breakdown:")
    for reason, count in sorted(exclude_reasons.items(), key=lambda x: -x[1]):
        print("  {}: {}".format(reason, count))

    print("\nPer Calibration Rig:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.groupby(["calibration_id", "decision"]).size().unstack(fill_value=0))

    print("\nCorrespondence Status Distribution:")
    print(df["correspondence_status"].value_counts())

    return df


if __name__ == "__main__":
    run_final_3d_qc()
