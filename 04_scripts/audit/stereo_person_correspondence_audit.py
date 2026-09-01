"""
Stereo Person Correspondence Audit (T5.E.2 - Priority 2)
Validates that CAM01 and CAM02 selected the SAME physical person for each capture.

Criteria:
  A. Seating ROI: bbox center must be within plausible seated region
  B. Person Size Consistency: bbox area ratio C2/C1 within expected range for stereo pair
  C. Pose Geometry Plausibility: normalized head position, torso proportions cross-view
  D. Stereo Correspondence Evidence: if triangulated, reprojection error as secondary signal

Decision Logic:
  SAME_PERSON         - both cameras clearly tracking the seated target
  LIKELY_SAME         - minor geometric discrepancy but consistent identity indicators
  SUSPICIOUS          - possible mismatch, needs further investigation
  WRONG_PERSON        - strong evidence that cameras track different people

Output:
  07_results/private_audit/stereo_person_correspondence_audit.csv
"""
import sys
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
RESULTS_DIR = PROJECT_ROOT / "07_results" / "private_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_correspondence_audit():
    print("=" * 80)
    print("  STEREO PERSON CORRESPONDENCE AUDIT (851 CAPTURES)")
    print("=" * 80)

    df_cap = pd.read_csv(META_DIR / "captures.csv")
    df_img = pd.read_csv(META_DIR / "images.csv")

    records = []
    
    same_count = 0
    likely_count = 0
    suspicious_count = 0
    wrong_count = 0
    no_target_count = 0

    for idx, row in df_cap.iterrows():
        cap_id = row["capture_id"]
        sub_id = row["subject_id"]
        posture = row["primary_posture"]
        cal_id = row["calibration_id"]
        
        img_c1 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM01")].iloc[0]["image_id"]
        img_c2 = df_img[(df_img["capture_id"] == cap_id) & (df_img["camera_id"] == "CAM02")].iloc[0]["image_id"]
        
        f1 = SELECTED_DIR / f"{img_c1}_selected_person.json"
        f2 = SELECTED_DIR / f"{img_c2}_selected_person.json"
        
        if not f1.exists() or not f2.exists():
            records.append({
                "capture_id": cap_id, "subject_id": sub_id, "posture": posture,
                "calibration_id": cal_id,
                "correspondence": "NO_DATA", "flags": "Missing annotation files",
                "c1_area_ratio": None, "c2_area_ratio": None, "area_ratio_c2_c1": None,
                "c1_nose_y_rel": None, "c2_nose_y_rel": None,
                "c1_sh_span_norm": None, "c2_sh_span_norm": None,
                "c1_torso_v": None, "c2_torso_v": None,
            })
            no_target_count += 1
            continue
        
        with open(f1, "r") as fp: d1 = json.load(fp)
        with open(f2, "r") as fp: d2 = json.load(fp)
        
        has_t1 = d1.get("has_target", False)
        has_t2 = d2.get("has_target", False)
        
        if not has_t1 or not has_t2:
            records.append({
                "capture_id": cap_id, "subject_id": sub_id, "posture": posture,
                "calibration_id": cal_id,
                "correspondence": "NO_TARGET",
                "flags": f"C1_target={has_t1}, C2_target={has_t2}",
                "c1_area_ratio": None, "c2_area_ratio": None, "area_ratio_c2_c1": None,
                "c1_nose_y_rel": None, "c2_nose_y_rel": None,
                "c1_sh_span_norm": None, "c2_sh_span_norm": None,
                "c1_torso_v": None, "c2_torso_v": None,
            })
            no_target_count += 1
            continue
        
        # Extract geometric features
        b1 = d1["bbox"]
        b2 = d2["bbox"]
        kp1 = np.array(d1["keypoints"])
        kp2 = np.array(d2["keypoints"])
        conf1 = np.array(d1["confidences"])
        conf2 = np.array(d2["confidences"])
        
        w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
        w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
        cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
        cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
        area1 = (w1 * h1) / (1920 * 1080)
        area2 = (w2 * h2) / (1920 * 1080)
        area_ratio = area2 / max(area1, 0.001)
        
        # A. Seating ROI Check
        # Person should be roughly centered and occupying a reasonable portion of frame
        c1_centered = 0.2 < (cx1 / 1920.0) < 0.8
        c2_centered = 0.2 < (cx2 / 1920.0) < 0.9  # lateral can be slightly off-center
        
        # B. Relative nose position (normalized to bbox height)
        nose_y_rel_1 = (kp1[0, 1] - b1[1]) / max(h1, 1) if conf1[0] > 0.25 else None
        nose_y_rel_2 = (kp2[0, 1] - b2[1]) / max(h2, 1) if conf2[0] > 0.25 else None
        
        # C. Shoulder span normalized by bbox width
        # Frontal: should be ~40-70% of bbox width (wide span)
        # Lateral: should be ~5-25% of bbox width (foreshortened)
        sh_span_1 = abs(kp1[5, 0] - kp1[6, 0]) / max(w1, 1) if (conf1[5] > 0.25 and conf1[6] > 0.25) else None
        sh_span_2 = abs(kp2[5, 0] - kp2[6, 0]) / max(w2, 1) if (conf2[5] > 0.25 and conf2[6] > 0.25) else None
        
        # D. Vertical torso extent (nose to mid-hip in pixels)
        if conf1[0] > 0.25 and conf1[11] > 0.25:
            torso_v_1 = abs(kp1[11, 1] - kp1[0, 1])
        else:
            torso_v_1 = None
        if conf2[0] > 0.25 and conf2[11] > 0.25:
            torso_v_2 = abs(kp2[11, 1] - kp2[0, 1])
        else:
            torso_v_2 = None
        
        # Score-based decision
        flags = []
        suspicion_score = 0
        
        # Check 1: Area ratio between cameras
        # For stereo pair at ~1-2.5m baseline, lateral view typically produces 
        # 1.2x - 2.0x area due to closer distance / wider field
        if area_ratio > 2.5 or area_ratio < 0.4:
            flags.append(f"AREA_RATIO_EXTREME({area_ratio:.2f})")
            suspicion_score += 2
        elif area_ratio > 2.0 or area_ratio < 0.5:
            flags.append(f"AREA_RATIO_HIGH({area_ratio:.2f})")
            suspicion_score += 1
        
        # Check 2: Nose relative position should be similar (top portion of bbox)
        # Same person sitting: nose is at ~10-25% from top of bbox
        if nose_y_rel_1 is not None and nose_y_rel_2 is not None:
            nose_diff = abs(nose_y_rel_1 - nose_y_rel_2)
            if nose_diff > 0.20:
                flags.append(f"NOSE_POS_MISMATCH(d={nose_diff:.3f})")
                suspicion_score += 2
            elif nose_diff > 0.10:
                flags.append(f"NOSE_POS_DIVERGENT(d={nose_diff:.3f})")
                suspicion_score += 1
        
        # Check 3: Shoulder span ratio
        # Frontal cam should have wide span (~0.35-0.70), lateral should have narrow (~0.01-0.25)
        # If BOTH cameras show wide spans, one might be tracking a different person frontally
        if sh_span_1 is not None and sh_span_2 is not None:
            if sh_span_1 > 0.30 and sh_span_2 > 0.30:
                # Both cameras show frontal-like shoulder span -> possibly different people
                flags.append(f"BOTH_FRONTAL_SPAN(c1={sh_span_1:.3f},c2={sh_span_2:.3f})")
                suspicion_score += 3
            elif sh_span_2 > 0.25:
                # Lateral camera shows too-wide span
                flags.append(f"LATERAL_WIDE_SPAN(c2={sh_span_2:.3f})")
                suspicion_score += 1
        
        # Check 4: Vertical torso proportions  
        # If one camera sees a much taller/shorter torso, it might be a different person
        if torso_v_1 is not None and torso_v_2 is not None:
            v_ratio = torso_v_2 / max(torso_v_1, 1)
            if v_ratio > 2.0 or v_ratio < 0.5:
                flags.append(f"TORSO_V_EXTREME(ratio={v_ratio:.2f})")
                suspicion_score += 2
        
        # Check 5: Centering
        if not c1_centered:
            flags.append("C1_OFF_CENTER")
            suspicion_score += 1
        if not c2_centered:
            flags.append("C2_OFF_CENTER")
            suspicion_score += 1
        
        # Decision
        if suspicion_score == 0:
            correspondence = "SAME_PERSON"
            same_count += 1
        elif suspicion_score <= 2:
            correspondence = "LIKELY_SAME"
            likely_count += 1
        elif suspicion_score <= 4:
            correspondence = "SUSPICIOUS"
            suspicious_count += 1
        else:
            correspondence = "WRONG_PERSON"
            wrong_count += 1
        
        records.append({
            "capture_id": cap_id,
            "subject_id": sub_id,
            "posture": posture,
            "calibration_id": cal_id,
            "correspondence": correspondence,
            "flags": "; ".join(flags) if flags else "CLEAN",
            "suspicion_score": suspicion_score,
            "c1_area_ratio": round(area1, 4),
            "c2_area_ratio": round(area2, 4),
            "area_ratio_c2_c1": round(area_ratio, 3),
            "c1_nose_y_rel": round(nose_y_rel_1, 4) if nose_y_rel_1 is not None else None,
            "c2_nose_y_rel": round(nose_y_rel_2, 4) if nose_y_rel_2 is not None else None,
            "c1_sh_span_norm": round(sh_span_1, 4) if sh_span_1 is not None else None,
            "c2_sh_span_norm": round(sh_span_2, 4) if sh_span_2 is not None else None,
            "c1_torso_v": round(torso_v_1, 1) if torso_v_1 is not None else None,
            "c2_torso_v": round(torso_v_2, 1) if torso_v_2 is not None else None,
        })

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "stereo_person_correspondence_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

    total = len(df_cap)
    print("\n" + "=" * 80)
    print("        RINGKASAN AUDIT KORESPONDENSI ORANG STEREO (851 CAPTURES)")
    print("=" * 80)
    print(f"  SAME_PERSON   : {same_count} ({same_count/total*100:.2f}%)")
    print(f"  LIKELY_SAME   : {likely_count} ({likely_count/total*100:.2f}%)")
    print(f"  SUSPICIOUS    : {suspicious_count} ({suspicious_count/total*100:.2f}%)")
    print(f"  WRONG_PERSON  : {wrong_count} ({wrong_count/total*100:.2f}%)")
    print(f"  NO_TARGET     : {no_target_count} ({no_target_count/total*100:.2f}%)")
    print("=" * 80)

    print("\nBreakdown SUSPICIOUS + WRONG_PERSON per calibration rig:")
    flagged = df[df["correspondence"].isin(["SUSPICIOUS", "WRONG_PERSON"])]
    if len(flagged) > 0:
        print(flagged.groupby(["calibration_id", "correspondence"]).size().unstack(fill_value=0))
    else:
        print("  None!")

    print("\nBreakdown SUSPICIOUS + WRONG_PERSON per posture:")
    if len(flagged) > 0:
        print(flagged.groupby(["posture", "correspondence"]).size().unstack(fill_value=0))

    print("\nAll SUSPICIOUS + WRONG_PERSON captures:")
    if len(flagged) > 0:
        for _, r in flagged.iterrows():
            print(f"  {r['capture_id']} | {r['subject_id']} | {r['posture']:16s} | {r['calibration_id']} | {r['correspondence']:13s} | {r['flags']}")

    return df


if __name__ == "__main__":
    run_correspondence_audit()
