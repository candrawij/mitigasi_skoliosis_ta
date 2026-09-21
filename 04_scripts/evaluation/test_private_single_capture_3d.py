"""
3D-14 -- Test Single Capture 3D Inference
==========================================
Verifies that the full pipeline from JSON keypoints to prediction works.
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\.Candra\Project\TA")
KP_DIR    = ROOT / "02_data/private_annotations/keypoints_3d"
FEAT3D    = ROOT / "02_data/private_processed/features/private_features_3d.csv"
MODEL_DIR = ROOT / "06_models/keypoint_3d/private_final"

sys.path.append(str(ROOT / "04_scripts/preprocessing"))
from private_feature_common import extract_3d_features, FEATURE_NAMES_3D

with open(MODEL_DIR / "class_map.json") as f:
    cmap = json.load(f)
ID_TO_CLASS = {int(k): v for k, v in cmap["id_to_class"].items()}

with open(MODEL_DIR / "xgboost_3d.pkl", "rb") as f:
    pipeline = pickle.load(f)

warnings.filterwarnings("ignore")

def predict_from_json(capture_id):
    fp = KP_DIR / f"{capture_id}_3d_keypoints.json"
    with open(fp) as f:
        d = json.load(f)
    kpts = np.array(d["keypoints_3d_m"], dtype=float)
    feat_dict, ok, msg = extract_3d_features(kpts)
    if not ok:
        return None, None, f"FEATURE_EXTRACTION_FAILED: {msg}"
    feat_vec = np.array([feat_dict[fn] for fn in FEATURE_NAMES_3D], dtype=np.float64).reshape(1, -1)
    y_pred = int(pipeline.predict(feat_vec)[0])
    y_prob = pipeline.predict_proba(feat_vec)[0]
    return ID_TO_CLASS[y_pred], float(y_prob[y_pred]), "OK"

def main():
    print("=" * 60)
    print("3D-14 -- Test Single Capture 3D Inference")
    print("=" * 60)

    feat3d = pd.read_csv(FEAT3D)
    usable = feat3d[feat3d["status_3d"]=="USABLE"]

    # Sample one from each class
    classes = sorted(usable["label"].unique())
    all_correct = 0
    all_total = 0

    for cls in classes:
        samples = usable[usable["label"]==cls].head(3)
        for _, row in samples.iterrows():
            cap_id = row["capture_id"]
            true_label = row["label"]
            pred_label, conf, msg = predict_from_json(cap_id)

            if pred_label is None:
                result = "FAIL"
                print(f"  {cap_id} [{true_label}] -> PIPELINE ERROR: {msg}")
            else:
                correct = pred_label == true_label
                result = "CORRECT" if correct else "WRONG"
                if correct:
                    all_correct += 1
                print(f"  {cap_id} [{true_label}] -> pred={pred_label} conf={conf:.3f} [{result}]")
            all_total += 1

    print(f"\nResult: {all_correct}/{all_total} correct ({all_correct/all_total*100:.1f}%)")
    print("[DONE]")

if __name__ == "__main__":
    main()
