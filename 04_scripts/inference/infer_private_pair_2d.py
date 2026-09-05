"""
infer_private_pair_2d.py — 2D Multi-View Posture Inference on Image Pair
Performs YOLOv8-pose estimation, target person selection, Reject Gate validation,
CAM02 canonicalization, 36-feature extraction, and deployment XGBoost prediction.

Usage:
  python 04_scripts/inference/infer_private_pair_2d.py \
      --cam01 path/to/front.jpg \
      --cam02 path/to/side.jpg \
      --lateral-side right
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

# Ensure UTF-8 stdout
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "04_scripts" / "inference"))
from private_inference_common import infer_pair_2d, MAIN_CLASSES


def main():
    parser = argparse.ArgumentParser(description="2D Multi-View Posture Inference on Image Pair")
    parser.add_argument("--cam01", type=str, required=True, help="Path to CAM01 (Frontal) image")
    parser.add_argument("--cam02", type=str, required=True, help="Path to CAM02 (Lateral) image")
    parser.add_argument("--lateral-side", type=str, default="right", choices=["left", "right"], help="Side of CAM02 relative to subject")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON prediction output")

    args = parser.parse_args()

    p1 = Path(args.cam01)
    p2 = Path(args.cam02)

    if not p1.exists():
        print(f"Error: CAM01 image not found at {p1}")
        sys.exit(1)
    if not p2.exists():
        print(f"Error: CAM02 image not found at {p2}")
        sys.exit(1)

    img1 = cv2.imread(str(p1))
    img2 = cv2.imread(str(p2))

    if img1 is None or img2 is None:
        print("Error: Failed to read one or both input images.")
        sys.exit(1)

    print("=" * 70)
    print("  2D MULTI-VIEW POSTURE INFERENCE (6-CLASS XGBOOST)")
    print("=" * 70)
    print(f"CAM01 (Frontal) : {p1.name} ({img1.shape[1]}x{img1.shape[0]})")
    print(f"CAM02 (Lateral) : {p2.name} ({img2.shape[1]}x{img2.shape[0]})")
    print(f"Lateral Side    : {args.lateral_side}")
    print("-" * 70)

    result = infer_pair_2d(img1, img2, lateral_side=args.lateral_side)

    if result["status"] == "REJECTED":
        print(f"STATUS       : REJECTED")
        print(f"REASON       : {result['reason']}")
        print(f"PREDICTION   : REJECT (Invalid Input / Low Confidence Keypoints)")
    else:
        pred = result["prediction"]
        conf = result["confidence"]
        print(f"STATUS       : VALID")
        print(f"PREDICTION   : {pred.upper()}")
        print(f"CONFIDENCE   : {conf:.4f} ({conf*100:.1f}%)")
        print("\nClass Probabilities:")
        for cls_name, prob in result["probabilities"].items():
            bar = "█" * int(prob * 25)
            marker = " <-- PREDICTED" if cls_name == pred else ""
            print(f"  {cls_name:<18} : {prob:.4f} ({prob*100:5.1f}%) {bar}{marker}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)
        print(f"\n[SAVED] Output JSON: {out_p}")


if __name__ == "__main__":
    main()
