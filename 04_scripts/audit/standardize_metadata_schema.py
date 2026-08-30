"""
Standardize Metadata Schema across captures.csv, images.csv, and chairs.csv.

Applies:
1. Addition of 'view_role' and 'lateral_side' to images.csv.
2. Addition of 'chair_id' and 'lateral_side' to captures.csv.
3. Creation of chairs.csv metadata specification.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"

def standardize():
    print("Standardizing metadata schema...")
    
    # 1. Update captures.csv
    captures_csv = META_DIR / "captures.csv"
    if captures_csv.exists():
        df_cap = pd.read_csv(captures_csv)
        if "chair_id" not in df_cap.columns:
            df_cap["chair_id"] = "CHR_001"
        if "lateral_side" not in df_cap.columns:
            # S001-S004 left, S005-S006 right
            df_cap["lateral_side"] = df_cap["subject_id"].apply(
                lambda s: "left" if s in ["S001", "S002", "S003", "S004"] else "right"
            )
        df_cap.to_csv(captures_csv, index=False)
        print(f"[OK] captures.csv updated with columns {list(df_cap.columns)}")

    # 2. Update images.csv
    images_csv = META_DIR / "images.csv"
    if images_csv.exists():
        df_img = pd.read_csv(images_csv)
        if "view_role" not in df_img.columns:
            df_img["view_role"] = df_img["camera_id"].apply(
                lambda c: "frontal" if c == "CAM01" else "lateral"
            )
        if "lateral_side" not in df_img.columns:
            def assign_lat(row):
                if row["camera_id"] == "CAM01":
                    return "none"
                # extract subject from image_id
                sub = str(row["image_id"]).split("_")[0]
                return "left" if sub in ["S001", "S002", "S003", "S004"] else "right"
            df_img["lateral_side"] = df_img.apply(assign_lat, axis=1)
        df_img.to_csv(images_csv, index=False)
        print(f"[OK] images.csv updated with columns {list(df_img.columns)}")

    # 3. Create chairs.csv
    chairs_csv = META_DIR / "chairs.csv"
    chair_rows = [{
        "chair_id": "CHR_001",
        "chair_type": "armless_standard",
        "seat_height_cm": 45.0,
        "seat_depth_cm": 42.0,
        "seat_width_cm": 42.0,
        "backrest_height_cm": 48.0,
        "has_backrest": "true",
        "has_armrests": "false",
        "material": "fabric_cushion_steel_frame",
        "notes": "Standard lab armless chair used for private posture dataset acquisition"
    }]
    pd.DataFrame(chair_rows).to_csv(chairs_csv, index=False)
    print(f"[OK] chairs.csv created at {chairs_csv.name}")

if __name__ == "__main__":
    standardize()
