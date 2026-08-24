"""
Generate keypoint overlay images for visual verification of mapping.

Picks sample images from each class, draws numbered keypoints with
color coding, and saves annotated images for manual inspection.
"""
import os
import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Requires Pillow: pip install Pillow")

DATASET_ROOT = Path(r"D:\.Candra\Project\TA\02_data\raw\Sitting_posture.v17i.yolov8")
OUTPUT_DIR = Path(r"D:\.Candra\Project\TA\07_results\dataset_audit\postureexercise_semantics\keypoint_overlay_verification")

CLASS_NAMES = {0: "nga_phai", 1: "nga_trai", 2: "nghieng_phai", 3: "nghieng_trai", 4: "thang"}

PROPOSED_MAPPING = {
    0: "L_shoulder",
    1: "R_shoulder",
    2: "L_eye",
    3: "L_ear",
    4: "nose",
    5: "R_ear",
    6: "R_eye",
}

# Colors per keypoint (vibrant, distinct)
KP_COLORS = {
    0: (255, 50, 50),     # red - L_shoulder
    1: (50, 50, 255),     # blue - R_shoulder
    2: (0, 255, 0),       # green - L_eye
    3: (255, 165, 0),     # orange - L_ear
    4: (255, 255, 0),     # yellow - nose
    5: (0, 255, 255),     # cyan - R_ear
    6: (255, 0, 255),     # magenta - R_eye
}

SAMPLES_PER_CLASS = 3


def parse_yolo_pose_line(line: str):
    parts = line.strip().split()
    class_id = int(parts[0])
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    keypoints = []
    for i in range(7):
        idx = 5 + i * 3
        kx, ky, vis = float(parts[idx]), float(parts[idx+1]), int(float(parts[idx+2]))
        keypoints.append((kx, ky, vis))
    return class_id, (cx, cy, w, h), keypoints


def draw_keypoints(img: Image.Image, keypoints, bbox):
    """Draw numbered keypoints on image with proposed labels."""
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size
    
    # Try to get a decent font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font
    
    # Draw bounding box
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw/2) * w_img)
    y1 = int((cy - bh/2) * h_img)
    x2 = int((cx + bw/2) * w_img)
    y2 = int((cy + bh/2) * h_img)
    draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=2)
    
    # Draw keypoints
    radius = 6
    for kp_idx, (kx, ky, vis) in enumerate(keypoints):
        if vis == 0:
            continue
        px = int(kx * w_img)
        py = int(ky * h_img)
        color = KP_COLORS[kp_idx]
        
        # Draw circle
        draw.ellipse([px-radius, py-radius, px+radius, py+radius], fill=color, outline=(255,255,255), width=1)
        
        # Draw index number
        label = f"{kp_idx}"
        draw.text((px + radius + 3, py - 8), label, fill=color, font=font)
        
        # Draw proposed name
        name = PROPOSED_MAPPING[kp_idx]
        draw.text((px + radius + 18, py - 8), name, fill=(255, 255, 255), font=font_small)
    
    return img


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect samples per class from train split (most data there)
    class_samples = {cls_id: [] for cls_id in CLASS_NAMES}
    
    for split in ["train", "valid", "test"]:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        if not img_dir.exists():
            continue
        
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            with open(lbl_file) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue
            
            class_id, bbox, keypoints = parse_yolo_pose_line(lines[0])
            
            # Find matching image
            stem = lbl_file.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path and class_id in class_samples:
                class_samples[class_id].append((img_path, bbox, keypoints, split))
    
    # Generate overlay images
    total_generated = 0
    
    for cls_id, cls_name in CLASS_NAMES.items():
        samples = class_samples[cls_id]
        if not samples:
            print(f"  No samples for class {cls_id} ({cls_name})")
            continue
        
        # Pick diverse samples
        random.seed(42)
        selected = random.sample(samples, min(SAMPLES_PER_CLASS, len(samples)))
        
        for i, (img_path, bbox, keypoints, split) in enumerate(selected):
            img = Image.open(img_path).convert("RGB")
            img = draw_keypoints(img, keypoints, bbox)
            
            # Add class info as text at top
            draw = ImageDraw.Draw(img)
            try:
                font_title = ImageFont.truetype("arial.ttf", 18)
            except (OSError, IOError):
                font_title = ImageFont.load_default()
            
            title = f"Class {cls_id}: {cls_name} | Split: {split} | {img_path.name}"
            draw.text((10, 10), title, fill=(255, 255, 0), font=font_title)
            
            out_name = f"verify_{cls_name}_sample{i+1}.jpg"
            img.save(OUTPUT_DIR / out_name, quality=90)
            total_generated += 1
            print(f"  Saved: {out_name}")
    
    # Also generate a legend image
    legend_img = Image.new("RGB", (500, 300), (30, 30, 30))
    draw = ImageDraw.Draw(legend_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    draw.text((20, 10), "PROPOSED KEYPOINT MAPPING", fill=(255, 255, 255), font=font)
    draw.text((20, 35), "(Verify against overlay images)", fill=(180, 180, 180), font=font)
    
    y = 70
    for kp_idx in range(7):
        color = KP_COLORS[kp_idx]
        name = PROPOSED_MAPPING[kp_idx]
        draw.ellipse([20, y, 36, y+16], fill=color, outline=(255,255,255))
        draw.text((45, y), f"KP {kp_idx}: {name}", fill=color, font=font)
        y += 30
    
    legend_img.save(OUTPUT_DIR / "00_legend.jpg", quality=90)
    
    print(f"\nTotal overlay images generated: {total_generated}")
    print(f"Legend saved: 00_legend.jpg")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\n--- VERIFICATION INSTRUCTIONS ---")
    print("1. Open the overlay images")
    print("2. Check if numbered dots match the proposed body parts")
    print("3. Pay special attention to:")
    print("   - KP 0 (red) should be on LEFT shoulder")
    print("   - KP 1 (blue) should be on RIGHT shoulder")
    print("   - KP 4 (yellow) should be on nose")
    print("   - KP 2 (green) vs KP 3 (orange): eye vs ear on LEFT side")
    print("   - KP 6 (magenta) vs KP 5 (cyan): eye vs ear on RIGHT side")


if __name__ == "__main__":
    main()
