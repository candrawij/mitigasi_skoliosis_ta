"""
T5.B.1: Calibration Board Generator.

Generates printable checkerboard patterns with millimeter-accurate dimensions
and scale validation rulers to ensure high-precision camera calibration.

Output:
  - 02_data/private_calibration/patterns/checkerboard_9x6_25mm.png
  - 02_data/private_calibration/patterns/checkerboard_spec.json
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "02_data" / "private_calibration" / "patterns"


def generate_checkerboard(
    cols=9,           # Inner corners horizontal (cols) -> squares = cols + 1 = 10
    rows=6,           # Inner corners vertical (rows) -> squares = rows + 1 = 7
    square_size_mm=25,# Real-world square size in mm
    dpi=300           # Print DPI for exact scale
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate pixels per mm at given DPI
    # 1 inch = 25.4 mm
    px_per_mm = dpi / 25.4
    square_px = int(round(square_size_mm * px_per_mm))
    
    n_sq_x = cols + 1
    n_sq_y = rows + 1
    
    board_w = n_sq_x * square_px
    board_h = n_sq_y * square_px
    
    # Add border/margin (20 mm)
    margin_px = int(round(20 * px_per_mm))
    img_w = board_w + 2 * margin_px
    img_h = board_h + 2 * margin_px + int(round(25 * px_per_mm))  # Extra for caption
    
    # Create white canvas
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    
    # Draw black squares
    for r in range(n_sq_y):
        for c in range(n_sq_x):
            if (r + c) % 2 == 1:  # Alternating pattern
                x0 = margin_px + c * square_px
                y0 = margin_px + r * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                draw.rectangle([x0, y0, x1, y1], fill="black")
                
    # Draw outer border line
    draw.rectangle([margin_px, margin_px, margin_px + board_w, margin_px + board_h], outline="black", width=2)
    
    # Add caption & scale validation text
    caption_y = margin_px + board_h + int(round(5 * px_per_mm))
    caption = (
        f"Checkerboard Pattern: {cols}x{rows} Inner Corners ({n_sq_x}x{n_sq_y} Squares)\n"
        f"Square Size: {square_size_mm} mm | DPI: {dpi} | Print at 100% Scale (Do NOT Fit to Page)\n"
        f"Verify square width with a physical ruler before calibration."
    )
    draw.text((margin_px, caption_y), caption, fill="black")
    
    # Save Image
    out_img_path = OUTPUT_DIR / f"checkerboard_{cols}x{rows}_{square_size_mm}mm.png"
    img.save(out_img_path, dpi=(dpi, dpi))
    print(f"[OK] Saved Checkerboard Pattern: {out_img_path}")
    print(f"     Resolution: {img_w}x{img_h} px ({img_w/px_per_mm:.1f}x{img_h/px_per_mm:.1f} mm)")
    
    # Save Specification JSON
    spec = {
        "pattern_type": "checkerboard",
        "inner_corners_cols": cols,
        "inner_corners_rows": rows,
        "squares_cols": n_sq_x,
        "squares_rows": n_sq_y,
        "square_size_mm": square_size_mm,
        "square_size_m": square_size_mm / 1000.0,
        "dpi": dpi,
        "image_file": str(out_img_path.name),
        "guidelines": "Print at 100% scale on flat cardboard or glass backing to prevent warping."
    }
    spec_path = OUTPUT_DIR / "checkerboard_spec.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"[OK] Saved Calibration Specification: {spec_path}")
    return out_img_path, spec_path


if __name__ == "__main__":
    generate_checkerboard(cols=9, rows=6, square_size_mm=25, dpi=300)
