#!/usr/bin/env python3
"""
Verifikasi semantik dataset YOLO-Pose:
1. Membuat overlay keypoint bernomor 0..K-1.
2. Mengelompokkan file berdasarkan sumber klip dari nama file.
3. Mendeteksi sumber klip yang tersebar pada split berbeda.
4. Meringkas distribusi kelas per sumber klip.

Tidak mengubah dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifikasi keypoint dan sumber klip dataset YOLO-Pose."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_yaml(dataset_dir: Path, explicit: Path | None) -> Path:
    if explicit:
        path = explicit.resolve()
        if path.exists():
            return path
        raise FileNotFoundError(path)

    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        path = dataset_dir / name
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("data.yaml tidak ditemukan.")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("YAML tidak valid.")
    return data


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {i: str(value) for i, value in enumerate(raw)}
    if isinstance(raw, dict):
        result = {}
        for key, value in raw.items():
            result[int(key)] = str(value)
        return result
    return {}


def resolve_split_dirs(dataset_dir: Path) -> dict[str, Path]:
    candidates = [
        ("train", dataset_dir / "train" / "images"),
        ("valid", dataset_dir / "valid" / "images"),
        ("valid", dataset_dir / "val" / "images"),
        ("test", dataset_dir / "test" / "images"),
        ("train", dataset_dir / "images" / "train"),
        ("valid", dataset_dir / "images" / "valid"),
        ("valid", dataset_dir / "images" / "val"),
        ("test", dataset_dir / "images" / "test"),
    ]
    result: dict[str, Path] = {}
    for split, path in candidates:
        if path.exists() and split not in result:
            result[split] = path.resolve()
    return result


def infer_label_path(image_path: Path, image_dir: Path) -> Path:
    if image_dir.name.lower() == "images":
        label_dir = image_dir.parent / "labels"
        return (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")

    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        idx = len(lowered) - 1 - lowered[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def iter_records(dataset_dir: Path) -> list[dict]:
    records = []
    split_dirs = resolve_split_dirs(dataset_dir)

    for split, image_dir in split_dirs.items():
        for image_path in sorted(image_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = infer_label_path(image_path, image_dir)
            records.append({
                "split": split,
                "image_path": image_path.resolve(),
                "label_path": label_path.resolve(),
                "filename": image_path.name,
            })
    return records


def remove_roboflow_suffix(stem: str) -> str:
    # contoh: abc_jpg.rf.a1b2c3 -> abc_jpg
    return re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem)


def infer_source_clip(filename: str) -> tuple[str, str]:
    """
    Menghasilkan source_clip dan frame_hint dari pola umum Roboflow/video frame.

    Contoh:
    nghiengtrai_mp4-0241_jpg.rf.hash.jpg
      -> nghiengtrai_mp4, 0241

    WIN_20251109_18_42_31_Pro_mp4-0042_jpg.rf.hash.jpg
      -> WIN_20251109_18_42_31_Pro_mp4, 0042
    """
    stem = remove_roboflow_suffix(Path(filename).stem)
    stem = re.sub(r"_jpg$", "", stem, flags=re.IGNORECASE)

    patterns = [
        r"^(.*?_mp4)[-_](\d+)$",
        r"^(.*?)[-_]frame[-_]?(\d+)$",
        r"^(.*?)[-_](\d{3,6})$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)

    return stem, ""


def parse_first_object(
    label_path: Path,
    kpt_count: int,
    kpt_dim: int,
) -> tuple[int, tuple[float, float, float, float], list[list[float]]] | None:
    if not label_path.exists():
        return None

    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    if not lines:
        return None

    tokens = lines[0].split()
    expected = 5 + kpt_count * kpt_dim
    if len(tokens) != expected:
        return None

    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return None

    class_id = int(values[0])
    bbox = tuple(values[1:5])
    flat = values[5:]
    keypoints = [
        flat[index * kpt_dim:(index + 1) * kpt_dim]
        for index in range(kpt_count)
    ]
    return class_id, bbox, keypoints


def draw_indexed_overlay(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    keypoints: list[list[float]],
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size

    bx, by, bw, bh = bbox
    left = (bx - bw / 2) * width
    top = (by - bh / 2) * height
    right = (bx + bw / 2) * width
    bottom = (by + bh / 2) * height
    draw.rectangle((left, top, right, bottom), outline="white", width=max(2, width // 300))

    radius = max(4, min(width, height) // 90)
    for index, point in enumerate(keypoints):
        x, y = point[0], point[1]
        visible = True
        if len(point) >= 3:
            visible = point[2] > 0
        if not visible or (x == 0 and y == 0):
            continue

        px, py = x * width, y * height
        draw.ellipse(
            (px-radius, py-radius, px+radius, py+radius),
            outline="white",
            width=max(2, radius // 2),
        )
        label_box = (px + radius, py - radius - 2)
        draw.text(label_box, str(index), fill="white", stroke_width=2, stroke_fill="black")
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_index_sheets(
    parsed_records: list[dict],
    names: dict[int, str],
    output_dir: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in parsed_records:
        grouped[record["class_id"]].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 300, 220
    text_h = 55
    columns = 3

    for class_id, class_records in sorted(grouped.items()):
        sample = rng.sample(
            class_records,
            min(samples_per_class, len(class_records)),
        )
        rows_count = math.ceil(len(sample) / columns)
        canvas = Image.new(
            "RGB",
            (columns * thumb_w, rows_count * (thumb_h + text_h) + 45),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        class_name = names.get(class_id, f"class_{class_id}")
        draw.text((8, 10), f"{class_id}: {class_name}", fill="black")

        for idx, record in enumerate(sample):
            x_cell = (idx % columns) * thumb_w
            y_cell = 45 + (idx // columns) * (thumb_h + text_h)

            try:
                with Image.open(record["image_path"]) as image:
                    overlay = draw_indexed_overlay(
                        image,
                        record["bbox"],
                        record["keypoints"],
                    )
                    fitted = ImageOps.contain(
                        overlay,
                        (thumb_w - 8, thumb_h - 8),
                        Image.Resampling.LANCZOS,
                    )
                    px = x_cell + (thumb_w - fitted.width) // 2
                    py = y_cell + (thumb_h - fitted.height) // 2
                    canvas.paste(fitted, (px, py))
            except (UnidentifiedImageError, OSError):
                draw.text((x_cell + 10, y_cell + 80), "IMAGE ERROR", fill="black")

            text = (
                f"{record['split']} | {record['source_clip']}\n"
                f"{Path(record['image_path']).name[:36]}"
            )
            draw.multiline_text(
                (x_cell + 4, y_cell + thumb_h + 3),
                text,
                fill="black",
                spacing=2,
            )

        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", class_name)
        canvas.save(
            output_dir / f"class_{class_id}_{safe_name}_indexed.jpg",
            quality=92,
        )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    try:
        yaml_path = find_yaml(dataset_dir, args.yaml)
        yaml_data = load_yaml(yaml_path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    kpt_shape = yaml_data.get("kpt_shape")
    if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) != 2:
        print("ERROR: kpt_shape tidak valid.", file=sys.stderr)
        return 1

    kpt_count, kpt_dim = int(kpt_shape[0]), int(kpt_shape[1])
    names = normalize_names(yaml_data.get("names"))
    records = iter_records(dataset_dir)

    parsed_records: list[dict] = []
    source_rows: list[dict] = []

    for record in records:
        source_clip, frame_hint = infer_source_clip(record["filename"])
        parsed = parse_first_object(
            record["label_path"],
            kpt_count,
            kpt_dim,
        )

        class_id = ""
        if parsed is not None:
            class_id, bbox, keypoints = parsed
            parsed_records.append({
                **record,
                "source_clip": source_clip,
                "frame_hint": frame_hint,
                "class_id": class_id,
                "bbox": bbox,
                "keypoints": keypoints,
            })

        source_rows.append({
            "split": record["split"],
            "filename": record["filename"],
            "image_path": str(record["image_path"]),
            "label_path": str(record["label_path"]),
            "source_clip": source_clip,
            "frame_hint": frame_hint,
            "class_id": class_id,
            "class_name": names.get(class_id, "") if class_id != "" else "",
        })

    write_csv(
        output_dir / "image_source_map.csv",
        source_rows,
        fieldnames=[
            "split", "filename", "image_path", "label_path",
            "source_clip", "frame_hint", "class_id", "class_name",
        ],
    )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        grouped[row["source_clip"]].append(row)

    summary_rows: list[dict] = []
    leakage_rows: list[dict] = []

    for source_clip, rows in sorted(grouped.items()):
        splits = sorted({row["split"] for row in rows})
        classes = sorted({
            row["class_name"] for row in rows if row["class_name"]
        })
        split_counts = Counter(row["split"] for row in rows)
        class_counts = Counter(
            row["class_name"] for row in rows if row["class_name"]
        )

        summary = {
            "source_clip": source_clip,
            "image_count": len(rows),
            "splits": " | ".join(splits),
            "split_count": len(splits),
            "classes": " | ".join(classes),
            "class_count": len(classes),
            "train_count": split_counts["train"],
            "valid_count": split_counts["valid"],
            "test_count": split_counts["test"],
            "class_distribution": " | ".join(
                f"{name}:{count}"
                for name, count in sorted(class_counts.items())
            ),
        }
        summary_rows.append(summary)
        if len(splits) > 1:
            leakage_rows.append(summary)

    fields = [
        "source_clip", "image_count", "splits", "split_count",
        "classes", "class_count", "train_count", "valid_count",
        "test_count", "class_distribution",
    ]
    write_csv(
        output_dir / "source_clip_summary.csv",
        summary_rows,
        fieldnames=fields,
    )
    write_csv(
        output_dir / "source_clip_cross_split.csv",
        leakage_rows,
        fieldnames=fields,
    )

    make_index_sheets(
        parsed_records,
        names,
        output_dir / "indexed_keypoint_sheets",
        args.samples_per_class,
        args.seed,
    )

    total_sources = len(summary_rows)
    cross_split_sources = len(leakage_rows)
    source_sizes = [row["image_count"] for row in summary_rows]
    largest = max(source_sizes) if source_sizes else 0

    markdown = f"""# Verifikasi Semantik Postureexercise

- Total citra: **{len(records)}**
- Total sumber klip terdeteksi dari nama file: **{total_sources}**
- Sumber klip yang muncul pada lebih dari satu split: **{cross_split_sources}**
- Ukuran sumber klip terbesar: **{largest}**
- kpt_shape: **[{kpt_count}, {kpt_dim}]**

## Output

- `image_source_map.csv`
- `source_clip_summary.csv`
- `source_clip_cross_split.csv`
- `indexed_keypoint_sheets/`

## Pemeriksaan manual wajib

1. Tentukan arti indeks keypoint 0–{kpt_count - 1}.
2. Pastikan apakah tujuh titik hanya mencakup wajah dan bahu.
3. Konfirmasi arti kelas `nga_*` dan `nghieng_*`.
4. Periksa apakah satu sumber klip tersebar pada train/valid/test.
5. Jangan memakai horizontal flip sebelum pasangan keypoint dan pertukaran kelas kiri–kanan ditetapkan.
"""
    (output_dir / "semantic_verification_summary.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print("Selesai.")
    print(f"Output: {output_dir}")
    print(f"Source clips: {total_sources}")
    print(f"Cross-split source clips: {cross_split_sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
