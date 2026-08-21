#!/usr/bin/env python3
"""
Audit dataset pose estimation berformat Ultralytics/YOLO-Pose.

Pemeriksaan:
- Membaca data.yaml dan kpt_shape.
- Memvalidasi pasangan image-label.
- Memvalidasi class ID, bbox, jumlah keypoint, koordinat, dan visibility.
- Menghitung distribusi split/kelas.
- Menghitung kelengkapan setiap keypoint.
- Mendeteksi exact/near duplicate dan leakage lintas split.
- Membuat contact sheet dengan overlay bbox/keypoint.
- Tidak mengubah atau menghapus dataset.

Contoh:
python audit_yolo_pose_dataset.py ^
  "02_data/raw/postureexercise" ^
  --output-dir "07_results/dataset_audit/postureexercise_initial"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "valid",
    "valid": "valid",
    "validation": "valid",
    "test": "test",
    "testing": "test",
}

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class ImageAudit:
    split: str
    image_path: str
    relative_path: str
    label_path: str
    width: int
    height: int
    file_size_bytes: int
    sha256: str
    dhash64: str
    brightness_mean: float
    edge_variance: float
    is_corrupt: bool
    label_exists: bool
    label_empty: bool
    object_count: int
    valid_object_count: int
    invalid_object_count: int
    error: str


@dataclass
class PoseObject:
    split: str
    image_path: str
    label_path: str
    line_number: int
    class_id: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    keypoints: list[list[float]]
    valid: bool
    issues: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dataset Ultralytics/YOLO-Pose."
    )
    parser.add_argument("dataset_dir", type=Path, help="Root dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pose_dataset_audit_output"),
        help="Folder hasil audit.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Path data.yaml. Jika kosong, dicari otomatis.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=24,
        help="Jumlah sampel overlay per kelas.",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=3,
        help="Batas Hamming dHash untuk near duplicate lintas split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assume-coco17",
        action="store_true",
        help=(
            "Gunakan nama dan skeleton COCO-17 bila kpt_shape berisi 17 titik. "
            "Aktifkan hanya bila urutan anotasi sudah diverifikasi."
        ),
    )
    parser.add_argument("--dark-threshold", type=float, default=45.0)
    parser.add_argument("--blur-threshold", type=float, default=120.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str) -> float | None:
    try:
        result = float(value)
        if not math.isfinite(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, dict):
        result = {}
        for key, value in raw.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    return {}


def find_yaml(dataset_dir: Path, explicit_yaml: Path | None) -> Path:
    if explicit_yaml is not None:
        yaml_path = explicit_yaml.resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML tidak ditemukan: {yaml_path}")
        return yaml_path

    preferred = [
        dataset_dir / "data.yaml",
        dataset_dir / "dataset.yaml",
        dataset_dir / "data.yml",
        dataset_dir / "dataset.yml",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate.resolve()

    yaml_files = sorted(list(dataset_dir.rglob("*.yaml")) + list(dataset_dir.rglob("*.yml")))
    if not yaml_files:
        raise FileNotFoundError("Tidak menemukan data.yaml/dataset.yaml.")
    return yaml_files[0].resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Isi YAML tidak valid: {path}")
    return data


def resolve_existing_path(
    raw_path: str | Path,
    yaml_dir: Path,
    dataset_dir: Path,
    yaml_base: Path,
) -> Path | None:
    path = Path(str(raw_path))
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            yaml_base / path,
            yaml_dir / path,
            dataset_dir / path,
        ])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def fallback_split_dirs(dataset_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)

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
    for split, path in candidates:
        if path.exists():
            result[split].append(path.resolve())
    return dict(result)


def resolve_split_dirs(
    yaml_data: dict[str, Any],
    yaml_path: Path,
    dataset_dir: Path,
) -> dict[str, list[Path]]:
    yaml_dir = yaml_path.parent
    path_value = yaml_data.get("path")
    yaml_base = yaml_dir

    if path_value:
        path_candidate = Path(str(path_value))
        if path_candidate.is_absolute():
            yaml_base = path_candidate
        else:
            yaml_base = (yaml_dir / path_candidate).resolve()

    result: dict[str, list[Path]] = defaultdict(list)
    for yaml_key, normalized_split in [("train", "train"), ("val", "valid"),
                                       ("valid", "valid"), ("test", "test")]:
        raw_value = yaml_data.get(yaml_key)
        if raw_value is None:
            continue

        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            resolved = resolve_existing_path(value, yaml_dir, dataset_dir, yaml_base)
            if resolved is not None:
                result[normalized_split].append(resolved)

    if not result:
        return fallback_split_dirs(dataset_dir)
    return dict(result)


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)

    # .../split/images -> .../split/labels
    if image_dir.name.lower() == "images":
        return image_dir.parent / "labels"

    # .../images/split -> .../labels/split
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)

    return image_dir.parent / "labels"


def iter_images(image_dirs: dict[str, list[Path]]) -> Iterable[tuple[str, Path, Path]]:
    seen: set[Path] = set()
    for split, directories in image_dirs.items():
        for image_dir in directories:
            label_dir = infer_label_dir(image_dir)
            for image_path in sorted(image_dir.rglob("*")):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                    and image_path.resolve() not in seen
                ):
                    seen.add(image_path.resolve())
                    relative = image_path.relative_to(image_dir)
                    label_path = (label_dir / relative).with_suffix(".txt")
                    yield split, image_path.resolve(), label_path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def image_metrics(image: Image.Image) -> tuple[float, float]:
    gray = image.convert("L")
    brightness = float(np.asarray(gray, dtype=np.float32).mean())
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = float(np.asarray(edges, dtype=np.float32).var())
    return brightness, edge_variance


def parse_pose_label(
    label_path: Path,
    split: str,
    image_path: Path,
    class_count: int | None,
    kpt_count: int,
    kpt_dim: int,
) -> tuple[list[PoseObject], list[dict]]:
    objects: list[PoseObject] = []
    issue_rows: list[dict] = []

    if not label_path.exists():
        issue_rows.append({
            "severity": "error",
            "split": split,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "line_number": "",
            "issue_type": "missing_label",
            "details": "File label tidak ditemukan.",
        })
        return objects, issue_rows

    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        issue_rows.append({
            "severity": "warning",
            "split": split,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "line_number": "",
            "issue_type": "empty_label",
            "details": "File label kosong.",
        })
        return objects, issue_rows

    expected_tokens = 5 + kpt_count * kpt_dim

    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = line.strip().split()
        issues: list[str] = []

        if len(tokens) != expected_tokens:
            issues.append(
                f"token_count={len(tokens)}, expected={expected_tokens}"
            )

        numeric = [safe_float(token) for token in tokens]
        if any(value is None for value in numeric):
            issues.append("non_numeric_or_non_finite_value")

        if issues:
            object_row = PoseObject(
                split=split,
                image_path=str(image_path),
                label_path=str(label_path),
                line_number=line_number,
                class_id=-1,
                bbox_x=0.0,
                bbox_y=0.0,
                bbox_w=0.0,
                bbox_h=0.0,
                keypoints=[],
                valid=False,
                issues=issues,
            )
            objects.append(object_row)
            for issue in issues:
                issue_rows.append({
                    "severity": "error",
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "line_number": line_number,
                    "issue_type": "invalid_annotation_line",
                    "details": issue,
                })
            continue

        values = [float(value) for value in numeric if value is not None]
        class_raw = values[0]
        class_id = int(class_raw)

        if class_raw != class_id:
            issues.append(f"class_id_not_integer={class_raw}")
        if class_id < 0:
            issues.append(f"negative_class_id={class_id}")
        if class_count is not None and class_id >= class_count:
            issues.append(f"class_id_out_of_range={class_id}")

        bbox = values[1:5]
        bbox_names = ["x", "y", "w", "h"]
        for name, value in zip(bbox_names, bbox):
            if not 0.0 <= value <= 1.0:
                issues.append(f"bbox_{name}_outside_0_1={value}")
        if bbox[2] <= 0 or bbox[3] <= 0:
            issues.append("bbox_non_positive_size")

        keypoint_values = values[5:]
        keypoints: list[list[float]] = []
        if len(keypoint_values) == kpt_count * kpt_dim:
            for index in range(kpt_count):
                start = index * kpt_dim
                point = keypoint_values[start:start + kpt_dim]
                x, y = point[0], point[1]

                # (0,0) lazim dipakai untuk keypoint tidak teranotasi.
                if not (x == 0.0 and y == 0.0):
                    if not 0.0 <= x <= 1.0:
                        issues.append(f"kp{index}_x_outside_0_1={x}")
                    if not 0.0 <= y <= 1.0:
                        issues.append(f"kp{index}_y_outside_0_1={y}")

                if kpt_dim >= 3:
                    visibility = point[2]
                    if not 0.0 <= visibility <= 2.0:
                        issues.append(
                            f"kp{index}_visibility_outside_0_2={visibility}"
                        )
                keypoints.append(point)

        valid = len(issues) == 0
        obj = PoseObject(
            split=split,
            image_path=str(image_path),
            label_path=str(label_path),
            line_number=line_number,
            class_id=class_id,
            bbox_x=bbox[0],
            bbox_y=bbox[1],
            bbox_w=bbox[2],
            bbox_h=bbox[3],
            keypoints=keypoints,
            valid=valid,
            issues=issues,
        )
        objects.append(obj)

        for issue in issues:
            issue_rows.append({
                "severity": "error",
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "line_number": line_number,
                "issue_type": "invalid_pose_value",
                "details": issue,
            })

    return objects, issue_rows


def inspect_image(
    split: str,
    image_path: Path,
    label_path: Path,
    dataset_dir: Path,
    class_count: int | None,
    kpt_count: int,
    kpt_dim: int,
) -> tuple[ImageAudit, list[PoseObject], list[dict]]:
    objects, issues = parse_pose_label(
        label_path, split, image_path, class_count, kpt_count, kpt_dim
    )

    try:
        file_hash = sha256_file(image_path)
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            brightness, edge_variance = image_metrics(image)
            dhash_value = dhash64(image)

        audit = ImageAudit(
            split=split,
            image_path=str(image_path),
            relative_path=str(image_path.relative_to(dataset_dir)),
            label_path=str(label_path),
            width=width,
            height=height,
            file_size_bytes=image_path.stat().st_size,
            sha256=file_hash,
            dhash64=f"{dhash_value:016x}",
            brightness_mean=round(brightness, 4),
            edge_variance=round(edge_variance, 4),
            is_corrupt=False,
            label_exists=label_path.exists(),
            label_empty=label_path.exists() and label_path.stat().st_size == 0,
            object_count=len(objects),
            valid_object_count=sum(obj.valid for obj in objects),
            invalid_object_count=sum(not obj.valid for obj in objects),
            error="",
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        audit = ImageAudit(
            split=split,
            image_path=str(image_path),
            relative_path=str(image_path.relative_to(dataset_dir)),
            label_path=str(label_path),
            width=0,
            height=0,
            file_size_bytes=image_path.stat().st_size if image_path.exists() else 0,
            sha256="",
            dhash64="",
            brightness_mean=0.0,
            edge_variance=0.0,
            is_corrupt=True,
            label_exists=label_path.exists(),
            label_empty=label_path.exists() and label_path.stat().st_size == 0,
            object_count=len(objects),
            valid_object_count=sum(obj.valid for obj in objects),
            invalid_object_count=sum(not obj.valid for obj in objects),
            error=f"{type(exc).__name__}: {exc}",
        )
        issues.append({
            "severity": "error",
            "split": split,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "line_number": "",
            "issue_type": "corrupt_image",
            "details": audit.error,
        })

    return audit, objects, issues


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_cross_split_duplicates(
    audits: list[ImageAudit],
    max_distance: int,
) -> list[dict]:
    valid = [
        audit for audit in audits
        if not audit.is_corrupt and audit.sha256 and audit.dhash64
    ]

    rows: list[dict] = []

    sha_groups: dict[str, list[ImageAudit]] = defaultdict(list)
    for audit in valid:
        sha_groups[audit.sha256].append(audit)

    exact_pairs: set[tuple[str, str]] = set()
    for group in sha_groups.values():
        splits = {item.split for item in group}
        if len(splits) < 2:
            continue
        for i in range(len(group) - 1):
            for j in range(i + 1, len(group)):
                if group[i].split == group[j].split:
                    continue
                pair = tuple(sorted((group[i].image_path, group[j].image_path)))
                exact_pairs.add(pair)
                rows.append({
                    "duplicate_type": "exact",
                    "hamming_distance": 0,
                    "split_a": group[i].split,
                    "path_a": group[i].image_path,
                    "split_b": group[j].split,
                    "path_b": group[j].image_path,
                })

    hashes = [int(item.dhash64, 16) for item in valid]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, hash_value in enumerate(hashes):
        for band in range(4):
            segment = (hash_value >> (band * 16)) & 0xFFFF
            buckets[(band, segment)].append(index)

    candidates: set[tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for pos_a in range(len(indices) - 1):
            for pos_b in range(pos_a + 1, len(indices)):
                a, b = sorted((indices[pos_a], indices[pos_b]))
                if valid[a].split != valid[b].split:
                    candidates.add((a, b))

    for index_a, index_b in candidates:
        item_a, item_b = valid[index_a], valid[index_b]
        pair = tuple(sorted((item_a.image_path, item_b.image_path)))
        if pair in exact_pairs:
            continue
        distance = hamming_distance64(hashes[index_a], hashes[index_b])
        if distance <= max_distance:
            rows.append({
                "duplicate_type": "near",
                "hamming_distance": distance,
                "split_a": item_a.split,
                "path_a": item_a.image_path,
                "split_b": item_b.split,
                "path_b": item_b.image_path,
            })

    rows.sort(key=lambda row: (row["hamming_distance"], row["path_a"], row["path_b"]))
    return rows


def keypoint_names(
    yaml_data: dict[str, Any],
    kpt_count: int,
    assume_coco17: bool,
) -> list[str]:
    raw = yaml_data.get("kpt_names")
    if isinstance(raw, list) and len(raw) == kpt_count:
        return [str(item) for item in raw]

    if isinstance(raw, dict):
        # Beberapa YAML menyimpan nama per kelas.
        for value in raw.values():
            if isinstance(value, list) and len(value) == kpt_count:
                return [str(item) for item in value]

    if assume_coco17 and kpt_count == 17:
        return COCO17_NAMES.copy()

    return [f"kp_{index:02d}" for index in range(kpt_count)]


def create_keypoint_stats(
    objects: list[PoseObject],
    names: list[str],
    kpt_dim: int,
    class_names: dict[int, str],
) -> list[dict]:
    stats: dict[tuple[int, int], Counter] = defaultdict(Counter)

    for obj in objects:
        if not obj.valid or len(obj.keypoints) != len(names):
            continue
        for index, point in enumerate(obj.keypoints):
            counter = stats[(obj.class_id, index)]
            counter["total_objects"] += 1
            x, y = point[0], point[1]
            coordinate_present = not (x == 0.0 and y == 0.0)
            if coordinate_present:
                counter["coordinate_present"] += 1

            if kpt_dim >= 3:
                visibility = point[2]
                if visibility > 0:
                    counter["visibility_gt_0"] += 1
                if visibility >= 2:
                    counter["visibility_eq_2"] += 1
            elif coordinate_present:
                counter["visibility_gt_0"] += 1
                counter["visibility_eq_2"] += 1

    rows: list[dict] = []
    for (class_id, index), counter in sorted(stats.items()):
        total = counter["total_objects"]
        rows.append({
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            "keypoint_index": index,
            "keypoint_name": names[index],
            "total_objects": total,
            "coordinate_present": counter["coordinate_present"],
            "coordinate_present_pct": round(
                100.0 * counter["coordinate_present"] / total, 3
            ) if total else 0.0,
            "visibility_gt_0": counter["visibility_gt_0"],
            "visibility_gt_0_pct": round(
                100.0 * counter["visibility_gt_0"] / total, 3
            ) if total else 0.0,
            "visibility_eq_2": counter["visibility_eq_2"],
            "visibility_eq_2_pct": round(
                100.0 * counter["visibility_eq_2"] / total, 3
            ) if total else 0.0,
        })
    return rows


def draw_pose_overlay(
    image: Image.Image,
    obj: PoseObject,
    use_coco_edges: bool,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size

    left = (obj.bbox_x - obj.bbox_w / 2) * width
    top = (obj.bbox_y - obj.bbox_h / 2) * height
    right = (obj.bbox_x + obj.bbox_w / 2) * width
    bottom = (obj.bbox_y + obj.bbox_h / 2) * height
    draw.rectangle((left, top, right, bottom), width=max(2, width // 300))

    coordinates: list[tuple[float, float] | None] = []
    for point in obj.keypoints:
        x, y = point[0], point[1]
        visible = True
        if len(point) >= 3:
            visible = point[2] > 0
        if (x == 0 and y == 0) or not visible:
            coordinates.append(None)
            continue
        coordinates.append((x * width, y * height))

    if use_coco_edges and len(coordinates) == 17:
        for index_a, index_b in COCO17_EDGES:
            point_a, point_b = coordinates[index_a], coordinates[index_b]
            if point_a is not None and point_b is not None:
                draw.line((point_a, point_b), width=max(2, width // 350))

    radius = max(3, min(width, height) // 120)
    for point in coordinates:
        if point is None:
            continue
        x, y = point
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), width=2)

    return output


def create_contact_sheets(
    audits: list[ImageAudit],
    objects: list[PoseObject],
    class_names: dict[int, str],
    output_dir: Path,
    samples_per_class: int,
    seed: int,
    use_coco_edges: bool,
) -> None:
    rng = random.Random(seed)
    objects_by_class: dict[int, list[PoseObject]] = defaultdict(list)
    for obj in objects:
        if obj.valid:
            objects_by_class[obj.class_id].append(obj)

    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 240, 180
    text_h = 48
    columns = 4

    for class_id, class_objects in sorted(objects_by_class.items()):
        sample = rng.sample(
            class_objects,
            min(samples_per_class, len(class_objects)),
        )
        row_count = math.ceil(len(sample) / columns)
        canvas = Image.new(
            "RGB",
            (columns * thumb_w, row_count * (thumb_h + text_h) + 40),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        class_name = class_names.get(class_id, f"class_{class_id}")
        draw.text((8, 10), f"{class_id}: {class_name}", fill="black")

        for index, obj in enumerate(sample):
            x_cell = (index % columns) * thumb_w
            y_cell = 40 + (index // columns) * (thumb_h + text_h)

            try:
                with Image.open(obj.image_path) as image:
                    overlay = draw_pose_overlay(image, obj, use_coco_edges)
                    fitted = ImageOps.contain(
                        overlay,
                        (thumb_w - 8, thumb_h - 8),
                        Image.Resampling.LANCZOS,
                    )
                    paste_x = x_cell + (thumb_w - fitted.width) // 2
                    paste_y = y_cell + (thumb_h - fitted.height) // 2
                    canvas.paste(fitted, (paste_x, paste_y))
            except (UnidentifiedImageError, OSError):
                draw.rectangle(
                    (x_cell + 5, y_cell + 5,
                     x_cell + thumb_w - 5, y_cell + thumb_h - 5),
                    outline="black",
                )
                draw.text((x_cell + 15, y_cell + 70), "IMAGE ERROR", fill="black")

            filename = Path(obj.image_path).name[:28]
            text = f"{obj.split} | {filename}"
            draw.text((x_cell + 4, y_cell + thumb_h + 3), text, fill="black")

        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in class_name
        )
        canvas.save(
            output_dir / f"class_{class_id}_{safe_name}.jpg",
            quality=90,
        )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_dir.exists():
        print(f"ERROR: Dataset tidak ditemukan: {dataset_dir}", file=sys.stderr)
        return 1

    try:
        yaml_path = find_yaml(dataset_dir, args.yaml)
        yaml_data = load_yaml(yaml_path)
        split_dirs = resolve_split_dirs(yaml_data, yaml_path, dataset_dir)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    kpt_shape = yaml_data.get("kpt_shape")
    if (
        not isinstance(kpt_shape, (list, tuple))
        or len(kpt_shape) != 2
    ):
        print(
            "ERROR: data.yaml tidak memiliki kpt_shape: [jumlah_titik, dimensi].",
            file=sys.stderr,
        )
        return 1

    kpt_count = int(kpt_shape[0])
    kpt_dim = int(kpt_shape[1])
    if kpt_count <= 0 or kpt_dim not in {2, 3}:
        print(f"ERROR: kpt_shape tidak didukung: {kpt_shape}", file=sys.stderr)
        return 1

    class_names = normalize_names(yaml_data.get("names"))
    declared_nc = yaml_data.get("nc")
    class_count = None
    if class_names:
        class_count = max(class_names) + 1
    elif isinstance(declared_nc, int):
        class_count = declared_nc

    print(f"YAML: {yaml_path}")
    print(f"kpt_shape: [{kpt_count}, {kpt_dim}]")
    print(f"Classes: {class_names}")
    print(f"Split directories: {split_dirs}")

    image_entries = list(iter_images(split_dirs))
    if not image_entries:
        print("ERROR: Tidak menemukan gambar pada split dataset.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    audits: list[ImageAudit] = []
    pose_objects: list[PoseObject] = []
    issue_rows: list[dict] = []

    for index, (split, image_path, label_path) in enumerate(image_entries, start=1):
        audit, objects, issues = inspect_image(
            split,
            image_path,
            label_path,
            dataset_dir,
            class_count,
            kpt_count,
            kpt_dim,
        )
        audits.append(audit)
        pose_objects.extend(objects)
        issue_rows.extend(issues)

        if index % 250 == 0 or index == len(image_entries):
            print(f"Memeriksa {index}/{len(image_entries)} gambar...")

    image_rows = [asdict(audit) for audit in audits]
    write_csv(
        output_dir / "image_audit.csv",
        image_rows,
        fieldnames=list(ImageAudit.__dataclass_fields__.keys()),
    )

    issue_fields = [
        "severity", "split", "image_path", "label_path",
        "line_number", "issue_type", "details",
    ]
    write_csv(
        output_dir / "annotation_issues.csv",
        issue_rows,
        fieldnames=issue_fields,
    )

    split_counts = Counter(audit.split for audit in audits)
    split_rows = [
        {"split": split, "image_count": count}
        for split, count in sorted(split_counts.items())
    ]
    write_csv(
        output_dir / "split_summary.csv",
        split_rows,
        fieldnames=["split", "image_count"],
    )

    class_split_counts = Counter(
        (obj.split, obj.class_id)
        for obj in pose_objects
        if obj.valid
    )
    class_rows = [
        {
            "split": split,
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            "object_count": count,
        }
        for (split, class_id), count in sorted(class_split_counts.items())
    ]
    write_csv(
        output_dir / "class_distribution.csv",
        class_rows,
        fieldnames=["split", "class_id", "class_name", "object_count"],
    )

    names = keypoint_names(yaml_data, kpt_count, args.assume_coco17)
    keypoint_rows = create_keypoint_stats(
        pose_objects, names, kpt_dim, class_names
    )
    write_csv(
        output_dir / "keypoint_visibility.csv",
        keypoint_rows,
        fieldnames=[
            "class_id", "class_name", "keypoint_index", "keypoint_name",
            "total_objects", "coordinate_present", "coordinate_present_pct",
            "visibility_gt_0", "visibility_gt_0_pct",
            "visibility_eq_2", "visibility_eq_2_pct",
        ],
    )

    print("Mencari duplicate lintas split...")
    duplicate_rows = find_cross_split_duplicates(
        audits, args.near_duplicate_distance
    )
    write_csv(
        output_dir / "cross_split_duplicates.csv",
        duplicate_rows,
        fieldnames=[
            "duplicate_type", "hamming_distance",
            "split_a", "path_a", "split_b", "path_b",
        ],
    )

    print("Membuat contact sheet overlay...")
    create_contact_sheets(
        audits,
        pose_objects,
        class_names,
        output_dir / "annotated_contact_sheets",
        args.samples_per_class,
        args.seed,
        args.assume_coco17 and kpt_count == 17,
    )

    valid_objects = [obj for obj in pose_objects if obj.valid]
    corrupt_count = sum(audit.is_corrupt for audit in audits)
    missing_label_count = sum(not audit.label_exists for audit in audits)
    empty_label_count = sum(audit.label_empty for audit in audits)
    images_with_invalid_objects = sum(
        audit.invalid_object_count > 0 for audit in audits
    )
    dark_count = sum(
        (not audit.is_corrupt)
        and audit.brightness_mean < args.dark_threshold
        for audit in audits
    )
    blur_count = sum(
        (not audit.is_corrupt)
        and audit.edge_variance < args.blur_threshold
        for audit in audits
    )

    duplicate_type_counts = Counter(
        row["duplicate_type"] for row in duplicate_rows
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "yaml_path": str(yaml_path),
        "kpt_shape": [kpt_count, kpt_dim],
        "class_names": class_names,
        "split_image_counts": dict(sorted(split_counts.items())),
        "total_images": len(audits),
        "corrupt_images": corrupt_count,
        "missing_labels": missing_label_count,
        "empty_labels": empty_label_count,
        "total_annotation_objects": len(pose_objects),
        "valid_annotation_objects": len(valid_objects),
        "invalid_annotation_objects": len(pose_objects) - len(valid_objects),
        "images_with_invalid_objects": images_with_invalid_objects,
        "annotation_issue_count": len(issue_rows),
        "dark_images": dark_count,
        "potentially_blurry_images": blur_count,
        "cross_split_exact_duplicate_pairs": duplicate_type_counts["exact"],
        "cross_split_near_duplicate_pairs": duplicate_type_counts["near"],
        "near_duplicate_distance": args.near_duplicate_distance,
        "warnings": [],
    }

    if corrupt_count:
        summary["warnings"].append("Terdapat gambar rusak.")
    if missing_label_count:
        summary["warnings"].append("Terdapat gambar tanpa file label.")
    if empty_label_count:
        summary["warnings"].append("Terdapat file label kosong.")
    if issue_rows:
        summary["warnings"].append("Terdapat anotasi yang tidak valid.")
    if duplicate_rows:
        summary["warnings"].append(
            "Terdapat exact/near duplicate lintas split."
        )

    (output_dir / "pose_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown = [
        "# Ringkasan Audit Postureexercise",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- YAML: `{yaml_path}`",
        f"- kpt_shape: **[{kpt_count}, {kpt_dim}]**",
        f"- Kelas: **{class_names}**",
        f"- Total citra: **{len(audits)}**",
        f"- Citra rusak: **{corrupt_count}**",
        f"- Gambar tanpa label: **{missing_label_count}**",
        f"- Label kosong: **{empty_label_count}**",
        f"- Total objek anotasi: **{len(pose_objects)}**",
        f"- Objek anotasi valid: **{len(valid_objects)}**",
        f"- Objek anotasi tidak valid: **{len(pose_objects)-len(valid_objects)}**",
        f"- Exact duplicate lintas split: **{duplicate_type_counts['exact']}**",
        f"- Near duplicate lintas split: **{duplicate_type_counts['near']}**",
        "",
        "## Distribusi Split",
        "",
        "| Split | Gambar |",
        "|---|---:|",
    ]
    for split, count in sorted(split_counts.items()):
        markdown.append(f"| {split} | {count} |")

    markdown.extend([
        "",
        "## Peringatan",
        "",
    ])
    if summary["warnings"]:
        markdown.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        markdown.append("- Tidak ada peringatan otomatis utama.")

    markdown.extend([
        "",
        "## File yang Harus Diperiksa",
        "",
        "1. `class_distribution.csv`",
        "2. `keypoint_visibility.csv`",
        "3. `annotation_issues.csv`",
        "4. `cross_split_duplicates.csv`",
        "5. `annotated_contact_sheets/`",
        "",
        "## Catatan",
        "",
        "Keypoint pada format YOLO-Pose tetap merupakan koordinat 2D pada citra. "
        "Dimensi ketiga pada `kpt_shape: [K, 3]` adalah visibility, bukan koordinat kedalaman z.",
    ])

    (output_dir / "pose_audit_summary.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print("")
    print("Audit selesai.")
    print(f"Output: {output_dir}")
    print(f"Images: {len(audits)}")
    print(f"Valid objects: {len(valid_objects)}")
    print(f"Issues: {len(issue_rows)}")
    print(f"Cross-split duplicates: {len(duplicate_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
