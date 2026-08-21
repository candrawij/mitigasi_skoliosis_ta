#!/usr/bin/env python3
"""
Audit dataset format COCO Detection (bounding box, tanpa keypoint).

Struktur dataset yang didukung (Roboflow COCO export):
    dataset/
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── image1.jpg
    │   └── ...
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── ...
    └── test/
        ├── _annotations.coco.json
        └── ...

Pemeriksaan:
    - Memvalidasi struktur _annotations.coco.json per split.
    - Memeriksa pasangan gambar-anotasi (gambar tanpa anotasi, anotasi tanpa gambar).
    - Memvalidasi bbox (koordinat, area).
    - Menghitung distribusi split/kelas (berdasarkan anotasi).
    - Mendeteksi exact/near duplicate lintas split (dHash).
    - Membangun laporan leakage lintas split.
    - Membuat contact sheet dengan overlay bbox per kelas.
    - TIDAK mengubah atau menghapus dataset.

Contoh:
    python audit_coco_detection_dataset.py ^
        "02_data/raw/Sitting Posture Detection.v2i.coco" ^
        --output-dir "07_results/dataset_audit/sitting_posture_detection_initial"
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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError


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

BBOX_COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 120, 255),
    (255, 180, 0),
    (200, 80, 200),
    (0, 200, 200),
]


@dataclass
class ImageRecord:
    split: str
    image_id: int
    filename: str
    relative_path: str
    width: int
    height: int
    file_size_bytes: int
    sha256: str
    dhash64: str
    brightness_mean: float
    edge_variance: float
    is_corrupt: bool
    has_annotation: bool
    annotation_count: int
    error: str


@dataclass
class AnnotationRecord:
    split: str
    annotation_id: int
    image_id: int
    image_filename: str
    category_id: int
    category_name: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    area: float
    bbox_valid: bool
    bbox_issue: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dataset format COCO Detection."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Folder utama dataset COCO yang sudah diekstrak.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_audit_output"),
        help="Folder output audit. Default: dataset_audit_output",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=5,
        help="Batas Hamming distance dHash untuk near duplicate. Default: 5",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=25,
        help="Jumlah gambar sampel pada contact sheet per kelas. Default: 25",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=224,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hashing & metrik citra
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> int:
    """Difference hash 64-bit."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def image_metrics(image: Image.Image) -> tuple[float, float]:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    brightness = float(arr.mean())
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = float(np.asarray(edges, dtype=np.float32).var())
    return brightness, edge_variance


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Membaca dataset COCO
# ---------------------------------------------------------------------------

def load_coco_split(
    split_dir: Path,
    split_name: str,
) -> tuple[dict, list[ImageRecord], list[AnnotationRecord]]:
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  PERINGATAN: {ann_path} tidak ditemukan, split {split_name} dilewati.")
        return {}, [], []

    with ann_path.open(encoding="utf-8") as fh:
        coco = json.load(fh)

    categories: dict[int, str] = {
        c["id"]: c["name"] for c in coco.get("categories", [])
    }

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    image_records: list[ImageRecord] = []
    annotation_records: list[AnnotationRecord] = []

    for img_entry in coco.get("images", []):
        img_id = img_entry["id"]
        filename = img_entry["file_name"]
        img_path = split_dir / filename

        anns = anns_by_image.get(img_id, [])
        ann_count = len(anns)

        if not img_path.exists():
            record = ImageRecord(
                split=split_name,
                image_id=img_id,
                filename=filename,
                relative_path=f"{split_dir.name}/{filename}",
                width=img_entry.get("width", 0),
                height=img_entry.get("height", 0),
                file_size_bytes=0,
                sha256="",
                dhash64="",
                brightness_mean=0.0,
                edge_variance=0.0,
                is_corrupt=True,
                has_annotation=ann_count > 0,
                annotation_count=ann_count,
                error="FileNotFoundError: file tidak ditemukan",
            )
        else:
            try:
                file_hash = sha256_file(img_path)
                with Image.open(img_path) as img:
                    img.load()
                    w, h = img.size
                    brightness, edge_var = image_metrics(img)
                    dhash_value = dhash64(img)

                record = ImageRecord(
                    split=split_name,
                    image_id=img_id,
                    filename=filename,
                    relative_path=f"{split_dir.name}/{filename}",
                    width=w,
                    height=h,
                    file_size_bytes=img_path.stat().st_size,
                    sha256=file_hash,
                    dhash64=f"{dhash_value:016x}",
                    brightness_mean=round(brightness, 4),
                    edge_variance=round(edge_var, 4),
                    is_corrupt=False,
                    has_annotation=ann_count > 0,
                    annotation_count=ann_count,
                    error="",
                )
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                record = ImageRecord(
                    split=split_name,
                    image_id=img_id,
                    filename=filename,
                    relative_path=f"{split_dir.name}/{filename}",
                    width=img_entry.get("width", 0),
                    height=img_entry.get("height", 0),
                    file_size_bytes=img_path.stat().st_size if img_path.exists() else 0,
                    sha256="",
                    dhash64="",
                    brightness_mean=0.0,
                    edge_variance=0.0,
                    is_corrupt=True,
                    has_annotation=ann_count > 0,
                    annotation_count=ann_count,
                    error=f"{type(exc).__name__}: {exc}",
                )

        image_records.append(record)

        for ann in anns:
            bbox = ann.get("bbox", [])
            bbox_valid = True
            bbox_issue = ""

            if len(bbox) == 4:
                x, y, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    bbox_valid = False
                    bbox_issue = f"width/height tidak positif: w={bw}, h={bh}"
                elif x < 0 or y < 0:
                    bbox_valid = False
                    bbox_issue = f"koordinat negatif: x={x}, y={y}"
            else:
                bbox_valid = False
                bbox_issue = f"format bbox tidak valid: {bbox}"
                x, y, bw, bh = 0.0, 0.0, 0.0, 0.0

            cat_id = ann.get("category_id", -1)
            annotation_records.append(
                AnnotationRecord(
                    split=split_name,
                    annotation_id=ann["id"],
                    image_id=img_id,
                    image_filename=filename,
                    category_id=cat_id,
                    category_name=categories.get(cat_id, f"unknown_{cat_id}"),
                    bbox_x=x,
                    bbox_y=y,
                    bbox_w=bw,
                    bbox_h=bh,
                    area=ann.get("area", bw * bh),
                    bbox_valid=bbox_valid,
                    bbox_issue=bbox_issue,
                )
            )

    return coco, image_records, annotation_records


# ---------------------------------------------------------------------------
# Deteksi duplicate
# ---------------------------------------------------------------------------

def find_exact_duplicates(records: list[ImageRecord]) -> list[dict]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if not r.is_corrupt and r.sha256:
            groups[r.sha256].append(r)

    rows: list[dict] = []
    group_id = 0
    for file_hash, group in groups.items():
        if len(group) < 2:
            continue
        group_id += 1
        for r in group:
            rows.append({
                "duplicate_group": group_id,
                "sha256": file_hash,
                "split": r.split,
                "filename": r.filename,
                "relative_path": r.relative_path,
            })
    return rows


def find_near_duplicates(
    records: list[ImageRecord],
    max_distance: int,
) -> list[dict]:
    valid = [r for r in records if not r.is_corrupt and r.dhash64]
    hashes = [int(r.dhash64, 16) for r in valid]

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, hv in enumerate(hashes):
        for band in range(4):
            segment = (hv >> (band * 16)) & 0xFFFF
            buckets[(band, segment)].append(idx)

    candidates: set[tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for pos_a in range(len(indices) - 1):
            for pos_b in range(pos_a + 1, len(indices)):
                a, b = indices[pos_a], indices[pos_b]
                if a > b:
                    a, b = b, a
                candidates.add((a, b))

    rows: list[dict] = []
    for idx_a, idx_b in sorted(candidates):
        ra, rb = valid[idx_a], valid[idx_b]
        if ra.sha256 == rb.sha256:
            continue
        dist = hamming_distance64(hashes[idx_a], hashes[idx_b])
        if dist <= max_distance:
            rows.append({
                "hamming_distance": dist,
                "same_split": ra.split == rb.split,
                "split_a": ra.split,
                "filename_a": ra.filename,
                "path_a": ra.relative_path,
                "split_b": rb.split,
                "filename_b": rb.filename,
                "path_b": rb.relative_path,
            })

    rows.sort(key=lambda r: r["hamming_distance"])
    return rows


def build_split_leakage(
    exact_duplicates: list[dict],
    near_duplicates: list[dict],
) -> list[dict]:
    leakage: list[dict] = []

    exact_groups: dict[int, list[dict]] = defaultdict(list)
    for row in exact_duplicates:
        exact_groups[int(row["duplicate_group"])].append(row)

    for group_id, group in exact_groups.items():
        splits = {row["split"] for row in group}
        if len(splits) > 1:
            leakage.append({
                "type": "exact_duplicate",
                "distance": 0,
                "split_a": group[0]["split"],
                "path_a": group[0]["relative_path"],
                "split_b": group[1]["split"],
                "path_b": group[1]["relative_path"],
                "note": f"Exact duplicate group {group_id} muncul di {sorted(splits)}",
            })

    for row in near_duplicates:
        if not row["same_split"]:
            leakage.append({
                "type": "near_duplicate",
                "distance": row["hamming_distance"],
                "split_a": row["split_a"],
                "path_a": row["path_a"],
                "split_b": row["split_b"],
                "path_b": row["path_b"],
                "note": "Near duplicate muncul pada split berbeda",
            })

    return leakage


# ---------------------------------------------------------------------------
# Contact sheet dengan overlay bbox
# ---------------------------------------------------------------------------

def create_contact_sheets(
    image_records: list[ImageRecord],
    annotation_records: list[AnnotationRecord],
    dataset_dir: Path,
    output_dir: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    ann_by_img_split: dict[tuple[str, str], list[AnnotationRecord]] = defaultdict(list)
    for ann in annotation_records:
        ann_by_img_split[(ann.split, ann.image_filename)].append(ann)

    class_images: dict[str, list[tuple[str, ImageRecord]]] = defaultdict(list)
    for ann in annotation_records:
        if ann.bbox_valid:
            for r in image_records:
                if r.split == ann.split and r.filename == ann.image_filename and not r.is_corrupt:
                    class_images[ann.category_name].append((ann.split, r))
                    break

    # Deduplikasi
    deduped: dict[str, list[tuple[str, ImageRecord]]] = {}
    for cat, items in class_images.items():
        seen: set[str] = set()
        unique = []
        for split, rec in items:
            key = rec.relative_path
            if key not in seen:
                seen.add(key)
                unique.append((split, rec))
        deduped[cat] = unique

    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    thumb_size = (200, 200)
    label_height = 36
    columns = 5

    for class_name, items in sorted(deduped.items()):
        sample = rng.sample(items, min(samples_per_class, len(items)))
        rows_count = math.ceil(len(sample) / columns)
        sheet = Image.new(
            "RGB",
            (columns * thumb_size[0], rows_count * (thumb_size[1] + label_height)),
            (240, 240, 240),
        )
        draw = ImageDraw.Draw(sheet)

        for idx, (split, rec) in enumerate(sample):
            img_path = dataset_dir / rec.relative_path
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    ann_draw = ImageDraw.Draw(img)
                    anns = ann_by_img_split.get((split, rec.filename), [])

                    color_map: dict[str, tuple[int, int, int]] = {}
                    color_idx = 0
                    for ann in anns:
                        if ann.category_name not in color_map:
                            color_map[ann.category_name] = BBOX_COLORS[color_idx % len(BBOX_COLORS)]
                            color_idx += 1
                        c = color_map[ann.category_name]
                        x1 = int(ann.bbox_x)
                        y1 = int(ann.bbox_y)
                        x2 = int(ann.bbox_x + ann.bbox_w)
                        y2 = int(ann.bbox_y + ann.bbox_h)
                        ann_draw.rectangle([x1, y1, x2, y2], outline=c, width=3)

                    img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    x_pos = (idx % columns) * thumb_size[0]
                    y_pos = (idx // columns) * (thumb_size[1] + label_height)
                    paste_x = x_pos + (thumb_size[0] - img.width) // 2
                    paste_y = y_pos + (thumb_size[1] - img.height) // 2
                    sheet.paste(img, (paste_x, paste_y))

                    label_text = f"{split}: {rec.filename[:22]}"
                    draw.text((x_pos + 3, y_pos + thumb_size[1] + 3), label_text, fill=(40, 40, 40))
            except Exception:
                pass

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in class_name)
        sheet.save(sheet_dir / f"{safe_name}.jpg", quality=90)


# ---------------------------------------------------------------------------
# CSV & Ringkasan
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    image_records: list[ImageRecord],
    annotation_records: list[AnnotationRecord],
    exact_duplicates: list[dict],
    near_duplicates: list[dict],
    leakage: list[dict],
    categories: dict[int, str],
    dataset_dir: Path,
    args: argparse.Namespace,
) -> dict:
    valid_imgs = [r for r in image_records if not r.is_corrupt]
    corrupt_imgs = [r for r in image_records if r.is_corrupt]
    split_counts = Counter(r.split for r in valid_imgs)
    unannotated = [r for r in valid_imgs if r.annotation_count == 0]

    class_dist = Counter(
        (ann.split, ann.category_name)
        for ann in annotation_records
        if ann.bbox_valid
    )
    class_total = Counter(
        ann.category_name
        for ann in annotation_records
        if ann.bbox_valid
    )
    invalid_bbox_count = sum(1 for ann in annotation_records if not ann.bbox_valid)

    dark_count = sum(r.brightness_mean < args.dark_threshold for r in valid_imgs)
    blur_count = sum(r.edge_variance < args.blur_threshold for r in valid_imgs)
    low_res_count = sum(
        r.width < args.min_width or r.height < args.min_height for r in valid_imgs
    )
    exact_groups = len({row["duplicate_group"] for row in exact_duplicates})

    summary = {
        "dataset_dir": str(dataset_dir),
        "total_images": len(image_records),
        "valid_images": len(valid_imgs),
        "corrupt_images": len(corrupt_imgs),
        "unannotated_images": len(unannotated),
        "total_annotations": len(annotation_records),
        "invalid_bbox_annotations": invalid_bbox_count,
        "categories": categories,
        "split_counts": dict(sorted(split_counts.items())),
        "class_distribution": [
            {"split": split, "category": cat, "count": cnt}
            for (split, cat), cnt in sorted(class_dist.items())
        ],
        "class_total": dict(sorted(class_total.items())),
        "dark_images": dark_count,
        "dark_threshold": args.dark_threshold,
        "blurry_images": blur_count,
        "blur_threshold": args.blur_threshold,
        "low_resolution_images": low_res_count,
        "min_resolution": {"width": args.min_width, "height": args.min_height},
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": len(near_duplicates),
        "cross_split_leakage_pairs": len(leakage),
        "warnings": [],
    }

    if corrupt_imgs:
        summary["warnings"].append("Terdapat citra rusak atau tidak dapat dibaca.")
    if unannotated:
        summary["warnings"].append(
            f"Terdapat {len(unannotated)} gambar tanpa anotasi."
        )
    if invalid_bbox_count:
        summary["warnings"].append(
            f"Terdapat {invalid_bbox_count} anotasi dengan bbox tidak valid."
        )
    if leakage:
        summary["warnings"].append(
            "Terdapat exact/near duplicate pada split berbeda; ini berpotensi data leakage."
        )
    if len(class_total) > 1:
        counts = list(class_total.values())
        ratio = max(counts) / min(counts)
        summary["class_imbalance_ratio"] = round(ratio, 4)
        if ratio >= 2:
            summary["warnings"].append(
                "Rasio kelas terbesar:terkecil >= 2; pertimbangkan strategi imbalance."
            )
    else:
        summary["class_imbalance_ratio"] = None

    return summary


def write_summary_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# Ringkasan Audit Dataset COCO Detection",
        "",
        f"- Folder dataset: `{summary['dataset_dir']}`",
        f"- Total citra: **{summary['total_images']}**",
        f"- Citra valid: **{summary['valid_images']}**",
        f"- Citra rusak: **{summary['corrupt_images']}**",
        f"- Citra tanpa anotasi: **{summary['unannotated_images']}**",
        f"- Total anotasi: **{summary['total_annotations']}**",
        f"- Anotasi bbox tidak valid: **{summary['invalid_bbox_annotations']}**",
        f"- Grup exact duplicate: **{summary['exact_duplicate_groups']}**",
        f"- Pasangan near duplicate: **{summary['near_duplicate_pairs']}**",
        f"- Potensi leakage lintas split: **{summary['cross_split_leakage_pairs']}**",
        "",
        "## Distribusi Kelas (Total Anotasi)",
        "",
        "| Kelas | Jumlah Anotasi |",
        "|---|---:|",
    ]
    for cat, cnt in summary["class_total"].items():
        lines.append(f"| {cat} | {cnt} |")

    lines.extend([
        "",
        "## Distribusi Kelas per Split",
        "",
        "| Split | Kelas | Jumlah Anotasi |",
        "|---|---|---:|",
    ])
    for item in summary["class_distribution"]:
        lines.append(f"| {item['split']} | {item['category']} | {item['count']} |")

    lines.extend([
        "",
        "## Distribusi Split (Jumlah Gambar)",
        "",
        "| Split | Jumlah Gambar |",
        "|---|---:|",
    ])
    for split, cnt in summary["split_counts"].items():
        lines.append(f"| {split} | {cnt} |")

    lines.extend([
        "",
        "## Pemeriksaan Kualitas",
        "",
        f"- Citra gelap: **{summary['dark_images']}** "
        f"(threshold brightness < {summary['dark_threshold']})",
        f"- Citra berpotensi blur: **{summary['blurry_images']}** "
        f"(threshold edge variance < {summary['blur_threshold']})",
        f"- Citra di bawah resolusi minimum: **{summary['low_resolution_images']}**",
        "",
        "## Peringatan",
        "",
    ])
    if summary["warnings"]:
        for w in summary["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- Tidak ada peringatan otomatis utama.")

    lines.extend([
        "",
        "## Catatan",
        "",
        "Nilai blur, brightness, dan near duplicate adalah indikator awal. "
        "Keputusan akhir harus melalui pemeriksaan visual (lihat contact_sheets/).",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"ERROR: Folder dataset tidak ditemukan: {dataset_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    all_image_records: list[ImageRecord] = []
    all_annotation_records: list[AnnotationRecord] = []
    all_categories: dict[int, str] = {}

    splits_found = []
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        normalized = SPLIT_ALIASES.get(folder.name.lower())
        if normalized is None:
            continue
        ann_path = folder / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        splits_found.append((normalized, folder))

    if not splits_found:
        print(
            "ERROR: Tidak ada split valid ditemukan "
            "(train/valid/test dengan _annotations.coco.json).",
            file=sys.stderr,
        )
        return 1

    print(f"Split ditemukan: {[s for s, _ in splits_found]}")

    for split_name, split_dir in splits_found:
        print(f"Memproses split '{split_name}'...")
        coco, img_records, ann_records = load_coco_split(split_dir, split_name)
        if coco:
            for c in coco.get("categories", []):
                all_categories[c["id"]] = c["name"]
        all_image_records.extend(img_records)
        all_annotation_records.extend(ann_records)
        print(f"  {len(img_records)} gambar, {len(ann_records)} anotasi")

    # Tulis CSV gambar
    write_csv(
        output_dir / "image_audit.csv",
        [asdict(r) for r in all_image_records],
        fieldnames=list(ImageRecord.__dataclass_fields__.keys()),
    )

    # Tulis CSV anotasi
    write_csv(
        output_dir / "annotation_audit.csv",
        [asdict(r) for r in all_annotation_records],
        fieldnames=list(AnnotationRecord.__dataclass_fields__.keys()),
    )

    # Distribusi kelas
    class_counter = Counter(
        (ann.split, ann.category_name)
        for ann in all_annotation_records
        if ann.bbox_valid
    )
    write_csv(
        output_dir / "class_distribution.csv",
        [{"split": split, "category": cat, "count": cnt}
         for (split, cat), cnt in sorted(class_counter.items())],
        fieldnames=["split", "category", "count"],
    )

    print("Mencari exact duplicate...")
    exact_duplicates = find_exact_duplicates(all_image_records)
    write_csv(
        output_dir / "exact_duplicates.csv",
        exact_duplicates,
        fieldnames=["duplicate_group", "sha256", "split", "filename", "relative_path"],
    )

    print("Mencari near duplicate...")
    near_duplicates = find_near_duplicates(all_image_records, args.near_duplicate_distance)
    write_csv(
        output_dir / "near_duplicates.csv",
        near_duplicates,
        fieldnames=[
            "hamming_distance", "same_split",
            "split_a", "filename_a", "path_a",
            "split_b", "filename_b", "path_b",
        ],
    )

    leakage = build_split_leakage(exact_duplicates, near_duplicates)
    write_csv(
        output_dir / "split_leakage.csv",
        leakage,
        fieldnames=["type", "distance", "split_a", "path_a", "split_b", "path_b", "note"],
    )

    print("Membuat contact sheet...")
    create_contact_sheets(
        all_image_records,
        all_annotation_records,
        dataset_dir,
        output_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    summary = build_summary(
        all_image_records,
        all_annotation_records,
        exact_duplicates,
        near_duplicates,
        leakage,
        all_categories,
        dataset_dir,
        args,
    )
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_markdown(summary, output_dir / "audit_summary.md")

    print()
    print("Audit selesai.")
    print(f"Output              : {output_dir}")
    print(f"Citra valid         : {summary['valid_images']}")
    print(f"Citra rusak         : {summary['corrupt_images']}")
    print(f"Citra tanpa anotasi : {summary['unannotated_images']}")
    print(f"Exact dup groups    : {summary['exact_duplicate_groups']}")
    print(f"Near dup pairs      : {summary['near_duplicate_pairs']}")
    print(f"Cross-split leakage : {summary['cross_split_leakage_pairs']}")
    if summary.get("warnings"):
        print("\nPeringatan:")
        for w in summary["warnings"]:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
