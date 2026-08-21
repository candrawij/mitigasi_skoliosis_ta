#!/usr/bin/env python3
"""
Audit dataset citra klasifikasi postur.

Mendukung struktur umum Roboflow:
dataset/
├── train/
│   ├── upright/
│   ├── leaning_forward/
│   └── ...
├── valid/ atau val/
└── test/

Output:
- images_audit.csv
- class_distribution.csv
- exact_duplicates.csv
- near_duplicates.csv
- split_leakage.csv
- audit_summary.json
- audit_summary.md
- contact_sheets/*.jpg

Skrip ini hanya mengaudit dan TIDAK menghapus gambar.
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
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
    "testing": "test",
}


@dataclass
class ImageRecord:
    path: str
    relative_path: str
    split: str
    label: str
    width: int
    height: int
    aspect_ratio: float
    file_size_bytes: int
    sha256: str
    dhash64: str
    brightness_mean: float
    edge_variance: float
    is_corrupt: bool
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dataset citra klasifikasi postur."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Folder utama dataset yang sudah diekstrak.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_audit_output"),
        help="Folder output audit.",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=5,
        help="Batas Hamming distance dHash untuk near duplicate. Default: 5.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=25,
        help="Jumlah contoh gambar pada contact sheet per kelas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pemilihan sampel.",
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=45.0,
        help="Brightness mean di bawah nilai ini ditandai gelap.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=120.0,
        help="Edge variance di bawah nilai ini ditandai berpotensi blur.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=224,
        help="Lebar minimum yang diharapkan.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=224,
        help="Tinggi minimum yang diharapkan.",
    )
    return parser.parse_args()


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def infer_split_and_label(path: Path, root: Path) -> tuple[str, str]:
    rel_parts = path.relative_to(root).parts
    split = "unspecified"

    split_index = None
    for idx, part in enumerate(rel_parts[:-1]):
        normalized = SPLIT_ALIASES.get(part.lower())
        if normalized:
            split = normalized
            split_index = idx
            break

    # Untuk struktur train/class/image.jpg
    if split_index is not None and split_index + 1 < len(rel_parts) - 1:
        label = rel_parts[split_index + 1]
    else:
        # Untuk struktur class/image.jpg
        label = path.parent.name

    return split, label


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
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

    # Variansi edge sebagai indikator kasar blur.
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = float(np.asarray(edges, dtype=np.float32).var())
    return brightness, edge_variance


def inspect_image(path: Path, root: Path) -> ImageRecord:
    split, label = infer_split_and_label(path, root)
    relative_path = str(path.relative_to(root))

    try:
        file_hash = sha256_file(path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            brightness, edge_variance = image_metrics(image)
            dhash_value = dhash64(image)

        return ImageRecord(
            path=str(path.resolve()),
            relative_path=relative_path,
            split=split,
            label=label,
            width=width,
            height=height,
            aspect_ratio=round(width / height, 6) if height else 0.0,
            file_size_bytes=path.stat().st_size,
            sha256=file_hash,
            dhash64=f"{dhash_value:016x}",
            brightness_mean=round(brightness, 4),
            edge_variance=round(edge_variance, 4),
            is_corrupt=False,
            error="",
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageRecord(
            path=str(path.resolve()),
            relative_path=relative_path,
            split=split,
            label=label,
            width=0,
            height=0,
            aspect_ratio=0.0,
            file_size_bytes=path.stat().st_size if path.exists() else 0,
            sha256="",
            dhash64="",
            brightness_mean=0.0,
            edge_variance=0.0,
            is_corrupt=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_exact_duplicates(records: list[ImageRecord]) -> list[dict]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if not record.is_corrupt and record.sha256:
            groups[record.sha256].append(record)

    rows: list[dict] = []
    group_id = 0
    for file_hash, group in groups.items():
        if len(group) < 2:
            continue
        group_id += 1
        for record in group:
            rows.append(
                {
                    "duplicate_group": group_id,
                    "sha256": file_hash,
                    "split": record.split,
                    "label": record.label,
                    "relative_path": record.relative_path,
                }
            )
    return rows


def find_near_duplicates(
    records: list[ImageRecord],
    max_distance: int,
) -> list[dict]:
    valid = [record for record in records if not record.is_corrupt and record.dhash64]
    hashes = [int(record.dhash64, 16) for record in valid]

    # LSH sederhana: 4 band x 16 bit.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, hash_value in enumerate(hashes):
        for band in range(4):
            segment = (hash_value >> (band * 16)) & 0xFFFF
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
        record_a = valid[idx_a]
        record_b = valid[idx_b]

        # Exact duplicate sudah memiliki laporan terpisah.
        if record_a.sha256 == record_b.sha256:
            continue

        distance = hamming_distance64(hashes[idx_a], hashes[idx_b])
        if distance <= max_distance:
            rows.append(
                {
                    "hamming_distance": distance,
                    "same_split": record_a.split == record_b.split,
                    "same_label": record_a.label == record_b.label,
                    "split_a": record_a.split,
                    "label_a": record_a.label,
                    "path_a": record_a.relative_path,
                    "split_b": record_b.split,
                    "label_b": record_b.label,
                    "path_b": record_b.relative_path,
                }
            )

    rows.sort(key=lambda row: row["hamming_distance"])
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
            leakage.append(
                {
                    "type": "exact_duplicate",
                    "distance": 0,
                    "split_a": group[0]["split"],
                    "label_a": group[0]["label"],
                    "path_a": group[0]["relative_path"],
                    "split_b": group[1]["split"],
                    "label_b": group[1]["label"],
                    "path_b": group[1]["relative_path"],
                    "note": f"Exact duplicate group {group_id} muncul di {sorted(splits)}",
                }
            )

    for row in near_duplicates:
        if not row["same_split"]:
            leakage.append(
                {
                    "type": "near_duplicate",
                    "distance": row["hamming_distance"],
                    "split_a": row["split_a"],
                    "label_a": row["label_a"],
                    "path_a": row["path_a"],
                    "split_b": row["split_b"],
                    "label_b": row["label_b"],
                    "path_b": row["path_b"],
                    "note": "Near duplicate muncul pada split berbeda",
                }
            )

    return leakage


def create_contact_sheets(
    records: list[ImageRecord],
    root: Path,
    output_dir: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records:
        if not record.is_corrupt:
            grouped[record.label].append(record)

    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    thumb_size = (180, 180)
    label_height = 38
    columns = 5

    for label, group in sorted(grouped.items()):
        sample = rng.sample(group, min(samples_per_class, len(group)))
        rows_count = math.ceil(len(sample) / columns)
        sheet = Image.new(
            "RGB",
            (columns * thumb_size[0], rows_count * (thumb_size[1] + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for idx, record in enumerate(sample):
            image_path = root / record.relative_path
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail(thumb_size, Image.Resampling.LANCZOS)

                x = (idx % columns) * thumb_size[0]
                y = (idx // columns) * (thumb_size[1] + label_height)
                paste_x = x + (thumb_size[0] - image.width) // 2
                paste_y = y + (thumb_size[1] - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))

                text = f"{record.split}: {Path(record.relative_path).name}"
                draw.text((x + 3, y + thumb_size[1] + 3), text[:27], fill="black")

        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        sheet.save(sheet_dir / f"{safe_label}.jpg", quality=90)


def summarize(
    records: list[ImageRecord],
    exact_duplicates: list[dict],
    near_duplicates: list[dict],
    leakage: list[dict],
    args: argparse.Namespace,
) -> dict:
    valid = [record for record in records if not record.is_corrupt]
    corrupt = [record for record in records if record.is_corrupt]

    class_counts = Counter(record.label for record in valid)
    split_counts = Counter(record.split for record in valid)
    split_class_counts = Counter((record.split, record.label) for record in valid)
    resolution_counts = Counter((record.width, record.height) for record in valid)

    dark_count = sum(record.brightness_mean < args.dark_threshold for record in valid)
    blur_count = sum(record.edge_variance < args.blur_threshold for record in valid)
    low_resolution_count = sum(
        record.width < args.min_width or record.height < args.min_height
        for record in valid
    )
    unlabeled_count = sum(
        record.label.lower() in {"unlabeled", "unlabelled", "unknown", "none"}
        for record in valid
    )

    exact_group_count = len({row["duplicate_group"] for row in exact_duplicates})
    exact_file_count = len(exact_duplicates)

    class_distribution = [
        {
            "split": split,
            "label": label,
            "count": count,
        }
        for (split, label), count in sorted(split_class_counts.items())
    ]

    top_resolutions = [
        {"width": width, "height": height, "count": count}
        for (width, height), count in resolution_counts.most_common(10)
    ]

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "total_files_detected": len(records),
        "valid_images": len(valid),
        "corrupt_images": len(corrupt),
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "class_distribution_by_split": class_distribution,
        "top_resolutions": top_resolutions,
        "dark_images_below_threshold": dark_count,
        "dark_threshold": args.dark_threshold,
        "potentially_blurry_images_below_threshold": blur_count,
        "blur_threshold": args.blur_threshold,
        "low_resolution_images": low_resolution_count,
        "minimum_expected_resolution": {
            "width": args.min_width,
            "height": args.min_height,
        },
        "unlabeled_images": unlabeled_count,
        "exact_duplicate_groups": exact_group_count,
        "files_in_exact_duplicate_groups": exact_file_count,
        "near_duplicate_pairs": len(near_duplicates),
        "cross_split_leakage_pairs": len(leakage),
        "warnings": [],
    }

    if corrupt:
        summary["warnings"].append("Terdapat citra rusak atau tidak dapat dibaca.")
    if unlabeled_count:
        summary["warnings"].append("Terdapat kelas Unlabeled/Unknown yang perlu ditinjau.")
    if leakage:
        summary["warnings"].append(
            "Terdapat exact/near duplicate pada split berbeda; ini berpotensi data leakage."
        )
    if len(class_counts) > 1:
        counts = list(class_counts.values())
        imbalance_ratio = max(counts) / min(counts)
        summary["class_imbalance_ratio_max_to_min"] = round(imbalance_ratio, 4)
        if imbalance_ratio >= 2:
            summary["warnings"].append(
                "Rasio kelas terbesar:terkecil >= 2; pertimbangkan strategi imbalance."
            )
    else:
        summary["class_imbalance_ratio_max_to_min"] = None

    return summary


def write_summary_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# Ringkasan Audit Dataset",
        "",
        f"- Folder dataset: `{summary['dataset_dir']}`",
        f"- Total file citra terdeteksi: **{summary['total_files_detected']}**",
        f"- Citra valid: **{summary['valid_images']}**",
        f"- Citra rusak: **{summary['corrupt_images']}**",
        f"- Jumlah kelas: **{summary['class_count']}**",
        f"- Citra Unlabeled/Unknown: **{summary['unlabeled_images']}**",
        f"- Grup exact duplicate: **{summary['exact_duplicate_groups']}**",
        f"- Pasangan near duplicate: **{summary['near_duplicate_pairs']}**",
        f"- Potensi leakage lintas split: **{summary['cross_split_leakage_pairs']}**",
        "",
        "## Distribusi Kelas",
        "",
        "| Kelas | Jumlah |",
        "|---|---:|",
    ]

    for label, count in summary["class_counts"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Distribusi Split",
            "",
            "| Split | Jumlah |",
            "|---|---:|",
        ]
    )
    for split, count in summary["split_counts"].items():
        lines.append(f"| {split} | {count} |")

    lines.extend(
        [
            "",
            "## Pemeriksaan Kualitas",
            "",
            f"- Citra gelap: **{summary['dark_images_below_threshold']}** "
            f"(threshold brightness < {summary['dark_threshold']})",
            f"- Citra berpotensi blur: **{summary['potentially_blurry_images_below_threshold']}** "
            f"(threshold edge variance < {summary['blur_threshold']})",
            f"- Citra di bawah resolusi minimum: **{summary['low_resolution_images']}**",
            "",
            "## Peringatan",
            "",
        ]
    )

    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- Tidak ada peringatan otomatis utama.")

    lines.extend(
        [
            "",
            "## Catatan",
            "",
            "Hasil blur, brightness, dan near duplicate adalah indikator awal. "
            "Keputusan penghapusan gambar tetap harus melalui pemeriksaan visual.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"ERROR: Folder dataset tidak ditemukan: {dataset_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(iter_image_paths(dataset_dir))

    if not image_paths:
        print("ERROR: Tidak ada file citra ditemukan.", file=sys.stderr)
        return 1

    print(f"Menemukan {len(image_paths)} file citra.")
    records: list[ImageRecord] = []

    for idx, path in enumerate(image_paths, start=1):
        records.append(inspect_image(path, dataset_dir))
        if idx % 250 == 0 or idx == len(image_paths):
            print(f"Memeriksa {idx}/{len(image_paths)} gambar...")

    record_rows = [asdict(record) for record in records]
    write_csv(
        output_dir / "images_audit.csv",
        record_rows,
        fieldnames=list(ImageRecord.__dataclass_fields__.keys()),
    )

    valid_records = [record for record in records if not record.is_corrupt]

    class_counter = Counter((record.split, record.label) for record in valid_records)
    class_rows = [
        {"split": split, "label": label, "count": count}
        for (split, label), count in sorted(class_counter.items())
    ]
    write_csv(
        output_dir / "class_distribution.csv",
        class_rows,
        fieldnames=["split", "label", "count"],
    )

    print("Mencari exact duplicate...")
    exact_duplicates = find_exact_duplicates(records)
    write_csv(
        output_dir / "exact_duplicates.csv",
        exact_duplicates,
        fieldnames=[
            "duplicate_group",
            "sha256",
            "split",
            "label",
            "relative_path",
        ],
    )

    print("Mencari near duplicate...")
    near_duplicates = find_near_duplicates(
        records,
        max_distance=args.near_duplicate_distance,
    )
    write_csv(
        output_dir / "near_duplicates.csv",
        near_duplicates,
        fieldnames=[
            "hamming_distance",
            "same_split",
            "same_label",
            "split_a",
            "label_a",
            "path_a",
            "split_b",
            "label_b",
            "path_b",
        ],
    )

    leakage = build_split_leakage(exact_duplicates, near_duplicates)
    write_csv(
        output_dir / "split_leakage.csv",
        leakage,
        fieldnames=[
            "type",
            "distance",
            "split_a",
            "label_a",
            "path_a",
            "split_b",
            "label_b",
            "path_b",
            "note",
        ],
    )

    print("Membuat contact sheet...")
    create_contact_sheets(
        records,
        dataset_dir,
        output_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    summary = summarize(
        records,
        exact_duplicates,
        near_duplicates,
        leakage,
        args,
    )

    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_markdown(summary, output_dir / "audit_summary.md")

    print("")
    print("Audit selesai.")
    print(f"Output: {output_dir}")
    print(f"Citra valid: {summary['valid_images']}")
    print(f"Citra rusak: {summary['corrupt_images']}")
    print(f"Exact duplicate groups: {summary['exact_duplicate_groups']}")
    print(f"Near duplicate pairs: {summary['near_duplicate_pairs']}")
    print(f"Cross-split leakage: {summary['cross_split_leakage_pairs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())