#!/usr/bin/env python3
"""
Membangun metadata kurasi dan cluster near-duplicate dari hasil audit dataset.

Skrip ini:
1. Membaca images_audit.csv dan near_duplicates.csv.
2. Mengelompokkan gambar yang sangat mirip dengan Union-Find.
3. Membuat curation_master.csv untuk audit manual.
4. Membuat duplicate_clusters.csv dan priority_review.csv.
5. Membuat contact sheet untuk cluster prioritas.
6. TIDAK menghapus, memindahkan, atau mengubah gambar.

Disarankan dijalankan dari root proyek TA.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


@dataclass
class ImageInfo:
    relative_path: str
    absolute_path: str
    split: str
    label: str
    width: int
    height: int
    edge_variance: float
    brightness_mean: float
    is_corrupt: bool


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bangun metadata kurasi dan cluster near-duplicate."
    )
    parser.add_argument(
        "--images-audit",
        type=Path,
        required=True,
        help="Path ke images_audit.csv.",
    )
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        required=True,
        help="Path ke near_duplicates.csv.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root dataset mentah. Jika tidak diisi, skrip mencoba memakai "
            "kolom path dari images_audit.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder output metadata kurasi.",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=2,
        help=(
            "Maksimum Hamming distance yang dipakai untuk clustering. "
            "Mulai dari 2 agar cluster tidak terlalu agresif."
        ),
    )
    parser.add_argument(
        "--max-contact-sheets",
        type=int,
        default=250,
        help="Maksimum jumlah contact sheet cluster prioritas.",
    )
    parser.add_argument(
        "--max-images-per-sheet",
        type=int,
        default=30,
        help="Maksimum jumlah gambar per contact sheet.",
    )
    parser.add_argument(
        "--include-same-split",
        action="store_true",
        help=(
            "Sertakan cluster yang hanya ada pada satu split dalam priority_review. "
            "Default hanya cluster mixed-label atau cross-split."
        ),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_path_text(value: str) -> str:
    return str(Path(value.replace("\\", "/"))).replace("\\", "/")


def load_images(path: Path, dataset_root: Path | None) -> dict[str, ImageInfo]:
    rows = read_csv_rows(path)
    images: dict[str, ImageInfo] = {}

    for row in rows:
        relative_path = normalize_path_text(row.get("relative_path", ""))
        if not relative_path:
            continue

        if dataset_root is not None:
            absolute_path = str((dataset_root / Path(relative_path)).resolve())
        else:
            absolute_path = row.get("path", "").strip()

        images[relative_path] = ImageInfo(
            relative_path=relative_path,
            absolute_path=absolute_path,
            split=row.get("split", "unspecified"),
            label=row.get("label", "unknown"),
            width=safe_int(row.get("width", "0")),
            height=safe_int(row.get("height", "0")),
            edge_variance=safe_float(row.get("edge_variance", "0")),
            brightness_mean=safe_float(row.get("brightness_mean", "0")),
            is_corrupt=parse_bool(row.get("is_corrupt", "false")),
        )

    return images


def resolve_pair_path(raw_value: str, images: dict[str, ImageInfo]) -> str | None:
    normalized = normalize_path_text(raw_value)
    if normalized in images:
        return normalized

    # Kadang CSV menyimpan path absolut atau format separator berbeda.
    normalized_lower = normalized.lower()
    for relative_path in images:
        if normalized_lower.endswith(relative_path.lower()):
            return relative_path
    return None


def build_clusters(
    images: dict[str, ImageInfo],
    near_duplicate_rows: list[dict[str, str]],
    max_distance: int,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], int], int]:
    uf = UnionFind(images.keys())
    edge_distances: dict[tuple[str, str], int] = {}
    accepted_edges = 0

    for row in near_duplicate_rows:
        distance = safe_int(row.get("hamming_distance", "999"), 999)
        if distance > max_distance:
            continue

        path_a = resolve_pair_path(row.get("path_a", ""), images)
        path_b = resolve_pair_path(row.get("path_b", ""), images)
        if path_a is None or path_b is None or path_a == path_b:
            continue

        pair = tuple(sorted((path_a, path_b)))
        current = edge_distances.get(pair)
        if current is None or distance < current:
            edge_distances[pair] = distance

        uf.union(path_a, path_b)
        accepted_edges += 1

    grouped: dict[str, list[str]] = defaultdict(list)
    for relative_path in images:
        grouped[uf.find(relative_path)].append(relative_path)

    clusters = {
        root: sorted(members)
        for root, members in grouped.items()
        if len(members) >= 2
    }
    return clusters, edge_distances, accepted_edges


def extract_sequence_hint(filename: str) -> str:
    """
    Mengambil pola seperti extract0584 atau mb20131 sebagai petunjuk awal.
    Ini bukan subject/session ID final.
    """
    stem = Path(filename).stem
    match = re.search(r"([A-Za-z_]+)(\d{3,})", stem)
    if not match:
        return ""
    prefix, number_text = match.groups()
    number = int(number_text)
    block = number // 25
    return f"{prefix.lower()}_block_{block:04d}"


def choose_representative(members: list[str], images: dict[str, ImageInfo]) -> str:
    """
    Pilih kandidat representatif otomatis:
    - tidak corrupt;
    - edge_variance lebih tinggi;
    - resolusi lebih besar.
    Pilihan ini harus tetap ditinjau manual.
    """
    ranked = sorted(
        members,
        key=lambda path: (
            images[path].is_corrupt,
            -images[path].edge_variance,
            -(images[path].width * images[path].height),
            path,
        ),
    )
    return ranked[0]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(
    cluster_id: str,
    members: list[str],
    images: dict[str, ImageInfo],
    output_path: Path,
    max_images: int,
) -> None:
    selected = members[:max_images]
    thumb_w, thumb_h = 220, 150
    text_h = 55
    cols = 4
    rows = math.ceil(len(selected) / cols)

    canvas = Image.new(
        "RGB",
        (cols * thumb_w, rows * (thumb_h + text_h) + 45),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{cluster_id} | size={len(members)}", fill="black")

    for index, relative_path in enumerate(selected):
        info = images[relative_path]
        x = (index % cols) * thumb_w
        y = 45 + (index // cols) * (thumb_h + text_h)

        try:
            with Image.open(info.absolute_path) as image:
                image = image.convert("RGB")
                fitted = ImageOps.contain(
                    image,
                    (thumb_w - 8, thumb_h - 8),
                    Image.Resampling.LANCZOS,
                )
                paste_x = x + (thumb_w - fitted.width) // 2
                paste_y = y + (thumb_h - fitted.height) // 2
                canvas.paste(fitted, (paste_x, paste_y))
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            draw.rectangle(
                (x + 5, y + 5, x + thumb_w - 5, y + thumb_h - 5),
                outline="black",
            )
            draw.text((x + 12, y + 60), "IMAGE ERROR", fill="black")

        text = (
            f"{info.split} | {info.label}\n"
            f"{Path(relative_path).name[:30]}"
        )
        draw.multiline_text((x + 4, y + thumb_h + 3), text, fill="black", spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def main() -> int:
    args = parse_args()

    try:
        images = load_images(args.images_audit, args.dataset_root)
        near_rows = read_csv_rows(args.near_duplicates)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not images:
        print("ERROR: Tidak ada data gambar pada images_audit.csv.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Jumlah gambar: {len(images)}")
    print(f"Jumlah pasangan near duplicate pada CSV: {len(near_rows)}")
    print(f"Clustering dengan max distance <= {args.max_distance} ...")

    clusters, edge_distances, accepted_edges = build_clusters(
        images,
        near_rows,
        args.max_distance,
    )

    sorted_clusters = sorted(
        clusters.values(),
        key=lambda members: (-len(members), members[0]),
    )

    cluster_id_by_path: dict[str, str] = {}
    cluster_summaries: list[dict] = []
    priority_summaries: list[dict] = []

    for index, members in enumerate(sorted_clusters, start=1):
        cluster_id = f"CL{index:05d}"
        labels = sorted({images[path].label for path in members})
        splits = sorted({images[path].split for path in members})
        mixed_label = len(labels) > 1
        cross_split = len(splits) > 1
        representative = choose_representative(members, images)

        distances = []
        member_set = set(members)
        for (path_a, path_b), distance in edge_distances.items():
            if path_a in member_set and path_b in member_set:
                distances.append(distance)

        min_distance = min(distances) if distances else ""
        max_distance = max(distances) if distances else ""
        mean_distance = (
            round(sum(distances) / len(distances), 3)
            if distances else ""
        )

        summary_row = {
            "cluster_id": cluster_id,
            "cluster_size": len(members),
            "labels": " | ".join(labels),
            "splits": " | ".join(splits),
            "mixed_label": mixed_label,
            "cross_split": cross_split,
            "edge_count": len(distances),
            "min_edge_distance": min_distance,
            "max_edge_distance": max_distance,
            "mean_edge_distance": mean_distance,
            "suggested_representative": representative,
            "review_priority": (
                "P1_mixed_label_cross_split"
                if mixed_label and cross_split
                else "P2_cross_split"
                if cross_split
                else "P3_mixed_label"
                if mixed_label
                else "P4_same_label_same_split"
            ),
        }
        cluster_summaries.append(summary_row)

        should_prioritize = mixed_label or cross_split or args.include_same_split
        if should_prioritize:
            priority_summaries.append(summary_row)

        for path in members:
            cluster_id_by_path[path] = cluster_id

    # Gambar singleton tetap masuk curation_master.
    all_rows: list[dict] = []
    representative_paths = {
        row["suggested_representative"] for row in cluster_summaries
    }

    summary_by_cluster = {
        row["cluster_id"]: row for row in cluster_summaries
    }

    for index, relative_path in enumerate(sorted(images), start=1):
        info = images[relative_path]
        cluster_id = cluster_id_by_path.get(relative_path, "")
        cluster_summary = summary_by_cluster.get(cluster_id, {})

        all_rows.append(
            {
                "image_id": f"IMG{index:05d}",
                "relative_path": relative_path,
                "original_path": info.absolute_path,
                "original_split": info.split,
                "original_label": info.label,
                "final_label": "",
                "subject_group": "",
                "session_group": "",
                "sequence_group": cluster_id or extract_sequence_hint(relative_path),
                "camera_view": "",
                "torso_visible": "",
                "shoulders_visible": "",
                "hips_visible": "",
                "label_quality": "",
                "decision": "",
                "notes": "",
                "cluster_id": cluster_id,
                "cluster_size": cluster_summary.get("cluster_size", 1),
                "cluster_mixed_label": cluster_summary.get("mixed_label", False),
                "cluster_cross_split": cluster_summary.get("cross_split", False),
                "suggested_representative": relative_path in representative_paths,
                "width": info.width,
                "height": info.height,
                "edge_variance": info.edge_variance,
                "brightness_mean": info.brightness_mean,
            }
        )

    write_csv(
        args.output_dir / "curation_master.csv",
        all_rows,
        fieldnames=list(all_rows[0].keys()),
    )

    cluster_fields = [
        "cluster_id",
        "cluster_size",
        "labels",
        "splits",
        "mixed_label",
        "cross_split",
        "edge_count",
        "min_edge_distance",
        "max_edge_distance",
        "mean_edge_distance",
        "suggested_representative",
        "review_priority",
    ]
    write_csv(
        args.output_dir / "duplicate_clusters.csv",
        cluster_summaries,
        fieldnames=cluster_fields,
    )
    write_csv(
        args.output_dir / "priority_review.csv",
        priority_summaries,
        fieldnames=cluster_fields,
    )

    representatives_rows = [
        {
            "cluster_id": row["cluster_id"],
            "cluster_size": row["cluster_size"],
            "suggested_representative": row["suggested_representative"],
            "review_priority": row["review_priority"],
            "manual_status": "",
            "manual_notes": "",
        }
        for row in cluster_summaries
    ]
    write_csv(
        args.output_dir / "representatives.csv",
        representatives_rows,
        fieldnames=[
            "cluster_id",
            "cluster_size",
            "suggested_representative",
            "review_priority",
            "manual_status",
            "manual_notes",
        ],
    )

    contact_sheet_dir = args.output_dir / "contact_sheets_priority"
    priority_sorted = sorted(
        priority_summaries,
        key=lambda row: (
            row["review_priority"],
            -int(row["cluster_size"]),
            row["cluster_id"],
        ),
    )

    generated = 0
    members_by_cluster = {
        f"CL{idx:05d}": members
        for idx, members in enumerate(sorted_clusters, start=1)
    }

    for row in priority_sorted[: args.max_contact_sheets]:
        cluster_id = row["cluster_id"]
        output_name = (
            f"{row['review_priority']}_{cluster_id}_"
            f"n{row['cluster_size']}.jpg"
        )
        make_contact_sheet(
            cluster_id,
            members_by_cluster[cluster_id],
            images,
            contact_sheet_dir / output_name,
            args.max_images_per_sheet,
        )
        generated += 1

    total_clustered_images = sum(len(members) for members in sorted_clusters)
    mixed_count = sum(bool(row["mixed_label"]) for row in cluster_summaries)
    cross_split_count = sum(bool(row["cross_split"]) for row in cluster_summaries)

    summary_text = f"""# Hasil Clustering dan Metadata Kurasi

- Total gambar: **{len(images)}**
- Threshold clustering: **Hamming distance <= {args.max_distance}**
- Edge/pasangan yang dipakai: **{accepted_edges}**
- Jumlah cluster (ukuran >= 2): **{len(sorted_clusters)}**
- Gambar yang masuk cluster: **{total_clustered_images}**
- Gambar singleton: **{len(images) - total_clustered_images}**
- Cluster mixed-label: **{mixed_count}**
- Cluster cross-split: **{cross_split_count}**
- Contact sheet prioritas dibuat: **{generated}**

## File utama

- `curation_master.csv`: lembar kerja audit manual seluruh gambar.
- `duplicate_clusters.csv`: ringkasan semua cluster.
- `priority_review.csv`: cluster yang harus diperiksa lebih dahulu.
- `representatives.csv`: kandidat gambar perwakilan tiap cluster.
- `contact_sheets_priority/`: visual cluster prioritas.

## Urutan audit manual

1. `P1_mixed_label_cross_split`
2. `P2_cross_split`
3. `P3_mixed_label`
4. `P4_same_label_same_split`

## Keputusan manual

Isi kolom pada `curation_master.csv`:

- `decision`: `KEEP`, `RELABEL`, atau `EXCLUDE`
- `final_label`: diisi jika `KEEP` atau `RELABEL`
- `subject_group`: contoh `S01`
- `session_group`: contoh `S01_SE01`
- `camera_view`: `frontal`, `diagonal`, atau `lateral`
- `torso_visible`, `shoulders_visible`, `hips_visible`: `yes`/`no`
- `label_quality`: `clear`, `ambiguous`, atau `wrong`
- `notes`: alasan keputusan

Skrip tidak mengubah dataset mentah.
"""
    (args.output_dir / "clustering_summary.md").write_text(
        summary_text,
        encoding="utf-8",
    )

    print("")
    print("Selesai.")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Cluster: {len(sorted_clusters)}")
    print(f"Mixed-label clusters: {mixed_count}")
    print(f"Cross-split clusters: {cross_split_count}")
    print(f"Contact sheets: {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())