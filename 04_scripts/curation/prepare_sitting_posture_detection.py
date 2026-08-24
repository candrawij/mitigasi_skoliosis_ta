"""
Prepare Sitting Posture Detection Initial dataset for experiments.

Based on revised plan contact sheet audit:
- Dataset is heterogeneous (real photos, stock images, illustrations, composites)
- Needs curation: filter by quality, handle near-duplicates, group-aware split
- Format: COCO Detection (bounding box annotations)

This script:
1. Parses COCO detection annotations from all splits
2. Groups near-duplicates using audit data
3. Flags quality issues (no annotation, dark images)
4. Creates group-aware split to prevent leakage
5. Exports curated metadata for downstream use
"""
import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

DATASET_ROOT = Path(r"D:\.Candra\Project\TA\02_data\raw\Sitting Posture Detection.v2i.coco")
AUDIT_DIR = Path(r"D:\.Candra\Project\TA\07_results\dataset_audit\sitting_posture_detection_initial")
OUTPUT_DIR = Path(r"D:\.Candra\Project\TA\02_data\interim\sitting_posture_detection_curated")
SPLIT_OUTPUT = Path(r"D:\.Candra\Project\TA\03_metadata\final_split\sitting_posture_detection.csv")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def build_near_duplicate_groups():
    """Build groups from near_duplicates.csv."""
    nd_file = AUDIT_DIR / "near_duplicates.csv"
    if not nd_file.exists():
        return {}

    adjacency = defaultdict(set)
    with open(nd_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_a = row.get("filename_a", "")
            img_b = row.get("filename_b", "")
            if img_a and img_b:
                adjacency[img_a].add(img_b)
                adjacency[img_b].add(img_a)

    visited = set()
    groups = {}
    group_id = 0
    for node in adjacency:
        if node in visited:
            continue
        queue = [node]
        component = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        gid = f"ND_{group_id:04d}"
        for member in component:
            groups[member] = gid
        group_id += 1
    return groups


def load_image_audit():
    """Load image audit data for quality flags."""
    audit_file = AUDIT_DIR / "image_audit.csv"
    if not audit_file.exists():
        return {}
    audit = {}
    with open(audit_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "")
            if fname:
                audit[fname] = row
    return audit


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    nd_groups = build_near_duplicate_groups()
    print(f"Near-duplicate groups: {len(set(nd_groups.values()))}")
    print(f"Images in ND groups: {len(nd_groups)}")

    image_audit = load_image_audit()
    print(f"Image audit entries: {len(image_audit)}")

    all_records = []
    no_annotation = []

    for split_name in ["train", "valid", "test"]:
        split_dir = DATASET_ROOT / split_name
        ann_file = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"  No annotations found for {split_name}")
            continue

        with open(ann_file, "r") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco.get("images", [])}
        annotations = coco.get("annotations", [])
        categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

        # Group annotations by image
        img_anns = defaultdict(list)
        for ann in annotations:
            img_anns[ann["image_id"]].append(ann)

        for img_id, img_info in images.items():
            filename = img_info["file_name"]
            img_w = img_info.get("width", 1)
            img_h = img_info.get("height", 1)
            anns = img_anns.get(img_id, [])

            if not anns:
                no_annotation.append(filename)

            # Get quality flags from audit
            audit_info = image_audit.get(filename, {})
            brightness = float(audit_info.get("brightness_mean", 128))
            is_dark = brightness < 45.0

            nd_group = nd_groups.get(filename, f"SINGLE_{filename}")

            # Process each annotation (bbox + class)
            for ann in anns:
                cat_name = categories.get(ann["category_id"], "unknown")
                bbox = ann.get("bbox", [0, 0, 0, 0])  # COCO: [x, y, w, h]

                record = {
                    "image_id": filename,
                    "original_split": split_name,
                    "class_name": cat_name,
                    "nd_group": nd_group,
                    "img_width": img_w,
                    "img_height": img_h,
                    "bbox_x": round(bbox[0] / img_w, 6) if img_w else 0,
                    "bbox_y": round(bbox[1] / img_h, 6) if img_h else 0,
                    "bbox_w": round(bbox[2] / img_w, 6) if img_w else 0,
                    "bbox_h": round(bbox[3] / img_h, 6) if img_h else 0,
                    "brightness": round(brightness, 2),
                    "is_dark": is_dark,
                    "has_annotation": len(anns) > 0,
                    "num_annotations": len(anns),
                    "annotation_id": ann["id"],
                }
                all_records.append(record)

        # Also add images without annotations
        for img_id, img_info in images.items():
            if img_id not in img_anns or not img_anns[img_id]:
                filename = img_info["file_name"]
                nd_group = nd_groups.get(filename, f"SINGLE_{filename}")
                audit_info = image_audit.get(filename, {})
                brightness = float(audit_info.get("brightness_mean", 128))

                record = {
                    "image_id": filename,
                    "original_split": split_name,
                    "class_name": "no_annotation",
                    "nd_group": nd_group,
                    "img_width": img_info.get("width", 0),
                    "img_height": img_info.get("height", 0),
                    "bbox_x": 0, "bbox_y": 0, "bbox_w": 0, "bbox_h": 0,
                    "brightness": round(brightness, 2),
                    "is_dark": brightness < 45.0,
                    "has_annotation": False,
                    "num_annotations": 0,
                    "annotation_id": -1,
                }
                all_records.append(record)

    print(f"\nTotal annotation records: {len(all_records)}")
    print(f"Images without annotation: {len(no_annotation)}")

    # Group-level split (using unique images, not annotations)
    # Get unique images
    unique_images = {}
    for rec in all_records:
        if rec["image_id"] not in unique_images:
            unique_images[rec["image_id"]] = rec

    image_groups = defaultdict(list)
    for img_id, rec in unique_images.items():
        image_groups[rec["nd_group"]].append(rec)

    print(f"Total unique images: {len(unique_images)}")
    print(f"Total groups: {len(image_groups)}")

    # Balanced split
    random.seed(RANDOM_SEED)
    group_list = list(image_groups.items())
    random.shuffle(group_list)
    group_list.sort(key=lambda x: len(x[1]), reverse=True)

    total_images = len(unique_images)
    target_train = int(total_images * TRAIN_RATIO)
    target_valid = int(total_images * VALID_RATIO)
    target_test = total_images - target_train - target_valid

    new_split = {}
    buckets = {"train": 0, "valid": 0, "test": 0}
    targets = {"train": target_train, "valid": target_valid, "test": target_test}

    for gid, members in group_list:
        size = len(members)
        remaining = {s: max(0, targets[s] - buckets[s]) for s in ["train", "valid", "test"]}
        total_remaining = sum(remaining.values())
        if total_remaining == 0:
            split = "train"
        else:
            split = max(remaining, key=remaining.get)
        buckets[split] += size
        for rec in members:
            new_split[rec["image_id"]] = split

    # Apply split to all records
    for rec in all_records:
        rec["split"] = new_split.get(rec["image_id"], "train")

    # Save full metadata CSV
    output_csv = OUTPUT_DIR / "sitting_posture_detection_metadata.csv"
    fieldnames = list(all_records[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nSaved metadata to {output_csv}")

    # Save split manifest (unique images only)
    with open(SPLIT_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split", "nd_group", "original_split", "num_annotations", "is_dark"])
        writer.writeheader()
        for img_id, rec in sorted(unique_images.items()):
            writer.writerow({
                "image_id": img_id,
                "split": new_split.get(img_id, "train"),
                "nd_group": rec["nd_group"],
                "original_split": rec["original_split"],
                "num_annotations": rec["num_annotations"],
                "is_dark": rec["is_dark"],
            })
    print(f"Saved split manifest to {SPLIT_OUTPUT}")

    # Summary
    split_counts = Counter(new_split.values())
    class_counts = Counter(r["class_name"] for r in all_records if r["has_annotation"])

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique images: {len(unique_images)}")
    for s in ["train", "valid", "test"]:
        print(f"  {s}: {split_counts.get(s, 0)}")

    print(f"\nAnnotation class distribution:")
    for c, cnt in sorted(class_counts.items()):
        print(f"  {c}: {cnt}")

    print(f"\nQuality flags:")
    dark_count = sum(1 for r in unique_images.values() if r["is_dark"])
    no_ann_count = sum(1 for r in unique_images.values() if not r["has_annotation"])
    multi_ann = sum(1 for r in unique_images.values() if r["num_annotations"] > 1)
    print(f"  Dark images: {dark_count}")
    print(f"  No annotation: {no_ann_count}")
    print(f"  Multi-annotation: {multi_ann}")

    group_splits = defaultdict(set)
    for img_id, s in new_split.items():
        g = unique_images[img_id]["nd_group"]
        group_splits[g].add(s)
    leakage = {g: s for g, s in group_splits.items() if len(s) > 1}
    print(f"  Leakage (groups in multiple splits): {len(leakage)}")

    # Save summary
    summary = {
        "total_images": len(unique_images),
        "total_annotations": len([r for r in all_records if r["has_annotation"]]),
        "splits": dict(split_counts),
        "classes": dict(class_counts),
        "dark_images": dark_count,
        "no_annotation": no_ann_count,
        "multi_annotation_images": multi_ann,
        "nd_groups": len(image_groups),
        "leakage_groups": len(leakage),
    }
    with open(OUTPUT_DIR / "preparation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
