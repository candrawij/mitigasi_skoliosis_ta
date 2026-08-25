"""
Prepare IKORN 4-Keypoint dataset for experiments.

Parses COCO-Pose annotations, extracts 4 keypoints (bottom, shoulder,
head, back), groups near-duplicates, creates a clean split at group
level, and exports tabular features for MLP classifier.

Key decisions:
  - Near-duplicates are grouped first
  - Split is done at group level (not image level) to prevent leakage
  - Deduplication within groups is done by keeping representatives
"""
import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "02_data" / "raw" / "sitting posture.v4-sitting_posture_4keypoint.coco"
AUDIT_DIR = PROJECT_ROOT / "07_results" / "dataset_audit" / "sitting_posture_4kp_initial"
OUTPUT_DIR = PROJECT_ROOT / "02_data" / "interim" / "ikorn_4kp_keypoints"
SPLIT_OUTPUT = PROJECT_ROOT / "03_metadata" / "final_split" / "ikorn_4kp.csv"

KEYPOINT_NAMES = ["bottom", "shoulder", "head", "back"]

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def load_coco_annotations(split_dir: Path):
    """Load COCO annotations from a split directory."""
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        return [], [], [], {}

    with open(ann_file, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    # Extract keypoint names from categories
    kp_names = []
    for cat in coco.get("categories", []):
        if "keypoints" in cat:
            kp_names = cat["keypoints"]
            break

    return images, annotations, kp_names, categories


def build_near_duplicate_groups():
    """Build groups from near_duplicates.csv for group-level splitting."""
    nd_file = AUDIT_DIR / "near_duplicates.csv"
    if not nd_file.exists():
        return {}

    # Build adjacency list
    adjacency = defaultdict(set)
    with open(nd_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_a = row.get("filename_a", "")
            img_b = row.get("filename_b", "")
            if img_a and img_b:
                adjacency[img_a].add(img_b)
                adjacency[img_b].add(img_a)

    # Connected components via BFS
    visited = set()
    groups = {}
    group_id = 0

    for node in adjacency:
        if node in visited:
            continue
        # BFS
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

        gid = f"ND_GROUP_{group_id:04d}"
        for member in component:
            groups[member] = gid
        group_id += 1

    return groups


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Build near-duplicate groups
    nd_groups = build_near_duplicate_groups()
    print(f"Near-duplicate groups found: {len(set(nd_groups.values()))}")
    print(f"Images in groups: {len(nd_groups)}")

    all_records = []
    split_map = {}  # image_filename → original_split

    for split_name in ["train", "valid", "test"]:
        split_dir = DATASET_ROOT / split_name
        if not split_dir.exists():
            continue

        images, annotations, kp_names, categories = load_coco_annotations(split_dir)

        if kp_names and kp_names != KEYPOINT_NAMES:
            print(f"  WARNING: keypoint names from annotations: {kp_names}")
            print(f"  Expected: {KEYPOINT_NAMES}")

        for ann in annotations:
            img_info = images.get(ann["image_id"])
            if not img_info:
                continue

            filename = img_info["file_name"]
            cat_name = categories.get(ann["category_id"], "unknown")
            img_w = img_info.get("width", 1)
            img_h = img_info.get("height", 1)

            # Parse keypoints (COCO format: x, y, vis, x, y, vis, ...)
            raw_kp = ann.get("keypoints", [])
            keypoints = []
            for i in range(4):
                idx = i * 3
                if idx + 2 < len(raw_kp):
                    kx = raw_kp[idx] / img_w  # normalize
                    ky = raw_kp[idx + 1] / img_h
                    vis = int(raw_kp[idx + 2])
                    keypoints.append((kx, ky, vis))
                else:
                    keypoints.append((0, 0, 0))

            # Determine group
            nd_group = nd_groups.get(filename, f"SINGLE_{filename}")

            record = {
                "image_id": filename,
                "original_split": split_name,
                "class_name": cat_name,
                "nd_group": nd_group,
                "img_width": img_w,
                "img_height": img_h,
                "bbox_x": ann.get("bbox", [0])[0] / img_w if ann.get("bbox") else 0,
                "bbox_y": ann.get("bbox", [0, 0])[1] / img_h if ann.get("bbox") else 0,
                "bbox_w": ann.get("bbox", [0, 0, 0])[2] / img_w if ann.get("bbox") else 0,
                "bbox_h": ann.get("bbox", [0, 0, 0, 0])[3] / img_h if ann.get("bbox") else 0,
            }

            for kp_idx, (kx, ky, vis) in enumerate(keypoints):
                kp_name = KEYPOINT_NAMES[kp_idx]
                record[f"{kp_name}_x"] = round(kx, 6)
                record[f"{kp_name}_y"] = round(ky, 6)
                record[f"{kp_name}_vis"] = vis

            all_records.append(record)
            split_map[filename] = split_name

    print(f"\nTotal records parsed: {len(all_records)}")

    # Group-level split
    # Collect groups
    group_images = defaultdict(list)
    for rec in all_records:
        group_images[rec["nd_group"]].append(rec)

    print(f"Total groups (including singletons): {len(group_images)}")

    # Shuffle and assign
    random.seed(RANDOM_SEED)
    group_list = list(group_images.items())
    # Shuffle and sort by group size descending for better distribution
    random.seed(RANDOM_SEED)
    random.shuffle(group_list)
    # Sort by size descending to fill buckets more evenly
    group_list.sort(key=lambda x: len(x[1]), reverse=True)

    total_images = len(all_records)
    target_train = int(total_images * TRAIN_RATIO)
    target_valid = int(total_images * VALID_RATIO)
    target_test = total_images - target_train - target_valid

    new_split = {}
    buckets = {"train": 0, "valid": 0, "test": 0}
    targets = {"train": target_train, "valid": target_valid, "test": target_test}

    for gid, members in group_list:
        size = len(members)
        # Assign to the bucket that is most under-target
        # Calculate remaining capacity ratio for each bucket
        remaining = {s: max(0, targets[s] - buckets[s]) for s in ["train", "valid", "test"]}
        total_remaining = sum(remaining.values())
        if total_remaining == 0:
            split = "train"  # fallback
        else:
            # Pick the split with the most remaining capacity
            split = max(remaining, key=remaining.get)
        
        buckets[split] += size

        for rec in members:
            new_split[rec["image_id"]] = split

    # Apply new split
    for rec in all_records:
        rec["split"] = new_split.get(rec["image_id"], "train")

    # Save keypoint CSV
    output_csv = OUTPUT_DIR / "ikorn_4kp_keypoints.csv"
    fieldnames = list(all_records[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nSaved keypoint data to {output_csv}")

    # Save split manifest
    with open(SPLIT_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split", "class_name", "nd_group", "original_split"])
        writer.writeheader()
        for r in all_records:
            writer.writerow({
                "image_id": r["image_id"],
                "split": r["split"],
                "class_name": r["class_name"],
                "nd_group": r["nd_group"],
                "original_split": r["original_split"],
            })

    print(f"Saved split manifest to {SPLIT_OUTPUT}")

    # Summary
    split_counts = Counter(r["split"] for r in all_records)
    class_counts = Counter(r["class_name"] for r in all_records)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(all_records)}")
    for s in ["train", "valid", "test"]:
        print(f"  {s}: {split_counts[s]}")
    print(f"\nPer class:")
    for c, cnt in sorted(class_counts.items()):
        print(f"  {c}: {cnt}")
    print(f"\nPer split × class:")
    for s in ["train", "valid", "test"]:
        for c in sorted(class_counts.keys()):
            cnt = sum(1 for r in all_records if r["split"] == s and r["class_name"] == c)
            print(f"  {s}/{c}: {cnt}")

    # Leakage check
    group_splits = defaultdict(set)
    for r in all_records:
        group_splits[r["nd_group"]].add(r["split"])

    leakage = {g: s for g, s in group_splits.items() if len(s) > 1}
    print(f"\nLeakage check: {len(leakage)} groups span multiple splits")

    # Save summary
    summary = {
        "total": len(all_records),
        "splits": dict(split_counts),
        "classes": dict(class_counts),
        "keypoint_names": KEYPOINT_NAMES,
        "nd_groups_total": len(group_images),
        "leakage_groups": len(leakage),
        "split_method": "group-level (near-duplicate aware)",
    }
    with open(OUTPUT_DIR / "preparation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
