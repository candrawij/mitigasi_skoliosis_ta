"""
Create clean train/valid/test split for Project Design 20242025.

Uses curated images and handles near-duplicate groups to avoid
leakage. If subject_group or session_group is available from
curation_master, splits at the group level.
"""
import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

CURATION_CSV = Path(r"D:\.Candra\Project\TA\03_metadata\curation\project_design_v1\curation_master_flagged.csv")
CURATED_DIR = Path(r"D:\.Candra\Project\TA\02_data\interim\project_design_curated")
SPLIT_OUTPUT = Path(r"D:\.Candra\Project\TA\03_metadata\final_split\project_design.csv")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def main():
    SPLIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Read curated data
    with open(CURATION_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to kept/relabeled images only
    decision_col = "decision"
    auto_col = "auto_decision"

    kept_rows = []
    for row in rows:
        manual = (row.get(decision_col) or "").strip().upper()
        auto = (row.get(auto_col) or "").strip().upper()
        effective = manual if manual else auto
        if effective in ("KEEP", "RELABEL"):
            final_label = row.get("final_label") or row["original_label"]
            row["_final_label"] = final_label
            kept_rows.append(row)

    print(f"Total curated images: {len(kept_rows)}")

    # Group by cluster_id to prevent leakage
    # Images in the same cluster should be in the same split
    cluster_groups = defaultdict(list)
    for row in kept_rows:
        cluster_id = row.get("cluster_id", "")
        # Singletons get unique group
        if not cluster_id or int(row.get("cluster_size", 1) or 1) <= 1:
            cluster_id = f"SINGLE_{row['image_id']}"
        cluster_groups[cluster_id].append(row)

    print(f"Total groups (clusters + singletons): {len(cluster_groups)}")

    # Check if subject/session groups are available
    has_subject = any(row.get("subject_group", "").strip() for row in kept_rows)
    if has_subject:
        print("Subject groups detected — splitting at subject level")
        # Group by subject
        subject_groups = defaultdict(list)
        for gid, members in cluster_groups.items():
            subj = members[0].get("subject_group", "").strip() or f"UNKNOWN_{gid}"
            subject_groups[subj].append((gid, members))

        split_units = list(subject_groups.items())
        split_key = "subject"
    else:
        print("No subject groups — splitting at cluster level (sequence-aware)")
        # Check for sequence_group
        has_sequence = any(row.get("sequence_group", "").strip() for row in kept_rows)
        if has_sequence:
            seq_groups = defaultdict(list)
            for gid, members in cluster_groups.items():
                seq = members[0].get("sequence_group", "").strip() or gid
                seq_groups[seq].append((gid, members))
            split_units = list(seq_groups.items())
            split_key = "sequence"
        else:
            # Fall back to cluster-level split
            split_units = [(gid, [(gid, members)]) for gid, members in cluster_groups.items()]
            split_key = "cluster"

    print(f"Split level: {split_key} ({len(split_units)} units)")

    # Stratified split by label distribution
    # Shuffle and assign
    random.seed(RANDOM_SEED)
    random.shuffle(split_units)

    # Count total images per unit
    unit_sizes = [(uid, sum(len(m) for _, m in unit_members), unit_members)
                  for uid, unit_members in split_units]

    total_images = sum(s for _, s, _ in unit_sizes)
    target_train = int(total_images * TRAIN_RATIO)
    target_valid = int(total_images * VALID_RATIO)

    # Assign splits
    assignments = {}
    current_train = 0
    current_valid = 0

    for uid, size, unit_members in unit_sizes:
        if current_train < target_train:
            split = "train"
            current_train += size
        elif current_valid < target_valid:
            split = "valid"
            current_valid += size
        else:
            split = "test"

        for gid, members in unit_members:
            for row in members:
                assignments[row["image_id"]] = split

    # Build output
    split_records = []
    for row in kept_rows:
        split = assignments.get(row["image_id"], "train")
        split_records.append({
            "image_id": row["image_id"],
            "split": split,
            "class_name": row["_final_label"],
            "original_split": row.get("original_split", ""),
            "cluster_id": row.get("cluster_id", ""),
            "filename": Path(row.get("relative_path", "")).name,
        })

    # Save
    with open(SPLIT_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=split_records[0].keys())
        writer.writeheader()
        writer.writerows(split_records)

    print(f"\nSaved split manifest to {SPLIT_OUTPUT}")

    # Summary
    split_counts = Counter(r["split"] for r in split_records)
    class_per_split = Counter((r["split"], r["class_name"]) for r in split_records)

    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(split_records)}")
    for s in ["train", "valid", "test"]:
        print(f"  {s}: {split_counts[s]}")

    print(f"\nPer split × class:")
    for s in ["train", "valid", "test"]:
        classes = sorted(set(r["class_name"] for r in split_records))
        for cls in classes:
            print(f"  {s}/{cls}: {class_per_split[(s, cls)]}")

    # Leakage check
    cluster_splits = defaultdict(set)
    for r in split_records:
        cid = r["cluster_id"]
        if cid:
            cluster_splits[cid].add(r["split"])

    leakage_clusters = {cid: splits for cid, splits in cluster_splits.items() if len(splits) > 1}
    print(f"\nLeakage check: {len(leakage_clusters)} clusters span multiple splits")
    if leakage_clusters:
        for cid, splits in list(leakage_clusters.items())[:5]:
            print(f"  {cid}: {splits}")


if __name__ == "__main__":
    main()
