"""
Apply curated decisions to Project Design 20242025 dataset.

Reads the flagged/manually reviewed curation_master CSV and:
1. Filters images with decision = KEEP or RELABEL
2. Copies to interim directory with correct labels
3. Generates a change log

This script does NOT modify raw data.
"""
import csv
import shutil
import json
from pathlib import Path
from collections import Counter

CURATION_CSV = Path(r"D:\.Candra\Project\TA\03_metadata\curation\project_design_v1\curation_master_flagged.csv")
RAW_ROOT = Path(r"D:\.Candra\Project\TA\02_data\raw\project_design_20242025")
OUTPUT_DIR = Path(r"D:\.Candra\Project\TA\02_data\interim\project_design_curated")
LOG_DIR = OUTPUT_DIR / "_curation_log"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CURATION_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Determine which decision column to use
    # Priority: manual 'decision' column > 'auto_decision'
    decision_col = "decision"
    auto_col = "auto_decision"

    stats = Counter()
    kept_records = []
    change_log = []

    for row in rows:
        image_id = row["image_id"]
        original_path = Path(row["original_path"])
        original_label = row["original_label"]

        # Use manual decision if available, otherwise auto
        manual_decision = (row.get(decision_col) or "").strip().upper()
        auto_decision = (row.get(auto_col) or "").strip().upper()

        effective_decision = manual_decision if manual_decision else auto_decision

        if not effective_decision:
            stats["no_decision"] += 1
            continue

        if effective_decision == "KEEP":
            final_label = row.get("final_label") or original_label
            stats["keep"] += 1
        elif effective_decision == "RELABEL":
            final_label = row.get("final_label", "").strip()
            if not final_label:
                stats["relabel_no_target"] += 1
                change_log.append(f"SKIP {image_id}: RELABEL but no final_label specified")
                continue
            stats["relabel"] += 1
        elif effective_decision in ("EXCLUDE", "REMOVE"):
            stats["exclude"] += 1
            continue
        elif effective_decision == "REVIEW":
            # Treat unresolved REVIEW as skip for now
            stats["review_pending"] += 1
            continue
        else:
            stats[f"unknown_{effective_decision}"] += 1
            continue

        # Copy image to curated directory (flat structure by label)
        dest_dir = OUTPUT_DIR / final_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        src = original_path
        if not src.exists():
            # Try relative path
            src = RAW_ROOT / row["relative_path"]

        if not src.exists():
            stats["missing_source"] += 1
            change_log.append(f"MISSING {image_id}: {src}")
            continue

        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)

        kept_records.append({
            "image_id": image_id,
            "final_label": final_label,
            "original_label": original_label,
            "decision": effective_decision,
            "filename": src.name,
        })

        if final_label != original_label:
            change_log.append(f"RELABEL {image_id}: {original_label} → {final_label}")

    # Save manifest of curated images
    manifest_path = LOG_DIR / "curated_manifest.csv"
    if kept_records:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=kept_records[0].keys())
            writer.writeheader()
            writer.writerows(kept_records)

    # Save change log
    log_path = LOG_DIR / "change_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(change_log) if change_log else "No changes logged.")

    # Save summary
    summary = {
        "total_rows": len(rows),
        "kept": stats["keep"],
        "relabeled": stats["relabel"],
        "excluded": stats["exclude"],
        "review_pending": stats["review_pending"],
        "no_decision": stats["no_decision"],
        "missing_source": stats.get("missing_source", 0),
        "curated_total": len(kept_records),
    }
    with open(LOG_DIR / "apply_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"{'='*60}")
    print("APPLY CURATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows in CSV: {len(rows)}")
    print(f"  KEEP:           {stats['keep']}")
    print(f"  RELABEL:        {stats['relabel']}")
    print(f"  EXCLUDE:        {stats['exclude']}")
    print(f"  REVIEW pending: {stats['review_pending']}")
    print(f"  No decision:    {stats['no_decision']}")
    print(f"  Missing source: {stats.get('missing_source', 0)}")
    print(f"\nCurated images written: {len(kept_records)}")
    print(f"Output: {OUTPUT_DIR}")

    # Count per-label distribution
    label_counts = Counter(r["final_label"] for r in kept_records)
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
