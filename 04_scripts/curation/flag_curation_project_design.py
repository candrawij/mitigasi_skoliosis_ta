"""
Semi-automatic curation flags for Project Design 20242025.

Reads curation_master.csv and applies heuristic-based flags to
pre-populate the decision/quality columns. All flags are suggestions
that must be reviewed manually — no images are deleted automatically.

Heuristics:
  1. Singletons (cluster_size=1) → flag as KEEP candidate
  2. Cross-split clusters → flag as REVIEW_LEAKAGE
  3. Mixed-label clusters → flag as REVIEW_LABEL
  4. Large clusters (>20 images) → flag as REVIEW_DEDUP
  5. Suggested representatives → flag as KEEP candidate within cluster
  6. Non-representatives in same-label same-split clusters → flag as EXCLUDE candidate
"""
import csv
from pathlib import Path
from collections import Counter

INPUT_CSV = Path(r"D:\.Candra\Project\TA\03_metadata\curation\project_design_v1\curation_master.csv")
OUTPUT_CSV = Path(r"D:\.Candra\Project\TA\03_metadata\curation\project_design_v1\curation_master_flagged.csv")
REPORT_PATH = Path(r"D:\.Candra\Project\TA\03_metadata\curation\project_design_v1\curation_flags_report.md")


def main():
    # Read existing curation master
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Add flag columns if not present
    new_cols = ["auto_flag", "auto_decision", "auto_notes"]
    for col in new_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    stats = Counter()
    total = len(rows)

    for row in rows:
        cluster_size = int(row.get("cluster_size", 1) or 1)
        mixed_label = row.get("cluster_mixed_label", "False") == "True"
        cross_split = row.get("cluster_cross_split", "False") == "True"
        is_rep = row.get("suggested_representative", "False") == "True"
        cluster_id = row.get("cluster_id", "")

        flags = []
        decision = ""
        notes = []

        # 1. Singleton — likely unique, keep
        if cluster_size <= 1 or not cluster_id:
            flags.append("SINGLETON")
            decision = "KEEP"
            notes.append("Unique image, no near-duplicates")
            stats["singleton_keep"] += 1

        else:
            # 2. Mixed label + cross split → highest priority review
            if mixed_label and cross_split:
                flags.append("P1_MIXED_CROSS")
                decision = "REVIEW"
                notes.append("Mixed labels AND cross-split: check label + reassign split")
                stats["p1_mixed_cross"] += 1

            # 3. Cross-split only
            elif cross_split and not mixed_label:
                flags.append("P2_CROSS_SPLIT")
                if is_rep:
                    decision = "KEEP"
                    notes.append("Representative of cross-split cluster: keep, fix split later")
                    stats["p2_cross_rep_keep"] += 1
                else:
                    decision = "REVIEW"
                    notes.append("Non-rep in cross-split cluster: may exclude to fix leakage")
                    stats["p2_cross_nonrep_review"] += 1

            # 4. Mixed label only
            elif mixed_label and not cross_split:
                flags.append("P3_MIXED_LABEL")
                decision = "REVIEW"
                notes.append("Mixed labels within cluster: verify correct label")
                stats["p3_mixed_label"] += 1

            # 5. Same label, same split — dedup candidates
            else:
                flags.append("P4_SAME")
                if is_rep:
                    decision = "KEEP"
                    notes.append("Representative of clean cluster")
                    stats["p4_same_rep_keep"] += 1
                else:
                    if cluster_size > 20:
                        decision = "EXCLUDE"
                        notes.append(f"Non-rep in large cluster (size={cluster_size}): suggest exclude")
                        stats["p4_large_exclude"] += 1
                    else:
                        decision = "EXCLUDE"
                        notes.append(f"Non-rep in cluster (size={cluster_size}): suggest exclude")
                        stats["p4_nonrep_exclude"] += 1

        row["auto_flag"] = "|".join(flags)
        row["auto_decision"] = decision
        row["auto_notes"] = "; ".join(notes)

    # Save flagged CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved flagged CSV to {OUTPUT_CSV}")

    # Generate report
    keep_count = sum(1 for r in rows if r["auto_decision"] == "KEEP")
    review_count = sum(1 for r in rows if r["auto_decision"] == "REVIEW")
    exclude_count = sum(1 for r in rows if r["auto_decision"] == "EXCLUDE")

    report_lines = [
        "# Laporan Curation Flags Semi-Otomatis — Project Design 20242025",
        "",
        f"- Total gambar: **{total}**",
        f"- Auto KEEP: **{keep_count}** ({keep_count/total*100:.1f}%)",
        f"- Auto REVIEW: **{review_count}** ({review_count/total*100:.1f}%)",
        f"- Auto EXCLUDE: **{exclude_count}** ({exclude_count/total*100:.1f}%)",
        "",
        "## Rincian Flags",
        "",
        "| Flag | Jumlah | Keterangan |",
        "|---|---:|---|",
    ]

    flag_descriptions = {
        "singleton_keep": ("Singleton (unique)", "Gambar tanpa near-duplicate → KEEP"),
        "p1_mixed_cross": ("P1: Mixed + Cross-split", "Label campur DAN tersebar antar split → REVIEW"),
        "p2_cross_rep_keep": ("P2: Cross-split, representative", "Wakil cluster cross-split → KEEP"),
        "p2_cross_nonrep_review": ("P2: Cross-split, non-rep", "Non-wakil di cluster cross-split → REVIEW"),
        "p3_mixed_label": ("P3: Mixed label", "Label campur dalam satu cluster → REVIEW"),
        "p4_same_rep_keep": ("P4: Same, representative", "Wakil cluster bersih → KEEP"),
        "p4_large_exclude": ("P4: Large cluster, non-rep", "Non-wakil di cluster besar (>20) → EXCLUDE"),
        "p4_nonrep_exclude": ("P4: Small cluster, non-rep", "Non-wakil di cluster kecil → EXCLUDE"),
    }

    for key, (name, desc) in flag_descriptions.items():
        count = stats.get(key, 0)
        report_lines.append(f"| {name} | {count} | {desc} |")

    report_lines.extend([
        "",
        "## Urutan Audit Manual",
        "",
        "1. **REVIEW** items terlebih dahulu (mixed/cross-split clusters)",
        "2. Verifikasi **KEEP** untuk representatives: pastikan label benar",
        "3. Konfirmasi **EXCLUDE** pada non-representatives: pastikan tidak ada variasi penting yang hilang",
        "",
        "## Catatan Penting",
        "",
        "- Semua keputusan auto bersifat **saran**, bukan final",
        "- Gambar **tidak dihapus** oleh script ini",
        "- Kolom `auto_flag`, `auto_decision`, `auto_notes` ditambahkan",
        "- Kolom `decision` asli tidak diubah — isi secara manual setelah review",
        f"- File output: `{OUTPUT_CSV.name}`",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved report to {REPORT_PATH}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"  KEEP:    {keep_count} ({keep_count/total*100:.1f}%)")
    print(f"  REVIEW:  {review_count} ({review_count/total*100:.1f}%)")
    print(f"  EXCLUDE: {exclude_count} ({exclude_count/total*100:.1f}%)")
    print()
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
