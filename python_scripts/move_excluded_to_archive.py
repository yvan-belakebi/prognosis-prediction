"""
move_excluded_to_archive.py — Move slides listed in an exclusion CSV from
source subdirectories to a parallel archive root, preserving the directory
structure.

Usage:
    # Dry run first (recommended)
    python python_scripts/move_excluded_to_archive.py --dry_run

    # Execute
    python python_scripts/move_excluded_to_archive.py
"""

import argparse
import os
import shutil

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Move excluded slides to an archive directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--excluded_csv",
        default="followup_data/iga_excluded_slides.csv",
        help="CSV with a 'file_name' column (stems, no extension) to move.",
    )
    parser.add_argument(
        "--source_root", default="WSI/IgA",
        help="Root directory containing the subdirs to process.",
    )
    parser.add_argument(
        "--dest_root", default="WSI/IgA_old",
        help="Archive root; subdirectory structure is mirrored here.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["coords", "labels", "UNI2-h_feats", "labels_classification_val"],
        help="Subdirectories under --source_root to process.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be moved without touching any files.",
    )
    args = parser.parse_args()

    excluded = set(
        pd.read_csv(args.excluded_csv)["file_name"].astype(str).str.strip()
    )
    print(f"Excluded basenames loaded: {len(excluded)}")
    prefix = "[DRY RUN] " if args.dry_run else ""

    total_moved = total_missing = 0

    for subdir in args.subdirs:
        src_dir = os.path.join(args.source_root, subdir)
        dst_dir = os.path.join(args.dest_root, subdir)

        if not os.path.isdir(src_dir):
            print(f"  {src_dir} — not found, skipping.")
            continue

        moved = missing = 0
        for fname in os.listdir(src_dir):
            stem = os.path.splitext(fname)[0]
            if stem not in excluded:
                continue
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            print(f"  {prefix}{src}  →  {dst}")
            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, dst)
            moved += 1

        # Report basenames that were expected but not found in this subdir
        present_stems = {os.path.splitext(f)[0] for f in os.listdir(src_dir)}
        missing = len(excluded - present_stems - {s for s in excluded})
        print(f"  {src_dir}: {prefix}moved {moved} file(s).")
        total_moved += moved

    print(f"\n{'Would move' if args.dry_run else 'Moved'} {total_moved} file(s) in total.")
    if args.dry_run:
        print("Re-run without --dry_run to apply.")


if __name__ == "__main__":
    main()
