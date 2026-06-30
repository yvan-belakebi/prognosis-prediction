"""
reorganize_trident_feats.py — Fold flat TRIDENT feature output into the biopsy-nested layout.

TRIDENT writes one .h5 per slide (keys 'features' (N,D) and 'coords' (N,2)) in a flat folder:
    {job_dir}/{mag}x_{ps}px_{ov}px_overlap/features_{enc_name}/{slide_id}.h5

ProcessedMILDataset (the CLAM-path layout) expects biopsy-nested files:
    {output_dir}/{biopsy_nr}/{slide_id}.h5

Each slide's biopsy_nr comes from the labels CSV (labels_unfiltered.csv from
define_labels.py), which carries file_name (slide stem) and biopsy_number (the
directory name) as separate columns. That CSV is the single source of truth for
nesting, so features fold into the same layout define_labels.py used for labels.

The h5 schema is identical, so this only relocates files (move, or copy with --copy).

Usage:
    python python_scripts/prepare_for_MIL/reorganize_trident_feats.py \\
        --features_dir WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --labels_csv   label_csvs/labels_unfiltered.csv \\
        --output_dir   WSI/IgA/UNI2-h_feats
"""

import argparse
import os
import shutil
import sys

import pandas as pd


def build_biopsy_lookup(labels_csv: str) -> dict:
    """Map slide_id (file_name) -> biopsy_nr (biopsy_number) from the labels CSV.

    biopsy_number is already the directory name produced by define_labels.py's
    biopsy_to_dirname; a missing/empty value means the slide nests at the root.
    """
    df = pd.read_csv(labels_csv, dtype=str)
    lookup = dict(zip(df["file_name"], df["biopsy_number"].fillna("")))
    return {k: ("" if v in ("", "nan", "None") else v) for k, v in lookup.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Relocate flat TRIDENT features into the biopsy-nested layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features_dir",
        required=True,
        help="Flat TRIDENT features_{enc} dir of {slide_id}.h5 files.",
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="labels_unfiltered.csv (from define_labels.py) mapping file_name -> biopsy_number.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination root for {biopsy_nr}/{slide_id}.h5.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (keeps the original TRIDENT output in place).",
    )
    parser.add_argument(
        "--no_auto_skip",
        action="store_true",
        help="Overwrite destination files that already exist (default: skip).",
    )
    args = parser.parse_args()

    lookup = build_biopsy_lookup(args.labels_csv)
    if not lookup:
        print(f"[ERROR] No slides found in {args.labels_csv}.")
        sys.exit(1)

    feat_files = [
        e.name
        for e in os.scandir(args.features_dir)
        if e.is_file() and e.name.endswith(".h5")
    ]
    if not feat_files:
        print(f"[ERROR] No .h5 files in {args.features_dir}.")
        sys.exit(1)
    print(f"Found {len(feat_files)} feature file(s); {len(lookup)} slides in labels CSV.")

    counts = {"done": 0, "skipped": 0, "unmatched": 0}
    op = shutil.copy2 if args.copy else shutil.move

    for name in sorted(feat_files):
        slide_id = os.path.splitext(name)[0]
        if slide_id not in lookup:
            print(f"  [WARN] {slide_id}: not in {args.labels_csv} — left in place.")
            counts["unmatched"] += 1
            continue

        biopsy_nr = lookup[slide_id]
        dst_dir = os.path.join(args.output_dir, biopsy_nr) if biopsy_nr else args.output_dir
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name)

        if not args.no_auto_skip and os.path.isfile(dst):
            counts["skipped"] += 1
            continue

        op(os.path.join(args.features_dir, name), dst)
        counts["done"] += 1

    print(
        f"\nDone.  {'Copied' if args.copy else 'Moved'}: {counts['done']}  "
        f"Skipped: {counts['skipped']}  Unmatched: {counts['unmatched']}"
    )
    print(f"Output → {args.output_dir}")


if __name__ == "__main__":
    main()
