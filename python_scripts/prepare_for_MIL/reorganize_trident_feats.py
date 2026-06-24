"""
reorganize_trident_feats.py — Fold flat TRIDENT feature output into the biopsy-nested layout.

TRIDENT writes one .h5 per slide (keys 'features' (N,D) and 'coords' (N,2)) in a flat folder:
    {job_dir}/{mag}x_{ps}px_{ov}px_overlap/features_{enc_name}/{slide_id}.h5

ProcessedMILDataset (the CLAM-path layout) expects biopsy-nested files:
    {output_dir}/{biopsy_nr}/{slide_id}.h5

Each slide's biopsy_nr is recovered from the nested raw-WSI directory — the same discovery
logic run_clam_tiling.py uses:
    {wsi_dir}/{biopsy_nr}/{slide_id}.{ext}   (nested)
    {wsi_dir}/{slide_id}.{ext}              (flat → biopsy_nr = "")

The h5 schema is identical, so this only relocates files (move, or copy with --copy).

Usage:
    python python_scripts/prepare_for_MIL/reorganize_trident_feats.py \\
        --features_dir WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --wsi_dir      data/raw_wsi/IgA \\
        --output_dir   WSI/IgA/UNI2-h_feats
"""

import argparse
import os
import shutil
import sys


def build_biopsy_lookup(wsi_dir: str, slide_exts: tuple) -> dict:
    """Map slide_id -> biopsy_nr by scanning the nested raw-WSI directory.

    Mirrors discover_slides() in run_clam_tiling.py: nested dirs take priority; if any
    subdirectory holds slide files, top-level files in wsi_dir are ignored.
    """

    def _matches(name: str) -> bool:
        return any(name.lower().endswith(ext) for ext in slide_exts)

    lookup = {}
    has_nested = False
    for entry in sorted(os.scandir(wsi_dir), key=lambda e: e.name):
        if entry.is_dir():
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_file() and _matches(sub.name):
                    slide_id = os.path.splitext(sub.name)[0]
                    lookup[slide_id] = entry.name
                    has_nested = True

    if not has_nested:
        for entry in sorted(os.scandir(wsi_dir), key=lambda e: e.name):
            if entry.is_file() and _matches(entry.name):
                slide_id = os.path.splitext(entry.name)[0]
                lookup[slide_id] = ""

    return lookup


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
        "--wsi_dir",
        required=True,
        help="Nested raw-WSI dir ({wsi_dir}/{biopsy_nr}/{slide_id}.ext) used to recover biopsy_nr.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination root for {biopsy_nr}/{slide_id}.h5.",
    )
    parser.add_argument(
        "--slide_ext",
        nargs="+",
        default=[".svs", ".ndpi"],
        help="WSI extension(s) used to scan --wsi_dir.",
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

    slide_exts = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.slide_ext
    )

    lookup = build_biopsy_lookup(args.wsi_dir, slide_exts)
    if not lookup:
        print(f"[ERROR] No slides found in {args.wsi_dir} (exts: {slide_exts}).")
        sys.exit(1)

    feat_files = [
        e.name
        for e in os.scandir(args.features_dir)
        if e.is_file() and e.name.endswith(".h5")
    ]
    if not feat_files:
        print(f"[ERROR] No .h5 files in {args.features_dir}.")
        sys.exit(1)
    print(f"Found {len(feat_files)} feature file(s); {len(lookup)} slides in WSI dir.")

    counts = {"done": 0, "skipped": 0, "unmatched": 0}
    op = shutil.copy2 if args.copy else shutil.move

    for name in sorted(feat_files):
        slide_id = os.path.splitext(name)[0]
        if slide_id not in lookup:
            print(f"  [WARN] {slide_id}: no matching WSI under {args.wsi_dir} — left in place.")
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
