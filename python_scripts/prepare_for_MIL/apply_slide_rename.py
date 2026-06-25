"""
apply_slide_rename.py — Apply an existing old_name -> new_name mapping to slide artifacts.

Renames files across one or more directories using the mapping produced by
reorganize_wsi_dirs.py --rename:
    followup_data/derived/renamed/slide_name_mapping.csv
    (columns: biopsy_number, old_name, new_name)

Unlike reorganize_wsi_dirs.py --rename (which rebuilds the plan from the source metadata
CSVs and assumes filename stem == slide name), this script just APPLIES an existing mapping
and understands TRIDENT's '_patches' coord-file suffix:

    {old}_patches.h5   ->  {new}_patches.h5      (coords)
    {old}.svs/.ndpi    ->  {new}.svs/.ndpi       (raw WSI)
    {old}.jpg          ->  {new}.jpg             (contour / thumbnail)
    {old}.geojson      ->  {new}.geojson         (tissue contour)

Files whose stem is not in the mapping (already-safe names) are left untouched. Each
directory is searched recursively, so biopsy-nested raw-WSI layouts are handled.

Dry run by default — prints planned renames without touching disk. Pass --apply to execute.

Usage:
    python python_scripts/prepare_for_MIL/apply_slide_rename.py \\
        --dirs data/raw_wsi/IgA \\
               WSI/IgA/trident/20x_224px_0px_overlap/patches \\
               WSI/IgA/trident/contours \\
               WSI/IgA/trident/contours_geojson \\
               WSI/IgA/trident/thumbnails
    # inspect the dry run, then re-run with --apply
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trident_io import TRIDENT_COORD_SUFFIX  # noqa: E402  ("_patches")


def load_mapping(path):
    """Return {old_name: new_name} from the mapping CSV."""
    df = pd.read_csv(path)
    for col in ("old_name", "new_name"):
        if col not in df.columns:
            raise SystemExit(
                f"Mapping CSV must contain '{col}' (got {list(df.columns)})."
            )
    mapping = {}
    for _, r in df.iterrows():
        old = str(r["old_name"]).strip()
        new = str(r["new_name"]).strip()
        if old and new and old.lower() != "nan" and new.lower() != "nan":
            mapping[old] = new
    return mapping


def split_name(filename):
    """Split a filename into (base_stem, middle, ext).

    middle is TRIDENT's coord suffix ('_patches') when present, else ''. base_stem is the
    slide name to look up in the mapping. e.g.:
        'foo_patches.h5' -> ('foo', '_patches', '.h5')
        'foo.svs'        -> ('foo', '',         '.svs')
    """
    stem, ext = os.path.splitext(filename)
    if stem.endswith(TRIDENT_COORD_SUFFIX):
        return stem[: -len(TRIDENT_COORD_SUFFIX)], TRIDENT_COORD_SUFFIX, ext
    return stem, "", ext


def plan_dir(directory, mapping):
    """Yield (src_path, dst_path) for files whose base stem is in the mapping (recursive)."""
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            base, middle, ext = split_name(fn)
            new = mapping.get(base)
            if new is None:
                continue
            yield os.path.join(root, fn), os.path.join(root, f"{new}{middle}{ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply an old_name->new_name mapping to slide artifact directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mapping_csv",
        default="followup_data/derived/renamed/slide_name_mapping.csv",
        help="CSV with old_name / new_name columns (reorganize_wsi_dirs.py --rename output).",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Directories to rename within (searched recursively).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames. Omitting this flag is a safe dry run.",
    )
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_csv)
    print(f"Loaded {len(mapping)} old->new name mappings from {args.mapping_csv}.")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}\n")

    counts = {"renamed": 0, "skipped_exists": 0, "noop": 0}
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"[SKIP] Not a directory: {d}")
            continue
        print(f"--- {d} ---")
        n = 0
        for src, dst in plan_dir(d, mapping):
            if os.path.abspath(src) == os.path.abspath(dst):
                counts["noop"] += 1
                continue
            if os.path.exists(dst):
                print(f"  [WARN] dest exists, skipping: {os.path.basename(dst)}")
                counts["skipped_exists"] += 1
                continue
            tag = "REN " if args.apply else "DRY "
            print(f"  [{tag}] {os.path.basename(src)}  ->  {os.path.basename(dst)}")
            if args.apply:
                os.rename(src, dst)
            counts["renamed"] += 1
            n += 1
        print(f"  {n} file(s) {'renamed' if args.apply else 'to rename'}\n")

    print("=" * 60)
    print(f"{'Renamed' if args.apply else 'To rename'} : {counts['renamed']}")
    print(f"Skipped (dest exists)    : {counts['skipped_exists']}")
    if not args.apply:
        print("\nDry run — nothing was changed. Re-run with --apply to perform the renames.")


if __name__ == "__main__":
    main()
