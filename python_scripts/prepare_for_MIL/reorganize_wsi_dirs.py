"""
reorganize_wsi_dirs.py — Restructure flat WSI slide directories to biopsy-nested layout.

Before: WSI/IgA/UNI2-h_feats/slide_name.h5
After:  WSI/IgA/UNI2-h_feats/<biopsy_dir>/slide_name.h5

Biopsy directories are derived from:
  - IgA cohort     : followup_data/IgA_slide_data.csv   (Biopsy Number + File Location)
  - Registry cohort: followup_data/registry_anonymized.csv  (biop_number + ANON_name)

By default runs in dry-run mode — prints planned moves without touching files.
Pass --apply to perform the actual moves.

Run from the project root:
    # Dry run (default) — inspect what would happen
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py \\
        --iga_dirs WSI/IgA/UNI2-h_feats WSI/IgA/labels WSI/IgA/coords \\
        --registry_dirs WSI/IgA_registry/UNI2-h_feats WSI/IgA_registry/labels \\
                        WSI/non_IgA/UNI2-h_feats WSI/non_IgA/labels

    # Apply changes
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py \\
        --iga_dirs WSI/IgA/UNI2-h_feats WSI/IgA/labels WSI/IgA/coords \\
        --registry_dirs WSI/IgA_registry/UNI2-h_feats WSI/IgA_registry/labels \\
                        WSI/non_IgA/UNI2-h_feats WSI/non_IgA/labels \\
        --apply
"""

import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from define_labels import (
    biopsy_to_dirname,
    extract_file_name,
    transform_label,
)  # noqa: E402

SLIDE_EXTS = {".npy", ".h5"}


# ---------------------------------------------------------------------------
# Build slide-stem → biopsy-dir mappings from CSV metadata
# ---------------------------------------------------------------------------


def build_iga_mapping(iga_slides_csv: str) -> dict:
    """Return {slide_stem: biopsy_dir} for the IgA cohort."""
    df = pd.read_csv(iga_slides_csv)
    mapping = {}
    for _, row in df.iterrows():
        stem = extract_file_name(row["File Location"])
        biopsy_dir = biopsy_to_dirname(str(row["Biopsy Number"]))  # transform_label()
        if stem and not pd.isna(stem):
            mapping[stem] = biopsy_dir
    return mapping


def build_registry_mapping(registry_csv: str) -> dict:
    """Return {slide_stem: biopsy_dir} for the full registry (IgA + non-IgA)."""
    df = pd.read_csv(registry_csv, usecols=["ANON_name", "biop_number"])
    mapping = {}
    for _, row in df.iterrows():
        stem = str(row["ANON_name"]).strip()
        biopsy_dir = biopsy_to_dirname(str(row["biop_number"]))  # transform_label()
        if stem:
            mapping[stem] = biopsy_dir
    return mapping


# ---------------------------------------------------------------------------
# Core: reorganize one directory
# ---------------------------------------------------------------------------


def reorganize_dir(directory: str, mapping: dict, apply: bool) -> tuple:
    """Move flat slide files into biopsy subdirectories.

    Only processes files directly under `directory` (non-recursive). Files
    already inside a subdirectory are treated as already nested and skipped.

    Returns (moved, already_nested, unmapped) counts.
    """
    if not os.path.isdir(directory):
        print(f"  [SKIP] Directory not found: {directory}")
        return 0, 0, 0

    moved = already_nested = unmapped = 0

    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if entry.is_dir():
            already_nested += 1
            continue
        if not entry.is_file():
            continue

        stem, ext = os.path.splitext(entry.name)
        if ext not in SLIDE_EXTS:
            continue

        biopsy_dir = mapping.get(stem)
        if biopsy_dir is None:
            print(
                f"  [WARN ] No biopsy mapping for '{stem}' in {directory}"
                f" — will move to 'unknown/'"
            )
            biopsy_dir = "unknown"
            unmapped += 1

        dest_dir = os.path.join(directory, biopsy_dir)
        dest_path = os.path.join(dest_dir, entry.name)

        if apply:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(entry.path, dest_path)
            print(f"  [MOVE ] {entry.name}  →  {biopsy_dir}/{entry.name}")
        else:
            print(f"  [DRY  ] {entry.name}  →  {biopsy_dir}/{entry.name}")

        moved += 1

    return moved, already_nested, unmapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize flat WSI slide directories into biopsy-nested layout.\n"
            "Default: dry run (no files moved). Pass --apply to execute."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iga_dirs",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "Directories that hold IgA slides at the top level "
            "(feats / labels / coords). Biopsy mapping comes from --iga_slides_csv."
        ),
    )
    parser.add_argument(
        "--registry_dirs",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "Directories that hold registry slides (IgA_registry + non_IgA) at the "
            "top level. Biopsy mapping comes from --registry_csv."
        ),
    )
    parser.add_argument(
        "--iga_slides_csv",
        default="followup_data/IgA_slide_data.csv",
        help="IgA slide metadata CSV (columns: Biopsy Number, File Location).",
    )
    parser.add_argument(
        "--registry_csv",
        default="followup_data/registry_anonymized.csv",
        help="Registry metadata CSV (columns: ANON_name, biop_number).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the actual file moves. Omitting this flag is a safe dry run.",
    )
    args = parser.parse_args()

    if not args.iga_dirs and not args.registry_dirs:
        parser.error("Provide at least one of --iga_dirs or --registry_dirs.")

    mode_label = "APPLY" if args.apply else "DRY RUN"
    print(f"Mode: {mode_label}\n")

    # --- Build mappings -------------------------------------------------------
    iga_mapping: dict = {}
    if args.iga_dirs:
        print(f"Loading IgA slide mapping from: {args.iga_slides_csv}")
        iga_mapping = build_iga_mapping(args.iga_slides_csv)
        print(f"  {len(iga_mapping)} IgA slides mapped to biopsy directories.\n")

    registry_mapping: dict = {}
    if args.registry_dirs:
        print(f"Loading registry mapping from: {args.registry_csv}")
        registry_mapping = build_registry_mapping(args.registry_csv)
        print(
            f"  {len(registry_mapping)} registry slides mapped to biopsy directories.\n"
        )

    # --- Process directories --------------------------------------------------
    total_moved = total_nested = total_unmapped = 0

    for d in args.iga_dirs:
        print(f"--- IgA: {d} ---")
        m, n, u = reorganize_dir(d, iga_mapping, args.apply)
        total_moved += m
        total_nested += n
        total_unmapped += u
        print(
            f"  Summary: {m} files {'moved' if args.apply else 'to move'}"
            f" | {n} already nested (skipped) | {u} unmapped (→ unknown/)\n"
        )

    for d in args.registry_dirs:
        print(f"--- Registry: {d} ---")
        m, n, u = reorganize_dir(d, registry_mapping, args.apply)
        total_moved += m
        total_nested += n
        total_unmapped += u
        print(
            f"  Summary: {m} files {'moved' if args.apply else 'to move'}"
            f" | {n} already nested (skipped) | {u} unmapped (→ unknown/)\n"
        )

    print("=" * 60)
    print(f"Total files {'moved' if args.apply else 'to move'} : {total_moved}")
    print(f"Total already-nested (skipped)     : {total_nested}")
    print(f"Total unmapped (→ unknown/)         : {total_unmapped}")
    if not args.apply:
        print(
            "\nThis was a dry run — no files were changed.\n"
            "Run with --apply to perform the actual moves."
        )


if __name__ == "__main__":
    main()
