"""
rename_biopsy_dirs.py — Rename biopsy-level subdirectories to normalised form.

Applies transform_label() then biopsy_to_dirname() to each first-level
subdirectory inside WSI/{IgA,IgA_registry,non_IgA}/{labels,coords,UNI2-h_feats}.

Example: "B2312" → "12-23"

Dry-run by default; pass --apply to perform the actual renames.
Run from the project root.
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from define_labels import biopsy_to_dirname, transform_label  # noqa: E402

BASES = ["WSI/IgA", "WSI/IgA_registry", "WSI/non_IgA"]
SUBDIRS = ["labels", "coords", "UNI2-h_feats"]


def process(directory: str, apply: bool) -> int:
    if not os.path.isdir(directory):
        return 0
    count = 0
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        new_name = biopsy_to_dirname(transform_label(entry.name))
        if new_name == entry.name:
            continue
        new_path = os.path.join(directory, new_name)
        merge = os.path.isdir(new_path)
        tag = ("MERGE " if merge else "RENAME") if apply else "DRY   "
        print(f"  [{tag}] {entry.name!r:30s} → {new_name!r}")
        if apply:
            if merge:
                # Destination already exists — move every item across, then
                # remove the now-empty source directory.
                for item in os.scandir(entry.path):
                    shutil.move(item.path, os.path.join(new_path, item.name))
                os.rmdir(entry.path)
            else:
                os.rename(entry.path, new_path)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Rename biopsy dirs to normalised form.")
    parser.add_argument("--apply", action="store_true", help="Perform renames (default: dry run).")
    args = parser.parse_args()

    total = 0
    for base in BASES:
        for sub in SUBDIRS:
            d = os.path.join(base, sub)
            n = process(d, args.apply)
            if n:
                print(f"  ↳ {n} in {d}\n")
            total += n

    print(f"\nTotal: {total} director{'y' if total == 1 else 'ies'} {'renamed' if args.apply else 'to rename'}.")
    if not args.apply and total:
        print("Re-run with --apply to rename.")


if __name__ == "__main__":
    main()
