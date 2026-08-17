"""
apply_biopsy_rename.py — Move slides into their unambiguous biopsy directory, using the
mapping from build_biopsy_name_mapping.py.

Undoes the directory merge caused by define_labels.transform_label dropping the lab-code
prefix from a biopsy number (see build_biopsy_name_mapping.py for the full story).  The
target directory is the raw biop_number verbatim, e.g.::

    WSI/non_IgA/labels/10959-13/2013_220010_ANON.npy   ->  .../B1310959/2013_220010_ANON.npy
    WSI/non_IgA/labels/10959-13/2013_110672_ANON.npy   ->  .../BG1310959/2013_110672_ANON.npy

Directories are never renamed — every file is placed individually by its own slide name.
A directory holding two patients therefore drains into two targets with no special case,
and the "split one folder in two" problem never arises.

Properties that make this safe to re-run:

  * the on-disk parent directory decides whether a file needs moving, not the CSV, so a
    partially completed run is fixed by simply running again;
  * slides absent from the mapping are left strictly alone — this is what protects the
    IgA cohort, which needs no rename;
  * an existing destination file is reported as a conflict and never overwritten;
  * nothing moves without --apply.

Run from the project root (dry run first):
    python python_scripts/prepare_for_MIL/apply_biopsy_rename.py \\
        --roots WSI/non_IgA/labels WSI/IgA_registry/labels

    # then, once the plan looks right:
    python python_scripts/prepare_for_MIL/apply_biopsy_rename.py \\
        --roots WSI/non_IgA/labels WSI/IgA_registry/labels --apply

Pass every root that is keyed by biopsy directory — label trees *and* feature trees — in
the same invocation, so labels and features cannot end up on different layouts.

After renaming, regenerate the label CSVs and validation lists so their biopsy_number
columns match the new directories:
    python python_scripts/prepare_for_MIL/define_labels.py
    python python_scripts/prepare_for_MIL/define_regression_labels.py
"""

import argparse
import os
import shutil
import sys

import pandas as pd


def load_targets(mapping_csv):
    """Return {slide_name: target_dir} from the mapping CSV."""
    df = pd.read_csv(mapping_csv, dtype=str)
    for col in ("slide_name", "target_dir"):
        if col not in df.columns:
            raise ValueError(f"{mapping_csv} must contain a '{col}' column.")
    df = df[df.slide_name.notna() & df.target_dir.notna()]
    conflicting = df.groupby("slide_name")["target_dir"].nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting):
        raise ValueError(
            f"{len(conflicting)} slide(s) map to more than one target_dir, e.g. "
            f"{conflicting.index[:5].tolist()} - the mapping is not a function."
        )
    return dict(zip(df.slide_name, df.target_dir))


def plan_root(root, targets):
    """Return (moves, stats) for one root.

    ``moves`` is a list of (src, dst).  The whole plan is built before anything is
    moved, so the directory walk is never disturbed by its own side effects.
    """
    moves = []
    stats = {"total": 0, "unmapped": 0, "in_place": 0, "conflict": 0}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            stats["total"] += 1
            stem = os.path.splitext(name)[0]
            target = targets.get(stem)
            if target is None:
                stats["unmapped"] += 1
                continue
            # The on-disk parent is the source of truth, so re-runs are no-ops.
            current = os.path.relpath(dirpath, root).replace(os.sep, "/")
            if current == target:
                stats["in_place"] += 1
                continue
            dst = os.path.join(root, target, name)
            if os.path.exists(dst):
                stats["conflict"] += 1
                print(f"  [CONFLICT] {dst} already exists - leaving {name} in {current}")
                continue
            moves.append((os.path.join(dirpath, name), dst))
    return moves, stats


def prune_empty_dirs(root):
    """Remove directories left empty under root (bottom-up). Returns the count."""
    removed = 0
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Move slides into their unambiguous biopsy directory (dry run by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mapping_csv",
        default="followup_data/derived/renamed/biopsy_name_mapping.csv",
        help="Output of build_biopsy_name_mapping.py (slide_name / target_dir columns).",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Biopsy-keyed roots to reorganize (label and feature trees alike).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without it, the plan is only printed.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (leaves the original layout intact; needs 2x space).",
    )
    parser.add_argument(
        "--keep_empty_dirs",
        action="store_true",
        help="Do not remove the directories left empty by the move.",
    )
    args = parser.parse_args()

    targets = load_targets(args.mapping_csv)
    print(f"Loaded {len(targets)} slide -> biopsy-dir mappings from {args.mapping_csv}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}"
          f"{' (copy)' if args.copy else ''}\n")

    missing = [r for r in args.roots if not os.path.isdir(r)]
    if missing:
        parser.error(f"root(s) not found: {missing}")

    grand = {"moves": 0, "total": 0, "unmapped": 0, "in_place": 0, "conflict": 0}
    op = shutil.copy2 if args.copy else shutil.move

    for root in args.roots:
        print(f"--- {root}")
        moves, stats = plan_root(root, targets)
        for k in ("total", "unmapped", "in_place", "conflict"):
            grand[k] += stats[k]
        grand["moves"] += len(moves)

        for src, dst in moves[:5]:
            print(f"  [{'MOVE' if args.apply else 'PLAN'}] {src}  ->  {dst}")
        if len(moves) > 5:
            print(f"  ... and {len(moves) - 5} more")

        if args.apply:
            for src, dst in moves:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                op(src, dst)
            if not args.copy and not args.keep_empty_dirs:
                n = prune_empty_dirs(root)
                print(f"  removed {n} empty directories")

        print(
            f"  files {stats['total']} | to move {len(moves)} | already in place "
            f"{stats['in_place']} | unmapped {stats['unmapped']} | conflicts {stats['conflict']}\n"
        )

    print(
        f"TOTAL  files {grand['total']} | "
        f"{'moved' if args.apply else 'to move'} {grand['moves']} | "
        f"already in place {grand['in_place']} | "
        f"unmapped {grand['unmapped']} | conflicts {grand['conflict']}"
    )
    if not args.apply and grand["moves"]:
        print("\nDry run - nothing was changed. Re-run with --apply to execute.")
    if grand["conflict"]:
        print(f"\n[WARNING] {grand['conflict']} conflict(s) were skipped; resolve them and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
