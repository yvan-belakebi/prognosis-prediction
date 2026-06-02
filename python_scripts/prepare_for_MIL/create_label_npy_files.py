"""
create_label_npy_files.py — Convert a label CSV into per-slide .npy files for MIL training.

Supports two label formats, auto-detected from the CSV columns:

  Survival        columns: file_name, time, event  (+ any extras)
                  saves:   np.array([time, event])  shape (2,), float64

  Classification  columns: file_name, ckd_label  (+ any extras)
                  saves:   np.array(int)           scalar int

Output routing
--------------
Use --output_dir for a single destination directory.
Use --source_dirs when the CSV has a 'source' column and files should be
routed to different directories depending on the source value.

Run from the project root:

  # Survival, all rows in one directory
  python python_scripts/prepare_for_MIL/create_label_npy_files.py \\
      --csv followup_data/labels_pas.csv \\
      --output_dir WSI/IgA/labels

  # Survival, PAS only, split by source into IgA and registry_IgA
  python python_scripts/prepare_for_MIL/create_label_npy_files.py \\
      --csv followup_data/labels_pas.csv \\
      --source_dirs IgA=WSI/IgA/labels registry=WSI/registry_IgA/labels \\
      --stain_filter PAS

  # Classification labels (single source)
  python python_scripts/prepare_for_MIL/create_label_npy_files.py \\
      --csv followup_data/labels_classification.csv \\
      --output_dir WSI/IgA/labels_classification
"""

import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def detect_mode(df: pd.DataFrame, mode_hint: str | None) -> str:
    if mode_hint is not None:
        return mode_hint
    cols = set(df.columns)
    if {"time", "event"} <= cols:
        return "survival"
    if "ckd_label" in cols:
        return "classification"
    raise ValueError(
        "Cannot auto-detect mode: CSV must have either ('time', 'event') "
        "columns (survival) or a 'ckd_label' column (classification). "
        "Pass --mode to override."
    )


# ---------------------------------------------------------------------------
# Per-row savers (write a single .npy file, return True on success)
# ---------------------------------------------------------------------------

def _save_one_survival(row, output_dir: str) -> bool:
    if pd.isna(row["time"]) or pd.isna(row["event"]):
        print(f"  Warning: skipping '{row['file_name']}' — NaN in time or event.")
        return False
    arr = np.array([float(row["time"]), float(row["event"])], dtype=np.float64)
    np.save(os.path.join(output_dir, f"{row['file_name']}.npy"), arr)
    return True


def _save_one_classification(row, output_dir: str) -> bool:
    if pd.isna(row["ckd_label"]):
        print(f"  Warning: skipping '{row['file_name']}' — NaN in ckd_label.")
        return False
    np.save(
        os.path.join(output_dir, f"{row['file_name']}.npy"),
        np.array(int(row["ckd_label"])),
    )
    return True


# ---------------------------------------------------------------------------
# Batch save
# ---------------------------------------------------------------------------

def save_all(df: pd.DataFrame, mode: str, dir_map: dict[str, str]) -> tuple[int, int]:
    """Save all rows, routing each row to its output directory via dir_map.

    dir_map maps source_value → output_dir, or uses the single key None for
    a flat (no-source) destination.
    Returns (saved, skipped).
    """
    saver = _save_one_survival if mode == "survival" else _save_one_classification
    saved = skipped = 0

    for _, row in df.iterrows():
        src = row.get("source", None) if "source" in df.columns else None
        out_dir = dir_map.get(src, dir_map.get(None))
        if out_dir is None:
            print(
                f"  Warning: skipping '{row['file_name']}' — "
                f"no output directory mapped for source '{src}'."
            )
            skipped += 1
            continue
        if saver(row, out_dir):
            saved += 1
        else:
            skipped += 1

    return saved, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a label CSV into per-slide .npy files for MIL training."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Input CSV (e.g. labels_pas.csv or labels_classification.csv).",
    )

    dest_group = parser.add_mutually_exclusive_group(required=True)
    dest_group.add_argument(
        "--output_dir",
        help="Single output directory for all rows.",
    )
    dest_group.add_argument(
        "--source_dirs",
        nargs="+",
        metavar="SOURCE=PATH",
        help=(
            "Per-source output directories as KEY=PATH pairs "
            "(requires a 'source' column in the CSV). "
            "Example: IgA=WSI/IgA/labels registry=WSI/registry_IgA/labels"
        ),
    )

    parser.add_argument(
        "--mode", choices=["survival", "classification"], default=None,
        help="Label type. Auto-detected from CSV columns when omitted.",
    )
    parser.add_argument(
        "--stain_filter", default=None,
        help=(
            "Keep only rows whose 'stain' or 'Stain' column matches this value "
            "(e.g. PAS). Applied before writing."
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"file_name": str})
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Columns: {list(df.columns)}")

    mode = detect_mode(df, args.mode)
    print(f"Mode: {mode}")

    # Optional stain filter
    if args.stain_filter is not None:
        stain_col = "stain" if "stain" in df.columns else "Stain"
        if stain_col not in df.columns:
            parser.error(
                f"--stain_filter requires a 'stain' or 'Stain' column; "
                f"found: {list(df.columns)}"
            )
        before = len(df)
        df = df[df[stain_col] == args.stain_filter].copy()
        print(f"Stain filter '{args.stain_filter}': {before} → {len(df)} rows")

    # Build the source → output_dir mapping
    if args.source_dirs is not None:
        if "source" not in df.columns:
            parser.error(
                "--source_dirs requires a 'source' column in the CSV; "
                f"found: {list(df.columns)}"
            )
        dir_map: dict = {}
        for token in args.source_dirs:
            if "=" not in token:
                parser.error(
                    f"--source_dirs entries must be KEY=PATH pairs, got: '{token}'"
                )
            key, path = token.split("=", 1)
            dir_map[key] = path

        unknown = set(df["source"].dropna().unique()) - set(dir_map)
        if unknown:
            parser.error(
                f"CSV contains source values with no mapped directory: {sorted(unknown)}. "
                f"Add them to --source_dirs or pre-filter the CSV."
            )
    else:
        # Single flat destination, keyed by None
        dir_map = {None: args.output_dir}

    # Create all output directories up front
    for path in dir_map.values():
        os.makedirs(path, exist_ok=True)

    saved, skipped = save_all(df, mode, dir_map)

    # Summary
    print()
    if args.source_dirs is not None:
        for src, path in dir_map.items():
            n = (df["source"] == src).sum()
            print(f"  {src:>12s} ({n} rows) → {path}/")
    else:
        print(f"  → {args.output_dir}/")
    print(f"\nWrote {saved} .npy files.")
    if skipped:
        print(f"Skipped {skipped} rows (NaN values or unmapped source).")


if __name__ == "__main__":
    main()
