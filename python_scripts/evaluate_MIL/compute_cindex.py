"""
compute_cindex.py — Pool risk_scores.csv files from evaluate_survival.py and report C-index.

Merges one or more risk_scores.csv files into a single table and computes the
concordance index using sksurv.metrics.concordance_index_censored.

Usage:
    # From directories (looks for risk_scores.csv inside each):
    python python_scripts/MIL/compute_cindex.py \\
        --input_dirs results/eval_30pct/ results/eval_20pct/

    # From explicit CSV paths:
    python python_scripts/MIL/compute_cindex.py \\
        --inputs results/eval_30pct/risk_scores.csv results/eval_20pct/risk_scores.csv

    # Also report per-file C-indices before the pooled result:
    python python_scripts/MIL/compute_cindex.py \\
        --input_dirs results/eval_30pct/ results/eval_20pct/ --per_file

    # Override time/event with an external label CSV:
    python python_scripts/MIL/compute_cindex.py \\
        --inputs results/eval_30pct/risk_scores.csv \\
        --label_csv followup_data/labels_combined.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_risk_csv(path):
    """Load a risk_scores.csv and normalise the identifier column to 'id'.

    Accepts both biopsy-nested output (column 'biopsy') and flat output
    (column 'bag_name'), as produced by evaluate_survival.py.
    """
    df = pd.read_csv(path, dtype={"biopsy": str, "bag_name": str})
    if "biopsy" in df.columns:
        df = df.rename(columns={"biopsy": "id"})
    elif "bag_name" in df.columns:
        df = df.rename(columns={"bag_name": "id"})
    else:
        raise ValueError(
            f"{path}: expected a 'biopsy' or 'bag_name' column, "
            f"found: {list(df.columns)}"
        )
    for col in ("risk", "time", "event"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}'.")
    return df[["id", "risk", "time", "event"]].copy()


def _cindex(df, label=None):
    """Compute and print C-index for a DataFrame with risk/time/event columns.

    Returns the concordance float, or None when there are no events.
    """
    event = df["event"].astype(bool).to_numpy()
    time = df["time"].astype(float).to_numpy()
    risk = df["risk"].astype(float).to_numpy()

    prefix = f"[{label}] " if label else ""

    if event.sum() == 0:
        print(f"  {prefix}No events — C-index undefined.")
        return None

    c, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        event, time, risk
    )
    print(
        f"  {prefix}"
        f"N={len(df):>4d}  events={int(event.sum()):>3d}  "
        f"C-index={c:.4f}  "
        f"(concordant={concordant}, discordant={discordant}, "
        f"tied_risk={tied_risk}, tied_time={tied_time})"
    )
    return c


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute C-index from one or more evaluate_survival.py risk CSV files."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--inputs",
        nargs="+",
        metavar="CSV",
        help="Direct paths to risk_scores.csv files.",
    )
    src.add_argument(
        "--input_dirs",
        nargs="+",
        metavar="DIR",
        help="Directories that each contain a risk_scores.csv (or --csv_name).",
    )

    parser.add_argument(
        "--csv_name",
        default="risk_scores.csv",
        help="Filename to look for when using --input_dirs (default: risk_scores.csv).",
    )
    parser.add_argument(
        "--label_csv",
        default=None,
        help=(
            "Optional external label CSV with columns 'biopsy' (or 'file_name'), "
            "'time', and 'event'. When provided, the time/event values in the "
            "risk CSV files are replaced by these labels before computing C-index."
        ),
    )
    parser.add_argument(
        "--per_file",
        action="store_true",
        help="Report C-index for each input file individually, then the pooled result.",
    )
    parser.add_argument(
        "--warn_duplicates",
        action="store_true",
        help=(
            "Warn when the same biopsy/slide ID appears in more than one input file. "
            "Duplicates are kept by default (both scores contribute to the pooled C-index)."
        ),
    )
    args = parser.parse_args()

    # --- Resolve CSV paths ---------------------------------------------------
    if args.inputs:
        csv_paths = list(args.inputs)
    else:
        csv_paths = []
        for d in args.input_dirs:
            p = os.path.join(d, args.csv_name)
            if not os.path.isfile(p):
                print(f"Warning: {p} not found — skipping.")
                continue
            csv_paths.append(p)

    if not csv_paths:
        sys.exit("No valid input CSV files found.")

    # --- Load and optionally report per-file C-index -------------------------
    parts = []
    for path in csv_paths:
        df = load_risk_csv(path)
        df["_source"] = path
        parts.append(df)

    if args.per_file:
        print(f"\n--- Per-file C-index ---")
        for path, df in zip(csv_paths, parts):
            _cindex(df, label=os.path.relpath(path))

    # --- Pool ----------------------------------------------------------------
    pooled = pd.concat(parts, ignore_index=True)

    if args.warn_duplicates:
        dupes = pooled["id"][pooled["id"].duplicated(keep=False)].unique()
        if len(dupes):
            print(f"\nWarning: {len(dupes)} ID(s) appear in more than one file:")
            for d in sorted(dupes)[:10]:
                print(f"  {d}")
            if len(dupes) > 10:
                print(f"  … and {len(dupes) - 10} more.")

    # --- Optional label override ---------------------------------------------
    if args.label_csv is not None:
        labels = pd.read_csv(args.label_csv, dtype=str)
        id_col = next(
            (c for c in ("biopsy", "file_name") if c in labels.columns), None
        )
        if id_col is None:
            sys.exit(
                f"--label_csv must have a 'biopsy' or 'file_name' column; "
                f"found: {list(labels.columns)}"
            )
        labels = labels.rename(columns={id_col: "id"})
        labels["time"] = pd.to_numeric(labels["time"], errors="coerce")
        labels["event"] = pd.to_numeric(labels["event"], errors="coerce")
        labels = labels[["id", "time", "event"]].dropna()

        before = len(pooled)
        pooled = (
            pooled.drop(columns=["time", "event"])
            .merge(labels, on="id", how="inner")
        )
        n_dropped = before - len(pooled)
        print(
            f"\nExternal labels applied: matched {len(pooled)}/{before} rows"
            + (f", dropped {n_dropped} unmatched IDs." if n_dropped else ".")
        )

    pooled = pooled.dropna(subset=["risk", "time", "event"])

    # --- Pooled C-index ------------------------------------------------------
    print(f"\n--- Pooled C-index ({len(csv_paths)} file(s), {len(pooled)} rows) ---")
    _cindex(pooled)


if __name__ == "__main__":
    main()
