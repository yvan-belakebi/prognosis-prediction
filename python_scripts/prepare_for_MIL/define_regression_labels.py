"""
define_regression_labels.py — Create per-slide .npy label files for eGFR regression,
ready for use with regression_MIL.py.

Reads the same single registry-style input CSV as define_labels.py (same --query
and --*_col arguments) and writes:

  <labels_dir>/<file_name>.npy   One flat scalar float64 (the eGFR value) per slide.
  <labels_csv>                   One row per slide: file_name, biopsy_number,
                                 patient, stain, eGFR, split.

Outlier filter: slides with eGFR above --max_label (default 200) get no .npy at all,
which is what keeps them out of both training and validation — slides with no label
file are dropped at load time, so nothing has to be passed at training time.  The
filter runs before the validation split, so the quantile strata are built on
plausible values.

Validation split: patients are sampled stratified by eGFR quantile, so the value
distribution is preserved across train/val and all slides of a patient stay in the
same split.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_regression_labels.py \\
        --input_csv  followup_data/derived/renamed/registry_anonymized.csv \\
        --query      "is_IgA == True" \\
        --labels_dir WSI/IgA/trident/labels_regression \\
        --labels_csv label_csvs/labels_regression.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Reuse the input loader, split and CSV helpers from define_labels.py.
sys.path.insert(0, os.path.dirname(__file__))
from define_labels import (  # noqa: E402
    add_source_args,
    add_split_args,
    assign_split,
    load_source,
    write_labels_csv,
)


def _drop_outliers(df, label_col, max_label):
    """Drop rows whose label exceeds ``max_label``, reporting what went.

    Must be called *before* the split and before the .npy files are written, for
    two reasons:

      * the quantile strata in select_val_patients are then computed on plausible
        values only, instead of putting 130 and 20411 in the same top stratum;
      * not writing a label file is what excludes a biopsy from training *and*
        validation — slides with no .npy are dropped at load time — so no flag has
        to be remembered at training, evaluation or inference time.

    eGFR above ~200 is not physiologically reachable (normal adult 90-120,
    hyperfiltration up to ~180), and the offending values are computed ones that
    recur identically across unrelated biopsies — the signature of a bad creatinine
    fed through the estimating equation.
    """
    keep = ~(df[label_col] > max_label)
    dropped = df.loc[~keep]
    if len(dropped):
        print(
            f"Dropped {len(dropped)} slides / "
            f"{dropped['biopsy_number'].nunique()} biopsies / "
            f"{dropped['patient'].nunique()} patients "
            f"with {label_col} > {max_label:g} "
            f"(range {dropped[label_col].min():.1f}-{dropped[label_col].max():.1f})"
        )
    return df[keep].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Build eGFR regression label files for regression_MIL.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_source_args(parser)
    parser.add_argument("--label_col", default="eGFR", help="Regression target column.")
    parser.add_argument(
        "--max_label",
        type=float,
        default=200.0,
        help="Drop slides whose eGFR exceeds this, before the val split and before "
        "any .npy is written. eGFR above ~200 is not physiologically reachable, and "
        "the registry carries computed values up to 20411 that otherwise dominate "
        "the label variance. Pass 'inf' to disable.",
    )
    add_split_args(parser)
    parser.add_argument(
        "--labels_dir", required=True, help="Output folder for the per-slide .npy labels."
    )
    parser.add_argument(
        "--labels_csv",
        default="label_csvs/labels_regression.csv",
        help="Output labels CSV.",
    )
    args = parser.parse_args()

    df = load_source(args)
    if args.label_col not in df.columns:
        parser.error(f"--label_col '{args.label_col}' not found in {args.input_csv}.")
    df["eGFR"] = pd.to_numeric(df[args.label_col], errors="coerce")
    n_rows = len(df)
    df = df.dropna(subset=["eGFR"])
    if len(df) < n_rows:
        print(f"Dropped {n_rows - len(df)} slides with no {args.label_col}.")
    df = _drop_outliers(df, "eGFR", args.max_label)
    df = assign_split(df, "eGFR", args, agg="mean")

    os.makedirs(args.labels_dir, exist_ok=True)
    for row in df.itertuples(index=False):
        np.save(
            os.path.join(args.labels_dir, f"{row.file_name}.npy"),
            np.array(float(row.eGFR), dtype=np.float64),
        )
    write_labels_csv(
        df,
        ["file_name", "biopsy_number", "patient", "stain", "eGFR", "split"],
        args.labels_csv,
    )

    for split in ("train", "val"):
        part = df[df["split"] == split]
        print(
            f"  {split:5s} — slides: {len(part):>6d}  biopsies: "
            f"{part['biopsy_number'].nunique():>5d}  patients: "
            f"{part['patient'].nunique():>5d}  eGFR: {part['eGFR'].mean():.1f} "
            f"± {part['eGFR'].std():.1f}"
        )
    print("\nOutputs:")
    print(f"  {args.labels_dir}/  ({len(df)} .npy files)")
    print(f"  {args.labels_csv}")


if __name__ == "__main__":
    main()
