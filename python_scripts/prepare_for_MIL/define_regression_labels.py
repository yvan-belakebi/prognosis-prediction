"""
define_regression_labels.py — Create per-slide .npy label files for eGFR regression,
ready for use with regression_MIL.py.

Source: eGFR column in followup_data/registry_anonymized.csv (IgA patients only)

The .npy files contain a scalar float64 (the eGFR value).

Validation split: stratified by eGFR quantile so the value distribution is
preserved across train/val.  All slides from the same patient stay in the
same split.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_regression_labels.py

    # Custom target column or fraction:
    python python_scripts/prepare_for_MIL/define_regression_labels.py \\
        --label_col eGFR --val_frac 0.2 --output_dir WSI/registry_IgA/labels_regression
"""

import argparse
import math
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Validation-split helper
# ---------------------------------------------------------------------------

def select_val_patients(df, patient_col, value_col, frac, n_bins, random_state):
    """Return patient IDs for the validation set, stratified by value quantile.

    Patients are binned into n_bins equal-frequency strata on value_col, then
    ceil(frac) of patients is sampled from each stratum.  Patient-level
    selection guarantees every slide from one patient lands in the same split.
    """
    patient_df = df.groupby(patient_col)[value_col].mean().reset_index()
    patient_df["stratum"] = pd.qcut(
        patient_df[value_col], q=n_bins, labels=False, duplicates="drop"
    )

    def _sample(g):
        n = max(1, math.ceil(frac * len(g)))
        return g.sample(n=n, random_state=random_state)

    return (
        patient_df.groupby("stratum", group_keys=False).apply(_sample)
    )[patient_col].tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build eGFR regression label files for regression_MIL.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registry_csv", default="followup_data/registry_anonymized.csv",
    )
    parser.add_argument(
        "--label_col", default="eGFR",
        help="Column in registry_csv to use as the regression target.",
    )
    parser.add_argument(
        "--patient_col", default="ID_diagnosis",
        help="Column identifying patients (used for patient-level train/val split).",
    )
    parser.add_argument(
        "--output_dir", default="WSI/registry_IgA/labels_regression",
        help="Directory for .npy label files.",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.2,
    )
    parser.add_argument(
        "--n_bins", type=int, default=4,
        help="Number of quantile strata for stratified sampling.",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
    )
    parser.add_argument(
        "--summary_csv", default="followup_data/labels_regression.csv",
        help="Output summary CSV with file_name, label, split columns.",
    )
    parser.add_argument(
        "--val_csv", default="followup_data/regression_validation_files.csv",
        help="Output validation file list consumed by regression_MIL.py --val_csv.",
    )
    args = parser.parse_args()

    # ── Load registry ----------------------------------------------------------
    df = pd.read_csv(args.registry_csv)
    df = df[df["is_IgA"] == True].copy()

    if args.label_col not in df.columns:
        raise ValueError(
            f"Column '{args.label_col}' not found in {args.registry_csv}. "
            f"Available: {list(df.columns)}"
        )
    if args.patient_col not in df.columns:
        raise ValueError(
            f"Patient column '{args.patient_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    df = df.rename(columns={"ANON_name": "file_name"})
    df = df[["file_name", args.patient_col, args.label_col, "Stain"]].copy()
    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    print(f"Loaded {len(df)} registry IgA slides with {args.label_col}.")
    print(f"  {args.label_col} range: {df[args.label_col].min():.1f} – "
          f"{df[args.label_col].max():.1f}  (mean {df[args.label_col].mean():.1f})")

    # ── Validation split -------------------------------------------------------
    val_patients = select_val_patients(
        df, args.patient_col, args.label_col,
        frac=args.val_frac, n_bins=args.n_bins, random_state=args.random_state,
    )
    df["split"] = df[args.patient_col].isin(val_patients).map({True: "val", False: "train"})

    # ── Save .npy files --------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    for _, row in df.iterrows():
        np.save(
            os.path.join(args.output_dir, f"{row['file_name']}.npy"),
            np.array(float(row[args.label_col]), dtype=np.float64),
        )

    # ── Save CSVs --------------------------------------------------------------
    df[["file_name", args.label_col, "Stain", args.patient_col, "split"]].to_csv(
        args.summary_csv, index=False
    )
    df[df["split"] == "val"][["file_name"]].to_csv(args.val_csv, index=False)

    # ── Summary ---------------------------------------------------------------
    n_train = (df["split"] == "train").sum()
    n_val   = (df["split"] == "val").sum()
    print(f"\nSplit  — train: {n_train}  val: {n_val}")
    print(f"  Train {args.label_col}: {df[df['split']=='train'][args.label_col].mean():.1f} ± "
          f"{df[df['split']=='train'][args.label_col].std():.1f}")
    print(f"  Val   {args.label_col}: {df[df['split']=='val'][args.label_col].mean():.1f} ± "
          f"{df[df['split']=='val'][args.label_col].std():.1f}")
    print(f"\nOutputs:")
    print(f"  {args.output_dir}/  ({len(df)} .npy files)")
    print(f"  {args.summary_csv}")
    print(f"  {args.val_csv}")


if __name__ == "__main__":
    main()
