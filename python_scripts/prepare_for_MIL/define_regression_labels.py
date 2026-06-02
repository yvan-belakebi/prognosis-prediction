"""
define_regression_labels.py — Create per-slide .npy label files for eGFR regression,
ready for use with regression_MIL.py.

Two cohorts are processed in parallel, mirroring define_labels.py:

  IgA      — source column: eGFR_diagnosis  (followup_data/IgA_cohort_full_data.csv)
             output dir:    WSI/IgA/labels_regression/

  Registry — source column: eGFR            (followup_data/registry_anonymized.csv)
             output dir:    WSI/registry_IgA/labels_regression/

.npy files contain a scalar float64 (the eGFR value).

Validation split: stratified by eGFR quantile so the value distribution is
preserved across train/val.  All slides from the same patient stay in the
same split.  Use --val_source to control which cohort(s) contribute val slides.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_regression_labels.py

    # IgA-only validation, 15 % val fraction:
    python python_scripts/prepare_for_MIL/define_regression_labels.py \\
        --val_source IgA --val_frac 0.15
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

# Reuse cohort loaders from define_labels.py (avoids duplicating biopsy-number
# normalisation helpers and the IgA merge logic).
sys.path.insert(0, os.path.dirname(__file__))
from define_labels import load_iga_cohort, load_registry_cohort  # noqa: E402

# ---------------------------------------------------------------------------
# Validation-split helper
# ---------------------------------------------------------------------------


def select_val_patients(df, patient_col, value_col, frac, n_bins, random_state):
    """Return patient IDs for the val set, stratified by value quantile.

    Patients are binned into n_bins equal-frequency strata, then ceil(frac)
    patients are sampled from each stratum.  Patient-level selection ensures
    every slide from one patient lands in the same split.
    """
    patient_df = df.groupby(patient_col)[value_col].mean().reset_index()
    patient_df["stratum"] = pd.qcut(
        patient_df[value_col], q=n_bins, labels=False, duplicates="drop"
    )

    def _sample(g):
        n = max(1, math.ceil(frac * len(g)))
        return g.sample(n=n, random_state=random_state)

    return (patient_df.groupby("stratum", group_keys=False).apply(_sample))[
        patient_col
    ].tolist()


# ---------------------------------------------------------------------------
# Per-cohort label writing
# ---------------------------------------------------------------------------


def _write_cohort(df, label_col, patient_col, output_dir, val_patients, split_kwargs):
    """Assign split, save .npy files, return the annotated DataFrame."""
    df = df.dropna(subset=[label_col]).copy()
    df["split"] = df[patient_col].isin(val_patients).map({True: "val", False: "train"})

    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        np.save(
            os.path.join(output_dir, f"{row['file_name']}.npy"),
            np.array(float(row[label_col]), dtype=np.float64),
        )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build eGFR regression label files for regression_MIL.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    parser.add_argument("--iga_slides_csv", default="followup_data/IgA_slide_data.csv")
    parser.add_argument(
        "--iga_followup_csv", default="followup_data/IgA_cohort_full_data.csv"
    )
    parser.add_argument(
        "--registry_csv", default="followup_data/registry_anonymized.csv"
    )

    # Label columns
    parser.add_argument(
        "--iga_label_col",
        default="eGFR_diagnosis",
        help="eGFR column in IgA_cohort_full_data.csv.",
    )
    parser.add_argument(
        "--registry_label_col",
        default="eGFR",
        help="eGFR column in registry_anonymized.csv.",
    )

    # Date filter (consistent with define_labels.py)
    parser.add_argument(
        "--iga_date_filter",
        default="2006-01-01",
        help="Exclude IgA biopsies before this date (same filter as "
        "define_labels.py). Pass 'none' to disable.",
    )

    # Validation split
    parser.add_argument(
        "--val_source",
        choices=["IgA", "registry", "both"],
        default="both",
        help="Which cohort(s) contribute slides to the val set.",
    )
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument(
        "--n_bins", type=int, default=4, help="Quantile strata for stratified sampling."
    )
    parser.add_argument("--random_state", type=int, default=42)

    # Output directories
    parser.add_argument("--iga_output_dir", default="WSI/IgA/labels_regression")
    parser.add_argument(
        "--registry_output_dir", default="WSI/IgA_registry/labels_regression"
    )

    # Summary outputs
    parser.add_argument(
        "--summary_csv",
        default="followup_data/labels_regression.csv",
        help="Combined summary CSV (both cohorts, with split column).",
    )
    parser.add_argument(
        "--val_csv",
        default="followup_data/regression_validation_files.csv",
        help="Combined val slide list for regression_MIL.py --val_csv.",
    )

    args = parser.parse_args()

    split_kwargs = dict(
        frac=args.val_frac, n_bins=args.n_bins, random_state=args.random_state
    )

    # ── IgA cohort ────────────────────────────────────────────────────────────

    iga_full = load_iga_cohort(args.iga_slides_csv, args.iga_followup_csv)

    # Apply date filter (same logic as define_labels.py)
    if args.iga_date_filter.lower() != "none":
        iga_full["Biopsy_date"] = pd.to_datetime(
            iga_full["Biopsy_date"], errors="coerce"
        )
        iga_full = iga_full[iga_full["Biopsy_date"] >= args.iga_date_filter]

    if args.iga_label_col not in iga_full.columns:
        parser.error(
            f"--iga_label_col '{args.iga_label_col}' not found in merged IgA DataFrame. "
            f"Available: {[c for c in iga_full.columns if 'egfr' in c.lower() or 'GFR' in c]}"
        )

    iga_df = iga_full[["file_name", "PERSON_NR", args.iga_label_col, "Stain"]].copy()
    iga_df = iga_df.rename(columns={args.iga_label_col: "eGFR", "PERSON_NR": "patient"})
    iga_df["source"] = "IgA"

    iga_val_patients = (
        select_val_patients(iga_df, "patient", "eGFR", **split_kwargs)
        if args.val_source in ("IgA", "both")
        else []
    )
    iga_df = _write_cohort(
        iga_df, "eGFR", "patient", args.iga_output_dir, iga_val_patients, split_kwargs
    )

    print(
        f"IgA — {len(iga_df)} slides  "
        f"(train {(iga_df['split']=='train').sum()}, "
        f"val {(iga_df['split']=='val').sum()})"
    )
    print(
        f"  eGFR: {iga_df['eGFR'].min():.1f} – {iga_df['eGFR'].max():.1f}  "
        f"(mean {iga_df['eGFR'].mean():.1f})"
    )

    # ── Registry cohort ───────────────────────────────────────────────────────

    reg_full = load_registry_cohort(args.registry_csv)

    if args.registry_label_col not in reg_full.columns:
        parser.error(
            f"--registry_label_col '{args.registry_label_col}' not found in registry. "
            f"Available: {[c for c in reg_full.columns if 'egfr' in c.lower() or 'GFR' in c]}"
        )

    reg_df = reg_full[["file_name", "patient", args.registry_label_col, "stain"]].copy()
    reg_df = reg_df.rename(columns={args.registry_label_col: "eGFR", "stain": "Stain"})
    reg_df["source"] = "registry"

    reg_val_patients = (
        select_val_patients(reg_df, "patient", "eGFR", **split_kwargs)
        if args.val_source in ("registry", "both")
        else []
    )
    reg_df = _write_cohort(
        reg_df,
        "eGFR",
        "patient",
        args.registry_output_dir,
        reg_val_patients,
        split_kwargs,
    )

    print(
        f"Registry — {len(reg_df)} slides  "
        f"(train {(reg_df['split']=='train').sum()}, "
        f"val {(reg_df['split']=='val').sum()})"
    )
    print(
        f"  eGFR: {reg_df['eGFR'].min():.1f} – {reg_df['eGFR'].max():.1f}  "
        f"(mean {reg_df['eGFR'].mean():.1f})"
    )

    # ── Combined outputs ──────────────────────────────────────────────────────

    combined = pd.concat(
        [
            iga_df[["file_name", "eGFR", "Stain", "patient", "source", "split"]],
            reg_df[["file_name", "eGFR", "Stain", "patient", "source", "split"]],
        ],
        ignore_index=True,
    )
    combined.to_csv(args.summary_csv, index=False)

    val_list = combined[combined["split"] == "val"][["file_name"]]
    val_list.to_csv(args.val_csv, index=False)

    # Per-source val files (same naming convention as define_labels.py)
    val_stem, val_ext = os.path.splitext(args.val_csv)
    iga_df[iga_df["split"] == "val"][["file_name"]].to_csv(
        f"{val_stem}_IgA{val_ext}", index=False, header=False
    )
    reg_df[reg_df["split"] == "val"][["file_name"]].to_csv(
        f"{val_stem}_registry{val_ext}", index=False, header=False
    )

    print(f"\nCombined val slides: {len(val_list)}")
    print(
        f"  Combined eGFR: {combined['eGFR'].mean():.1f} ± {combined['eGFR'].std():.1f}"
    )
    print(f"\nOutputs:")
    print(f"  {args.iga_output_dir}/        ({len(iga_df)} .npy files)")
    print(f"  {args.registry_output_dir}/   ({len(reg_df)} .npy files)")
    print(f"  {args.summary_csv}")
    print(f"  {args.val_csv}")
    print(f"  {val_stem}_IgA{val_ext}")
    print(f"  {val_stem}_registry{val_ext}")


if __name__ == "__main__":
    main()
