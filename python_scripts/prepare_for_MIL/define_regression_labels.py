"""
define_regression_labels.py — Create per-slide .npy label files for eGFR regression,
ready for use with regression_MIL.py.

Two cohorts are processed in parallel, mirroring define_labels.py:

  IgA      — source column: eGFR_diagnosis  (followup_data/raw/IgA/IgA_cohort_full_data.csv)
             output dir:    WSI/IgA/labels_regression/

  Registry — source column: eGFR            (followup_data/derived/renamed/registry_anonymized.csv)
             output dir:    WSI/IgA_registry/labels_regression/

.npy files contain a scalar float64 (the eGFR value).

Outlier filter: slides with eGFR above --max_label (default 200) get no .npy at all,
which is what keeps them out of both training and validation — index_biopsies drops
slides with no label file, so nothing has to be passed at training time.  The filter
runs before the validation split, so the quantile strata are built on plausible values.

Validation split: stratified by eGFR quantile so the value distribution is
preserved across train/val.  All slides from the same patient stay in the
same split.  Use --val_source to control which cohort(s) contribute val slides.

The non-IgA cohort is the pretraining set and is handled separately: pass
--val_non_iga to hold out --val_frac of its patients, written to
<val_csv stem>_non_IgA<ext> for regression_MIL.py --pretrain_val_csv.  It is
kept out of --val_csv, which validates finetuning on IgA + registry only.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_regression_labels.py

    # IgA-only validation, 15 % val fraction:
    python python_scripts/prepare_for_MIL/define_regression_labels.py \\
        --val_source IgA --val_frac 0.15

    # Add a 20 % non-IgA validation set for the pretrain phase:
    python python_scripts/prepare_for_MIL/define_regression_labels.py \\
        --val_non_iga --val_frac 0.2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Reuse cohort loaders from define_labels.py (avoids duplicating biopsy-number
# normalisation helpers and the IgA merge logic) and the shared val-split helper.
sys.path.insert(0, os.path.dirname(__file__))
from define_labels import (
    load_iga_cohort,
    load_registry_cohort,
    load_non_iga_cohort,
    make_bag_name,
)  # noqa: E402
from val_split import select_val_patients, write_val_csvs  # noqa: E402

# ---------------------------------------------------------------------------
# Outlier filtering
# ---------------------------------------------------------------------------


def _drop_outliers(df, label_col, max_label, source):
    """Drop rows whose label exceeds ``max_label``, reporting what went.

    Must be called *before* ``select_val_patients`` and before the .npy files are
    written, for two reasons:

      * the quantile strata in select_val_patients are then computed on plausible
        values only, instead of putting 130 and 20411 in the same top stratum;
      * not writing a label file is what excludes a biopsy from training *and*
        validation — index_biopsies drops slides that have no .npy — so no flag has
        to be remembered at training, evaluation or inference time.

    eGFR above ~200 is not physiologically reachable (normal adult 90-120,
    hyperfiltration up to ~180), and the offending values are computed ones that
    recur identically across unrelated biopsies — the signature of a bad creatinine
    fed through the estimating equation.
    """
    if max_label is None:
        return df
    # NaN > x is False, so missing labels are preserved here and handled as before
    # by the dropna in _write_cohort.
    keep = ~(df[label_col] > max_label)
    dropped = df.loc[~keep]
    if len(dropped):
        print(
            f"  {source}: dropped {len(dropped)} slides / "
            f"{dropped['biopsy_number'].nunique()} biopsies / "
            f"{dropped['patient'].nunique()} patients "
            f"with {label_col} > {max_label:g} "
            f"(range {dropped[label_col].min():.1f}-{dropped[label_col].max():.1f})"
        )
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Per-cohort label writing
# ---------------------------------------------------------------------------


def _write_cohort(df, label_col, patient_col, output_dir, val_patients, split_kwargs):
    """Assign split, save .npy files, return the annotated DataFrame.

    Files are written under the bag_name path (biopsy_number/file_name when
    nested, file_name when flat); subdirectories are created as needed.
    """
    df = df.dropna(subset=[label_col]).copy()
    df["split"] = df[patient_col].isin(val_patients).map({True: "val", False: "train"})

    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        path = os.path.join(output_dir, f"{row['bag_name']}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(float(row[label_col]), dtype=np.float64))
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
    parser.add_argument(
        "--iga_slides_csv", default="followup_data/derived/renamed/IgA_full_data.csv"
    )
    parser.add_argument(
        "--iga_followup_csv", default="followup_data/raw/IgA/IgA_cohort_full_data.csv"
    )
    parser.add_argument(
        "--registry_csv",
        default="followup_data/derived/renamed/registry_anonymized.csv",
    )

    # Label columns
    parser.add_argument(
        "--iga_label_col",
        default="eGFR_diagnosis",
        help="eGFR column in raw/IgA/IgA_cohort_full_data.csv.",
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

    # Outlier filter
    parser.add_argument(
        "--max_label",
        type=float,
        default=200.0,
        help="Drop slides whose eGFR exceeds this, before the val split and before "
        "any .npy is written. eGFR above ~200 is not physiologically reachable, and "
        "the registry cohorts carry computed values up to 20411 that otherwise "
        "dominate the label variance. Pass 'inf' to disable.",
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
    parser.add_argument(
        "--val_non_iga",
        action="store_true",
        help="Also hold out --val_frac of the non-IgA patients as a validation "
        "set for the pretrain phase, written to <val_csv stem>_non_IgA.csv and "
        "consumed by regression_MIL.py --pretrain_val_csv. Off by default, "
        "which keeps every non-IgA slide in the pretrain training set.",
    )

    # Biopsy-nesting layout for the written .npy labels and validation lists.
    # The summary CSV always carries biopsy_number and file_name as separate
    # columns; this flag only controls the on-disk bag-name form (make_bag_name).
    parser.add_argument(
        "--nest_biopsy",
        dest="nest_biopsy",
        action="store_true",
        default=True,
        help="Nest .npy labels and val lists under biopsy dirs "
        "(biopsy_number/file_name). Default; matches the nested WSI layout.",
    )
    parser.add_argument(
        "--no_nest_biopsy",
        dest="nest_biopsy",
        action="store_false",
        help="Flat layout: key .npy labels and val lists by file_name only.",
    )

    # Output directories
    parser.add_argument("--iga_output_dir", default="WSI/IgA/labels_regression")
    parser.add_argument(
        "--registry_output_dir", default="WSI/IgA_registry/labels_regression"
    )
    parser.add_argument(
        "--non_iga_output_dir",
        default="WSI/non_IgA/labels_regression",
        help="Output dir for non-IgA .npy files (always train).",
    )

    # Summary outputs
    parser.add_argument(
        "--summary_csv",
        default="label_csvs/labels_regression.csv",
        help="Combined summary CSV (both cohorts, with split column).",
    )
    parser.add_argument(
        "--val_csv",
        default="validation_files_csvs/regression_validation_files.csv",
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

    iga_df = iga_full[
        ["biopsy_number", "file_name", "PERSON_NR", args.iga_label_col, "Stain"]
    ].copy()
    iga_df = iga_df.rename(columns={args.iga_label_col: "eGFR", "PERSON_NR": "patient"})
    iga_df["source"] = "IgA"
    iga_df["bag_name"] = make_bag_name(iga_df, args.nest_biopsy)
    iga_df = _drop_outliers(iga_df, "eGFR", args.max_label, "IgA")

    iga_val_patients = (
        select_val_patients(iga_df, "patient", "eGFR", agg="mean", **split_kwargs)
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

    reg_df = reg_full[
        ["biopsy_number", "file_name", "patient", args.registry_label_col, "stain"]
    ].copy()
    reg_df = reg_df.rename(columns={args.registry_label_col: "eGFR", "stain": "Stain"})
    reg_df["source"] = "registry"
    reg_df["bag_name"] = make_bag_name(reg_df, args.nest_biopsy)
    reg_df = _drop_outliers(reg_df, "eGFR", args.max_label, "registry")

    reg_val_patients = (
        select_val_patients(reg_df, "patient", "eGFR", agg="mean", **split_kwargs)
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

    # ── Non-IgA cohort (pretrain; validated only with --val_non_iga) ──────────

    non_iga_full = load_non_iga_cohort(args.registry_csv)
    non_iga_val_patients = []

    if args.registry_label_col not in non_iga_full.columns:
        print(
            f"Warning: '{args.registry_label_col}' not found in non-IgA registry data "
            f"— non-IgA will be skipped."
        )
        non_iga_df = pd.DataFrame(
            columns=[
                "biopsy_number", "file_name", "eGFR", "Stain",
                "patient", "source", "split", "bag_name",
            ]
        )
    else:
        non_iga_df = non_iga_full[
            ["biopsy_number", "file_name", "patient", args.registry_label_col, "stain"]
        ].copy()
        non_iga_df = non_iga_df.rename(
            columns={args.registry_label_col: "eGFR", "stain": "Stain"}
        )
        non_iga_df["source"] = "non_IgA"
        non_iga_df["bag_name"] = make_bag_name(non_iga_df, args.nest_biopsy)
        non_iga_df = _drop_outliers(non_iga_df, "eGFR", args.max_label, "non_IgA")
        # The non-IgA cohort is validated independently of --val_source: it feeds
        # the pretrain phase, not the finetune phase the other two lists
        # validate.  Without --val_non_iga the val list stays empty and every
        # row gets split="train".
        if args.val_non_iga:
            non_iga_val_patients = select_val_patients(
                non_iga_df, "patient", "eGFR", agg="mean", **split_kwargs
            )
        non_iga_df = _write_cohort(
            non_iga_df,
            "eGFR",
            "patient",
            args.non_iga_output_dir,
            non_iga_val_patients,
            split_kwargs,
        )
        print(
            f"non-IgA — {len(non_iga_df)} slides  "
            f"(train {(non_iga_df['split']=='train').sum()}, "
            f"val {(non_iga_df['split']=='val').sum()})"
        )
        print(
            f"  eGFR: {non_iga_df['eGFR'].min():.1f} – {non_iga_df['eGFR'].max():.1f}  "
            f"(mean {non_iga_df['eGFR'].mean():.1f})"
        )

    # ── Combined outputs ──────────────────────────────────────────────────────

    # biopsy_number and file_name (slide stem) are separate columns so downstream
    # can match flat (file_name) or nested (biopsy_number/file_name) layouts.
    _cols = ["biopsy_number", "file_name", "eGFR", "Stain", "patient", "source", "split"]
    combined = pd.concat(
        [iga_df[_cols], reg_df[_cols], non_iga_df[_cols]],
        ignore_index=True,
    )
    summary_dir = os.path.dirname(args.summary_csv)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    combined.to_csv(args.summary_csv, index=False)

    # Validation slide lists (combined + per-source) — see val_split.write_val_csvs.
    # The val lists carry the on-disk bag name under the "file_name" column header.
    val_list = write_val_csvs(
        args.val_csv,
        iga_df[iga_df["split"] == "val"][["bag_name"]].rename(
            columns={"bag_name": "file_name"}
        ),
        reg_df[reg_df["split"] == "val"][["bag_name"]].rename(
            columns={"bag_name": "file_name"}
        ),
        non_iga_val_files=(
            non_iga_df[non_iga_df["split"] == "val"][["bag_name"]].rename(
                columns={"bag_name": "file_name"}
            )
            if args.val_non_iga
            else None
        ),
    )

    print(f"\nCombined val slides: {len(val_list)}")
    print(
        f"  Combined eGFR: {combined['eGFR'].mean():.1f} ± {combined['eGFR'].std():.1f}"
    )
    print(f"\nOutputs:")
    print(f"  {args.iga_output_dir}/        ({len(iga_df)} .npy files)")
    print(f"  {args.registry_output_dir}/   ({len(reg_df)} .npy files)")
    print(f"  {args.non_iga_output_dir}/    ({len(non_iga_df)} .npy files)")
    print(f"  {args.summary_csv}")
    val_stem, val_ext = os.path.splitext(args.val_csv)
    print(f"  {args.val_csv}")
    print(f"  {val_stem}_IgA{val_ext}")
    print(f"  {val_stem}_registry{val_ext}")
    if args.val_non_iga:
        print(
            f"  {val_stem}_non_IgA{val_ext}  "
            f"(pass to regression_MIL.py --pretrain_val_csv)"
        )


if __name__ == "__main__":
    main()
