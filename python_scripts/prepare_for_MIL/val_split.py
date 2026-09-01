"""
val_split.py — Stratified patient-level train/validation split.

Shared by define_labels.py (survival) and define_regression_labels.py
(regression) so the splitting logic lives in one place.

Validation patients (not slides) are selected, which guarantees every slide
from one patient lands in the same split.  Patients are binned into n_bins
equal-frequency quantile strata on a stratification value, then ceil(frac)
patients are sampled from each stratum so the value distribution is preserved
across train/val.
"""

import math
import os

import pandas as pd


def select_val_patients(
    df, patient_col, value_col, frac, n_bins, random_state, agg="first"
):
    """Return patient IDs for the validation set, stratified by value quantile.

    Parameters
    ----------
    df : DataFrame with one row per slide.
    patient_col : column identifying the patient (all slides from one patient
        stay in the same split).
    value_col : column to stratify on (survival time or regression target).
    frac : fraction of patients sampled into validation, applied per stratum.
    n_bins : number of equal-frequency quantile strata.
    random_state : seed for reproducible sampling.
    agg : how to reduce value_col to a single value per patient before binning.
        "first" — survival: follow-up time is constant per patient.
        "mean"  — regression: average the target across a patient's slides.
    """
    patient_df = df.groupby(patient_col)[value_col].agg(agg).reset_index()
    patient_df["stratum"] = pd.qcut(
        patient_df[value_col], q=n_bins, labels=False, duplicates="drop"
    )

    def _sample(g):
        n = max(1, math.ceil(frac * len(g)))
        return g.sample(n=n, random_state=random_state)

    return (
        patient_df.groupby("stratum", group_keys=False).apply(
            _sample, include_groups=False
        )
    )[patient_col].tolist()


def write_val_csvs(
    val_csv, iga_val_files, registry_val_files, non_iga_val_files=None
):
    """Write the validation slide lists from per-cohort val file_name frames.

    Up to four files are written:
      <val_csv>              — combined list (IgA + registry), with a header;
                               consumed by MIL.py --val_csv.
      <stem>_IgA<ext>        — IgA-only list, no header.
      <stem>_registry<ext>   — registry-only list, no header.
      <stem>_non_IgA<ext>    — non-IgA list, with a header; written only when
                               non_iga_val_files is given.  Consumed by
                               MIL.py --pretrain_val_csv.
    where <stem>/<ext> are the splitext parts of val_csv.

    The non-IgA slides are deliberately NOT merged into <val_csv>.  That list
    validates the finetune phase, which trains on IgA + registry only; the
    non-IgA slides validate the pretrain phase and are passed separately.
    Merging them would report one loss over two different data distributions.

    Parameters
    ----------
    val_csv : path of the combined validation list.
    iga_val_files, registry_val_files : single-column DataFrames (column
        "file_name") holding the validation slides for each cohort.
    non_iga_val_files : same, for the non-IgA pretrain cohort.  Pass None (the
        default) to leave pretraining unvalidated and write no non-IgA list.

    Returns the combined (IgA + registry) validation DataFrame.
    """
    val_stem, val_ext = os.path.splitext(val_csv)

    val_dir = os.path.dirname(val_csv)
    if val_dir:
        os.makedirs(val_dir, exist_ok=True)

    iga_val_files.to_csv(f"{val_stem}_IgA{val_ext}", index=False, header=False)
    registry_val_files.to_csv(
        f"{val_stem}_registry{val_ext}", index=False, header=False
    )
    if non_iga_val_files is not None:
        non_iga_val_files.to_csv(f"{val_stem}_non_IgA{val_ext}", index=False)

    combined = pd.concat([iga_val_files, registry_val_files], ignore_index=True)
    combined.to_csv(val_csv, index=False)
    return combined
