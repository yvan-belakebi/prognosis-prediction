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

    return (patient_df.groupby("stratum", group_keys=False).apply(_sample))[
        patient_col
    ].tolist()


def write_val_csvs(val_csv, iga_val_files, registry_val_files):
    """Write the validation slide lists from per-cohort val file_name frames.

    Three files are written:
      <val_csv>              — combined list (IgA + registry), with a header;
                               consumed by MIL.py --val_csv.
      <stem>_IgA<ext>        — IgA-only list, no header.
      <stem>_registry<ext>   — registry-only list, no header.
    where <stem>/<ext> are the splitext parts of val_csv.

    Parameters
    ----------
    val_csv : path of the combined validation list.
    iga_val_files, registry_val_files : single-column DataFrames (column
        "file_name") holding the validation slides for each cohort.

    Returns the combined validation DataFrame.
    """
    val_stem, val_ext = os.path.splitext(val_csv)

    iga_val_files.to_csv(f"{val_stem}_IgA{val_ext}", index=False, header=False)
    registry_val_files.to_csv(
        f"{val_stem}_registry{val_ext}", index=False, header=False
    )

    combined = pd.concat([iga_val_files, registry_val_files], ignore_index=True)
    combined.to_csv(val_csv, index=False)
    return combined
