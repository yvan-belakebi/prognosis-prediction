"""
val_split.py — Stratified patient-level train/validation/test splits.

Shared by define_labels.py (survival) and define_regression_labels.py
(regression) so the splitting logic lives in one place.

Validation patients (not slides) are selected, which guarantees every slide
from one patient lands in the same split.  Patients are binned into n_bins
equal-frequency quantile strata on a stratification value, then ceil(frac)
patients are sampled from each stratum so the value distribution is preserved
across train/val.
"""

import math

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


def select_patients_by_event(df, patient_col, event_col, frac, random_state):
    """Return patient IDs for a held-out set, stratified by event status.

    A patient counts as an event when any of their slides carries an event
    (a patient with several biopsies can have different outcomes per biopsy).
    ``frac`` of the event patients and ``frac`` of the event-free patients are
    sampled, so the held-out set has the cohort's event rate.
    """
    patient_event = df.groupby(patient_col)[event_col].max().reset_index()
    picked = []
    for _, g in patient_event.groupby(event_col):
        n = round(frac * len(g))
        if n:
            picked += g.sample(n=n, random_state=random_state)[patient_col].tolist()
    return picked

