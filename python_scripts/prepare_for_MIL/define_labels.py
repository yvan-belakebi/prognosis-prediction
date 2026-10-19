"""
define_labels.py — Build survival labels and the train/val/test split for MIL.py.

One registry-style input CSV (one row per slide) is turned into:

  <labels_dir>/<file_name>.npy   One flat label file per slide, float64:
                                 [time, event] (or [time, event, code_idx] with
                                 --with_diagnosis). Consumed by --labels_path.
  <labels_csv>                   One row per slide: file_name, biopsy_number,
                                 patient, stain, time, event, diagnosis, split.
                                 The single source of truth downstream: biopsy
                                 grouping, stain selection and the
                                 train/val/test split are all read from it
                                 (--labels_csv).
  <labels_csv dir>/diagnosis_codes.json   With --with_diagnosis only.

Patients (not slides) are split, so every slide of a patient lands in the same
split (see val_split.py): first a test set (--test_frac, default 20%) stratified
by event status, then a validation set (--val_frac, default 20%) stratified by
follow-up time; the rest is train. The test set is for the final comparison of
models (e.g. stain fusion strategies): training and checkpoint selection never
see it — evaluate it once, at the end, with evaluate_survival.py --split test.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_labels.py \\
        --input_csv  followup_data/derived/renamed/registry_anonymized.csv \\
        --query      "is_IgA == True" \\
        --labels_dir WSI/IgA/trident/labels \\
        --labels_csv label_csvs/labels.csv

--query is any pandas DataFrame.query expression over the input columns, e.g.
"is_IgA == True" or "Biopsy_date >= '2006-01-01'". The --*_col arguments name
the input columns; their defaults match registry_anonymized.csv.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Shared stratified patient-level split helpers.
sys.path.insert(0, os.path.dirname(__file__))
from val_split import select_patients_by_event, select_val_patients  # noqa: E402

# ── input ─────────────────────────────────────────────────────────────────────


def add_source_args(parser):
    """Input CSV, row filter and column-name arguments shared with
    define_regression_labels.py. Defaults match registry_anonymized.csv."""
    parser.add_argument(
        "--input_csv",
        default="followup_data/derived/renamed/registry_anonymized.csv",
        help="Input CSV with one row per slide.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional pandas query selecting the rows to keep, "
        'e.g. "is_IgA == True".',
    )
    parser.add_argument("--slide_col", default="ANON_name", help="Slide stem column.")
    parser.add_argument("--biopsy_col", default="biop_number", help="Biopsy id column.")
    parser.add_argument(
        "--patient_col", default="AnonymizedID", help="Patient id column."
    )
    parser.add_argument("--stain_col", default="Stain", help="Stain column.")
    parser.add_argument(
        "--diagnosis_col",
        default="Diagnosis",
        help="Free-text diagnosis column (used by --with_diagnosis).",
    )


def load_source(args):
    """Read --input_csv, apply --query, and add the standard columns.

    Adds file_name, biopsy_number, patient and stain (string ids, stripped);
    every input column is kept alongside so callers can read their label column.
    Rows without a slide, biopsy or patient id are dropped.
    """
    id_cols = [args.slide_col, args.biopsy_col, args.patient_col]
    df = pd.read_csv(args.input_csv, dtype={c: str for c in id_cols}, low_memory=False)
    for col in id_cols + [args.stain_col]:
        if col not in df.columns:
            raise KeyError(f"{args.input_csv} has no '{col}' column.")
    if args.query:
        df = df.query(args.query)
    df = df.dropna(subset=id_cols).copy()
    df["file_name"] = df[args.slide_col].str.strip()
    df["biopsy_number"] = df[args.biopsy_col].str.strip()
    df["patient"] = df[args.patient_col].str.strip()
    df["stain"] = df[args.stain_col]
    dup = df["file_name"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"{dup.sum()} rows share a slide name, e.g. "
            f"{df.loc[dup, 'file_name'].iloc[0]!r}; file_name must be unique."
        )
    return df


def assign_split(df, value_col, args, agg="first", event_col=None):
    """Add a split column: test, val or train, assigned per patient.

    The test patients are drawn first — stratified by event status when
    ``event_col`` is given (survival), else by ``value_col`` quantile
    (regression) — so the test set does not change when --val_frac does. Val
    patients are then drawn from the rest, stratified by ``value_col``
    quantile. Both fractions are of all patients: the defaults give 60/20/20.
    """
    test_patients = []
    if args.test_frac > 0:
        if event_col is not None:
            test_patients = select_patients_by_event(
                df, "patient", event_col, args.test_frac, args.random_state
            )
        else:
            test_patients = select_val_patients(
                df, "patient", value_col, frac=args.test_frac,
                n_bins=args.n_bins, random_state=args.random_state, agg=agg,
            )
    is_test = df["patient"].isin(test_patients)
    val_patients = select_val_patients(
        df[~is_test],
        "patient",
        value_col,
        # Rescaled so --val_frac stays a fraction of all patients.
        frac=args.val_frac / (1 - args.test_frac),
        n_bins=args.n_bins,
        random_state=args.random_state,
        agg=agg,
    )
    df["split"] = np.where(
        is_test, "test", np.where(df["patient"].isin(val_patients), "val", "train")
    )
    return df


def print_split_summary(df, extra=lambda part: ""):
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        if part.empty:
            continue
        print(
            f"  {split:5s} — slides: {len(part):>6d}  biopsies: "
            f"{part['biopsy_number'].nunique():>5d}  patients: "
            f"{part['patient'].nunique():>5d}{extra(part)}"
        )


def add_split_args(parser):
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of all patients assigned to validation.",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Fraction of all patients held out as a test set, drawn first "
        "(stratified by event for survival labels). Never used for training or "
        "model selection; 0 disables it.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=4,
        help="Quantile strata for stratified patient sampling.",
    )
    parser.add_argument("--random_state", type=int, default=42)


def write_labels_csv(df, cols, path):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df[cols].to_csv(path, index=False)


# ── diagnosis codes ───────────────────────────────────────────────────────────


def normalize_diagnosis(val):
    """Strip a free-text diagnosis to a canonical string, or None if missing."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def build_diagnosis_code_map(diagnoses):
    """Map distinct diagnosis strings to integer codes 1..N (0 = unknown/missing).

    Index 0 is reserved for missing/unseen diagnoses (a neutral zero embedding in
    the model via padding_idx=0), so the first real category starts at 1.

    Returns
    -------
    (code_map, n_codes)
        code_map : dict[str, int] mapping each diagnosis string to its index.
        n_codes  : total vocabulary size including the reserved 0 slot.
    """
    uniq = sorted({d for d in map(normalize_diagnosis, diagnoses) if d is not None})
    code_map = {name: i + 1 for i, name in enumerate(uniq)}
    return code_map, len(uniq) + 1


def write_npy_labels(df, output_dir, code_map=None):
    """Write one <file_name>.npy per slide with the survival label (float64).

    Without ``code_map`` each file holds ``[time, event]`` (shape (2,)).  When a
    ``code_map`` is provided, a third element is appended — the integer diagnosis
    code from the row's ``diagnosis`` column, or 0 when missing/unseen — giving
    ``[time, event, code_idx]`` (shape (3,)) consumed by MIL.py --use_diagnosis.
    """
    os.makedirs(output_dir, exist_ok=True)
    for row in df.itertuples(index=False):
        values = [float(row.time), float(row.event)]
        if code_map is not None:
            values.append(float(code_map.get(row.diagnosis, 0)))
        np.save(
            os.path.join(output_dir, f"{row.file_name}.npy"),
            np.array(values, dtype=np.float64),
        )
    return len(df)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build survival labels and the train/val split for MIL.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_source_args(parser)
    parser.add_argument(
        "--time_col",
        default="time_to_event",
        help="Follow-up time in days; the first number is extracted, so values "
        "like '62 days' work.",
    )
    parser.add_argument(
        "--event_col",
        default="Event",
        help="Event column; event = 1 where it is non-empty, 0 where it is empty.",
    )
    add_split_args(parser)
    parser.add_argument(
        "--labels_dir",
        required=True,
        help="Output folder for the per-slide .npy labels.",
    )
    parser.add_argument(
        "--labels_csv", default="label_csvs/labels.csv", help="Output labels CSV."
    )
    # Diagnosis-code late fusion (opt-in). When set, .npy labels get a third
    # element [time, event, code_idx] and a diagnosis_codes.json vocabulary is
    # written for MIL.py --use_diagnosis. Default off keeps (2,) labels.
    parser.add_argument(
        "--with_diagnosis",
        action="store_true",
        help="Append an integer diagnosis code to each .npy label and write "
        "diagnosis_codes.json next to --labels_csv (for MIL.py --use_diagnosis).",
    )
    args = parser.parse_args()

    df = load_source(args)
    df["time"] = pd.to_numeric(
        df[args.time_col].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )
    df["event"] = df[args.event_col].notna().astype(int)
    df["diagnosis"] = (
        df[args.diagnosis_col].apply(normalize_diagnosis)
        if args.diagnosis_col in df.columns
        else None
    )
    n_rows = len(df)
    df = df.dropna(subset=["time"])
    if len(df) < n_rows:
        print(f"Dropped {n_rows - len(df)} slides with no follow-up time.")

    df = assign_split(df, "time", args, event_col="event")

    code_map = None
    if args.with_diagnosis:
        code_map, n_codes = build_diagnosis_code_map(df["diagnosis"])
        codes_path = os.path.join(
            os.path.dirname(args.labels_csv), "diagnosis_codes.json"
        )
        os.makedirs(os.path.dirname(codes_path) or ".", exist_ok=True)
        with open(codes_path, "w", encoding="utf-8") as f:
            json.dump(
                {"n_codes": n_codes, "codes": code_map}, f, ensure_ascii=False, indent=2
            )

    n_npy = write_npy_labels(df, args.labels_dir, code_map=code_map)
    write_labels_csv(
        df,
        [
            "file_name",
            "biopsy_number",
            "patient",
            "stain",
            "time",
            "event",
            "diagnosis",
            "split",
        ],
        args.labels_csv,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    # Event counts per slide over-count multi-slide biopsies; report per biopsy.
    print_split_summary(
        df,
        lambda part: (
            f"  biopsy events: "
            f"{int(part.groupby('biopsy_number')['event'].first().sum()):>5d} "
            f"({100 * part.groupby('biopsy_number')['event'].first().mean():.1f}%)"
        ),
    )
    print("\nOutputs:")
    print(f"  {args.labels_dir}/  ({n_npy} .npy files)")
    print(f"  {args.labels_csv}")
    if code_map is not None:
        print(
            f"  {codes_path}  ({n_codes} diagnosis codes incl. 0=unknown; "
            f"labels written as [time, event, code_idx])"
        )


if __name__ == "__main__":
    main()
