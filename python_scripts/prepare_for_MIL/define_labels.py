"""
define_labels.py — Build survival labels and the train/val split for MIL.py.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_labels.py [options]

Key outputs
-----------
  Per-slide .npy label files (shape (2,), float64: [time, event]) consumed by
  MIL.py --labels_paths.  Mirrors define_regression_labels.py.  By default the
  bag path is nested (biopsy_number/file_name); pass --no_nest_biopsy for a flat
  layout (file_name only).  Files are written under each cohort's output dir:
      <iga_output_dir>/        (default WSI/IgA/labels)
      <registry_output_dir>/   (default WSI/IgA_registry/labels)
      <non_iga_output_dir>/    (default WSI/non_IgA/labels — pretrain cohort)

  <output_dir>/labels_unfiltered.csv        All slides (IgA + registry + non-IgA) with
                                            biopsy_number, file_name, time, event,
                                            stain, source, split.  biopsy_number and
                                            file_name (slide stem) are separate columns
                                            so downstream can match flat (file_name) or
                                            nested (biopsy_number/file_name) layouts.
                                            non-IgA slides are split only with
                                            --val_non_iga, otherwise all train.
  <val_csv>                                 Flat list of validation slide names
                                            (file_name column); consumed directly
                                            by MIL.py --val_csv.
  <val_csv stem>_non_IgA<ext>               Written with --val_non_iga: the non-IgA
                                            validation slides for the pretrain
                                            phase; pass to MIL.py
                                            --pretrain_val_csv.  Kept out of
                                            <val_csv>, which validates finetuning
                                            on IgA + registry only.
  <output_dir>/full_data.csv               IgA cohort with all clinical columns.

Parameters
----------
--val_source    IgA | registry | both  (default: both)
                Which cohort(s) contribute slides to the finetune validation set.
--val_non_iga   Also hold out --val_frac of the non-IgA patients as a pretrain
                validation set (default: off, i.e. all non-IgA slides train).
--val_frac      Fraction of patients assigned to validation  (default: 0.2)
--n_bins        Quantile strata for time-stratified patient sampling  (default: 4)
--random_state  Random seed  (default: 42)
--val_csv       Full path of the combined validation list for MIL.py --val_csv
                (default: validation_files_csvs/survival_validation_files.csv)
--output_dir    Directory for all other CSV outputs  (default: label_csvs)
--iga_slides_csv
--iga_followup_csv
--registry_csv  Input file paths (rarely need changing).
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# Shared stratified train/val split helper.
sys.path.insert(0, os.path.dirname(__file__))
from val_split import select_val_patients, write_val_csvs  # noqa: E402

try:
    from itables import init_notebook_mode

    init_notebook_mode(connected=True)
except Exception:
    pass


# ── helpers ───────────────────────────────────────────────────────────────────


def transform_label(label):
    """Normalise biopsy-number format to 'number/year' (e.g. 'B2312' → '12/23').

    The leading prefix can be one or two letters of any case (e.g. 'B', 'AB').
    """
    if pd.isna(label):
        return label
    label = str(label).strip().replace("\xa0", " ")
    m = re.match(r"^[A-Za-z]{1,2}(\d{2})\s+(\d+)$", label)
    if m:
        return f"{m.group(2).lstrip('0')}/{m.group(1)}"
    m = re.match(r"^[A-Za-z]{1,2}(\d{2})(\d+)$", label)
    if m:
        return f"{m.group(2).lstrip('0')}/{m.group(1)}"
    return label


def extract_file_name(file_location):
    """Return the stem of a file path (no directory, no extension)."""
    if pd.isna(file_location):
        return file_location
    path = str(file_location).strip().replace("\xa0", " ")
    basename = re.split(r"[\\\\/]", path)[-1]
    return os.path.splitext(basename.split("?")[0].split("#")[0].strip())[0]


def biopsy_to_dirname(biopsy_nr):
    """Convert a normalised biopsy number to a filesystem-safe directory name.

    The normalised form produced by transform_label uses '/' as a separator
    (e.g. '34/12').  This function replaces that separator with '-' so it can
    be used safely as a directory component on all platforms.

    Examples: '34/12' → '34-12',  '1/23' → '1-23'
    """
    if pd.isna(biopsy_nr) or str(biopsy_nr).strip() in ("", "nan", "None"):
        return "unknown"
    return str(biopsy_nr).replace("/", "-").strip()


def make_bag_name(df, nest_biopsy):
    """Build the on-disk bag name from the biopsy_number and file_name columns.

    The bag name is what discover_bags() returns and what MIL.py/regression_MIL.py
    match against, so it must mirror the WSI feature directory layout:

        nested (default) : 'biopsy_number/file_name'   (e.g. '34-12/slide_abc')
        flat             : 'file_name'                  (e.g. 'slide_abc')

    file_name holds only the slide stem; biopsy_number holds the biopsy
    directory name.  Keeping them as separate CSV columns lets downstream code
    reconstruct either layout — this helper picks one for the .npy label files
    and validation lists written by this script.
    """
    if nest_biopsy:
        return df["biopsy_number"].astype(str) + "/" + df["file_name"].astype(str)
    return df["file_name"].astype(str)


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
    """Write one .npy per slide with the survival label (float64).

    Without ``code_map`` each file holds ``[time, event]`` (shape (2,)).  When a
    ``code_map`` is provided, a third element is appended — the integer diagnosis
    code from the row's ``diagnosis`` column, or 0 when missing/unseen — giving
    ``[time, event, code_idx]`` (shape (3,)) consumed by MIL.py --use_diagnosis.

    Mirrors define_regression_labels._write_cohort: rows with a missing time or
    event are dropped, and the bag_name path (biopsy_number/file_name when
    nested, file_name when flat) is recreated as subdirectories under
    output_dir.  Returns the number of label files written.
    """
    df = df.dropna(subset=["time", "event"]).copy()
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        path = os.path.join(output_dir, f"{row['bag_name']}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        values = [float(row["time"]), float(row["event"])]
        if code_map is not None:
            code = code_map.get(normalize_diagnosis(row.get("diagnosis")), 0)
            values.append(float(code))
        np.save(path, np.array(values, dtype=np.float64))
    return len(df)


def _follow_up_years(row):
    """Compute follow-up length in years for a single IgA cohort row."""
    if row["RRT_or_death"] != "Yes":
        return row["Length_follow_up"]
    if (
        pd.notna(row["Year_RRT_or_death"])
        and pd.notna(row["ESKD_year"])
        and pd.notna(row["Biopsy_year"])
    ):
        return min(row["Year_RRT_or_death"], row["ESKD_year"]) - row["Biopsy_year"]
    if pd.notna(row["Year_RRT_or_death"]) and pd.notna(row["Biopsy_year"]):
        return row["Year_RRT_or_death"] - row["Biopsy_year"]
    if pd.notna(row["ESKD_year"]) and pd.notna(row["Biopsy_year"]):
        return row["ESKD_year"] - row["Biopsy_year"]
    return None


# ── cohort loaders ────────────────────────────────────────────────────────────


def load_iga_cohort(iga_slides_csv, iga_followup_csv):
    """Return the IgA cohort DataFrame with time (days), event, file_name, source."""
    slides = pd.read_csv(iga_slides_csv)
    slides["Biopsy Number"] = slides["Biopsy Number"].astype(str).apply(transform_label)
    slides = slides[["Biopsy Number", "File Location", "Slide ID", "Stain"]]

    followup = pd.read_csv(iga_followup_csv)
    followup.rename(columns={"Biopsy_nr": "Biopsy Number"}, inplace=True)

    df = pd.merge(slides, followup, on="Biopsy Number", how="inner")
    df["time"] = df.apply(_follow_up_years, axis=1) * 365.25  # years → days
    df["event"] = (df["RRT_or_death"] == "Yes").astype(int)
    # file_name is the slide stem; biopsy_number is the biopsy directory name.
    # They are kept separate so downstream can reconstruct either the flat
    # ('file_name') or nested ('biopsy_number/file_name') layout — see make_bag_name.
    df["file_name"] = df["File Location"].apply(extract_file_name)
    df["biopsy_number"] = df["Biopsy Number"].apply(biopsy_to_dirname)
    df["source"] = "IgA"
    return df


def _parse_registry_time_event(df):
    """Add time (days), event, file_name, and biopsy_number columns to registry data."""
    df["time"] = (
        df["time_to_event"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    )
    df["event"] = df["Event"].notna().astype(int)
    # file_name is the slide stem; biopsy_number is the biopsy directory name
    # (kept separate so downstream can pick flat or nested layout — see make_bag_name).
    df["file_name"] = df["ANON_name"].astype(str)
    # The raw biop_number is used verbatim, NOT via transform_label/biopsy_to_dirname.
    # That pair matches the leading lab code without capturing it, so 'B1310959' and
    # 'BG1310959' — two different patients' biopsies from two different labs — both
    # normalise to '10959-13' and get merged into one directory (23 such collisions;
    # see build_biopsy_name_mapping.py).  The registry WSI/feature/label directories are
    # named with the raw form by apply_biopsy_rename.py, so this must match it exactly.
    #
    # The raw values are safe to use as directory names: all of them match
    # ^[A-Za-z]{1,2}\d{2}\d+$ with no whitespace, separators or nulls.
    #
    # The IgA cohort (load_iga_cohort) still goes through transform_label, because its
    # directories are not renamed — its only normalisation clash ('B14 1974' with
    # 'B1401974') is one biopsy written two ways, which should stay merged.
    df["biopsy_number"] = df["biop_number"].astype(str).str.strip()
    df.rename(
        columns={"ID_diagnosis": "patient", "Stain": "stain"},
        inplace=True,
    )
    return df


def load_registry_cohort(registry_csv):
    """Return the IgA registry cohort (is_IgA == True) with time, event, file_name."""
    df = pd.read_csv(registry_csv, low_memory=False)
    df = _parse_registry_time_event(df[df["is_IgA"] == True].copy())
    df["source"] = "registry"
    return df


def load_non_iga_cohort(registry_csv):
    """Return the non-IgA registry cohort (is_IgA == False) with time, event, file_name.

    Non-IgA slides are always assigned to the training set; they are never
    included in the validation split.
    """
    df = pd.read_csv(registry_csv, low_memory=False)
    df = _parse_registry_time_event(df[df["is_IgA"] == False].copy())
    df["source"] = "non_IgA"
    return df


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build survival labels and train/val split CSVs for MIL.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Date filter (consistent with define_regression_labels.py)
    parser.add_argument(
        "--iga_date_filter",
        default="2006-01-01",
        help="Exclude IgA biopsies before this date (IgA biopsies before 2006 "
        "have very short follow-up). Pass 'none' to disable.",
    )

    # Validation split
    parser.add_argument(
        "--val_source",
        choices=["IgA", "registry", "both"],
        default="both",
        help="Cohort(s) that contribute slides to the validation set.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of patients assigned to validation.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=4,
        help="Number of quantile strata for time-stratified patient sampling.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--val_non_iga",
        action="store_true",
        help="Also hold out --val_frac of the non-IgA patients as a validation "
        "set for the pretrain phase, written to <val_csv stem>_non_IgA.csv and "
        "consumed by MIL.py --pretrain_val_csv. Off by default, which keeps "
        "every non-IgA slide in the pretrain training set.",
    )

    # Output paths
    parser.add_argument(
        "--val_csv",
        default="validation_files_csvs/survival_validation_files.csv",
        help=(
            "Output path for the combined validation slide list consumed by "
            "MIL.py --val_csv."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="label_csvs",
        help="Directory for all other CSV outputs.",
    )

    # Diagnosis-code late fusion (opt-in). When set, .npy labels get a third
    # element [time, event, code_idx] and a diagnosis_codes.json vocabulary is
    # written for MIL.py --use_diagnosis. Default off keeps (2,) labels.
    parser.add_argument(
        "--with_diagnosis",
        action="store_true",
        help="Append an integer diagnosis code to each .npy label and write "
        "<output_dir>/diagnosis_codes.json (for MIL.py --use_diagnosis).",
    )

    # Biopsy-nesting layout for the written .npy labels and validation lists.
    # The CSVs always carry biopsy_number and file_name as separate columns;
    # this flag only controls the on-disk bag-name form (see make_bag_name).
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

    # Per-cohort .npy label output directories (consumed by MIL.py --labels_paths)
    parser.add_argument(
        "--iga_output_dir",
        default="WSI/IgA/labels",
        help="Output dir for IgA per-slide .npy survival labels.",
    )
    parser.add_argument(
        "--registry_output_dir",
        default="WSI/IgA_registry/labels",
        help="Output dir for registry per-slide .npy survival labels.",
    )
    parser.add_argument(
        "--non_iga_output_dir",
        default="WSI/non_IgA/labels",
        help="Output dir for non-IgA per-slide .npy survival labels (always train).",
    )

    # Input paths
    parser.add_argument(
        "--iga_slides_csv",
        default="followup_data/derived/renamed/IgA_full_data.csv",
        help="Slide metadata for the IgA cohort.",
    )
    parser.add_argument(
        "--iga_followup_csv",
        default="followup_data/raw/IgA/IgA_cohort_full_data.csv",
        help="Clinical follow-up data for the IgA cohort.",
    )
    parser.add_argument(
        "--registry_csv",
        default="followup_data/derived/renamed/registry_anonymized.csv",
        help="Registry cohort data.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load cohorts ──────────────────────────────────────────────────────────

    iga_df = load_iga_cohort(args.iga_slides_csv, args.iga_followup_csv)
    iga_df["bag_name"] = make_bag_name(iga_df, args.nest_biopsy)
    iga_df["Biopsy_date"] = pd.to_datetime(iga_df["Biopsy_date"], errors="coerce")
    iga_backup = iga_df.copy()  # full IgA cohort before any filtering

    # Apply date filter (same logic as define_regression_labels.py)
    if args.iga_date_filter.lower() != "none":
        iga_df = iga_df[iga_df["Biopsy_date"] >= args.iga_date_filter]

    # Slides present in iga_backup but removed by the date filter.
    # Saved so they can be explicitly excluded from feature directories
    # before training (e.g. passed to sync_dirs.py or inspected manually).
    iga_excluded = (
        iga_backup.loc[
            ~iga_backup["bag_name"].isin(set(iga_df["bag_name"])),
            ["biopsy_number", "file_name", "Biopsy_date", "Stain"],
        ]
        .drop_duplicates(["biopsy_number", "file_name"])
        .sort_values(["biopsy_number", "file_name"])
        .reset_index(drop=True)
    )

    registry_df = load_registry_cohort(args.registry_csv)
    registry_df["bag_name"] = make_bag_name(registry_df, args.nest_biopsy)

    # Non-IgA registry — pretrain cohort; split assigned below (all train
    # unless --val_non_iga).
    non_iga_df = load_non_iga_cohort(args.registry_csv)
    non_iga_df["bag_name"] = make_bag_name(non_iga_df, args.nest_biopsy)

    # ── Diagnosis code column + vocabulary (opt-in) ───────────────────────────
    # The IgA cohort CSV has no Diagnosis column; those slides are IgA
    # nephropathy by construction, so they share the registry's is_IgA code.
    # Registry/non-IgA slides carry the free-text Diagnosis from the CSV.
    iga_df["diagnosis"] = "IgA nefropati"
    registry_df["diagnosis"] = registry_df["Diagnosis"].apply(normalize_diagnosis)
    non_iga_df["diagnosis"] = non_iga_df["Diagnosis"].apply(normalize_diagnosis)

    code_map = None
    if args.with_diagnosis:
        all_diag = pd.concat(
            [iga_df["diagnosis"], registry_df["diagnosis"], non_iga_df["diagnosis"]]
        )
        code_map, n_codes = build_diagnosis_code_map(all_diag)
        codes_path = os.path.join(args.output_dir, "diagnosis_codes.json")
        with open(codes_path, "w", encoding="utf-8") as f:
            json.dump(
                {"n_codes": n_codes, "codes": code_map}, f, ensure_ascii=False, indent=2
            )

    # ── Assign train / val splits ─────────────────────────────────────────────

    split_kwargs = dict(
        frac=args.val_frac, n_bins=args.n_bins, random_state=args.random_state
    )

    iga_val_patients = (
        select_val_patients(iga_df, "PERSON_NR", "time", **split_kwargs)
        if args.val_source in ("IgA", "both")
        else []
    )
    registry_val_patients = (
        select_val_patients(registry_df, "patient", "time", **split_kwargs)
        if args.val_source in ("registry", "both")
        else []
    )
    # The non-IgA cohort is validated independently of --val_source: it feeds
    # the pretrain phase, not the finetune phase the other two lists validate.
    non_iga_val_patients = (
        select_val_patients(non_iga_df, "patient", "time", **split_kwargs)
        if args.val_non_iga
        else []
    )

    iga_df["split"] = (
        iga_df["PERSON_NR"].isin(iga_val_patients).map({True: "val", False: "train"})
    )
    registry_df["split"] = (
        registry_df["patient"]
        .isin(registry_val_patients)
        .map({True: "val", False: "train"})
    )
    non_iga_df["split"] = (
        non_iga_df["patient"]
        .isin(non_iga_val_patients)
        .map({True: "val", False: "train"})
    )

    # ── Save outputs ──────────────────────────────────────────────────────────

    # Per-slide .npy survival labels ([time, event]) — consumed by
    # MIL.py --labels_paths.  Split membership lives in the CSVs only; the .npy
    # files hold the label regardless of split (MIL.py splits at load time).
    n_iga_npy = write_npy_labels(iga_df, args.iga_output_dir, code_map=code_map)
    n_reg_npy = write_npy_labels(registry_df, args.registry_output_dir, code_map=code_map)
    n_non_iga_npy = write_npy_labels(
        non_iga_df, args.non_iga_output_dir, code_map=code_map
    )

    # IgA full cohort (all clinical columns, useful for downstream analysis)
    iga_df.to_csv(os.path.join(args.output_dir, "full_data.csv"), index=False)

    # Slides excluded by the date filter — use this list to remove them from
    # feature/label directories before training.
    excluded_path = os.path.join(args.output_dir, "iga_excluded_slides.csv")
    iga_excluded.to_csv(excluded_path, index=False)

    # Combined label file (all three cohorts).  biopsy_number and file_name are
    # separate columns so downstream can match either the flat (file_name) or
    # nested (biopsy_number/file_name) WSI layout.
    _label_cols = [
        "biopsy_number", "file_name", "time", "event", "stain", "source", "split",
        "diagnosis",
    ]
    iga_labels = iga_df[
        ["biopsy_number", "file_name", "time", "event", "Stain", "source", "split",
         "diagnosis"]
    ].rename(columns={"Stain": "stain"})
    registry_labels = registry_df[_label_cols]
    non_iga_labels = non_iga_df[_label_cols]
    labels_unfiltered = pd.concat(
        [iga_labels, registry_labels, non_iga_labels], ignore_index=True
    )
    labels_unfiltered.to_csv(
        os.path.join(args.output_dir, "labels_unfiltered.csv"), index=False
    )

    # Validation slide lists (combined + per-source) — see val_split.write_val_csvs.
    # The val lists carry the on-disk bag name (matched directly against
    # discover_bags), so they use bag_name under the "file_name" column header.
    survival_val = write_val_csvs(
        args.val_csv,
        iga_df[iga_df["split"] == "val"][["bag_name"]].rename(
            columns={"bag_name": "file_name"}
        ),
        registry_df[registry_df["split"] == "val"][["bag_name"]].rename(
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

    # ── Summary ───────────────────────────────────────────────────────────────

    n_iga_train = (iga_df["split"] == "train").sum()
    n_iga_val = (iga_df["split"] == "val").sum()
    n_reg_train = (registry_df["split"] == "train").sum()
    n_reg_val = (registry_df["split"] == "val").sum()

    print(
        f"val_source={args.val_source}  val_frac={args.val_frac}"
        f"  n_bins={args.n_bins}  random_state={args.random_state}"
    )
    print(
        f"  IgA      — train: {n_iga_train:>5d}  val: {n_iga_val:>4d}  "
        f"excluded (date filter): {len(iga_excluded):>4d}"
    )
    print(f"  Registry — train: {n_reg_train:>5d}  val: {n_reg_val:>4d}")
    n_non_iga_train = (non_iga_df["split"] == "train").sum()
    n_non_iga_val = (non_iga_df["split"] == "val").sum()
    print(
        f"  non-IgA  — train: {n_non_iga_train:>5d}  val: {n_non_iga_val:>4d}"
        f"  (pretrain cohort)"
    )
    print(f"  Combined val slides: {len(survival_val)}")
    print(f"\nOutputs:")
    print(f"  {args.iga_output_dir}/        ({n_iga_npy} .npy files)")
    print(f"  {args.registry_output_dir}/   ({n_reg_npy} .npy files)")
    print(f"  {args.non_iga_output_dir}/    ({n_non_iga_npy} .npy files)")
    val_stem, val_ext = os.path.splitext(args.val_csv)
    print(f"  {args.val_csv}")
    print(f"  {val_stem}_IgA{val_ext}")
    print(f"  {val_stem}_registry{val_ext}")
    if args.val_non_iga:
        print(f"  {val_stem}_non_IgA{val_ext}  (pass to MIL.py --pretrain_val_csv)")
    print(f"  {os.path.join(args.output_dir, 'labels_unfiltered.csv')}")
    print(f"  {os.path.join(args.output_dir, 'full_data.csv')}")
    print(f"  {excluded_path}  ({len(iga_excluded)} slides excluded by date filter)")
    if code_map is not None:
        print(
            f"  {codes_path}  ({n_codes} diagnosis codes incl. 0=unknown; "
            f"labels written as [time, event, code_idx])"
        )


if __name__ == "__main__":
    main()
