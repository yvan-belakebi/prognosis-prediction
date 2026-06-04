"""
define_labels.py — Build survival labels and the train/val split for MIL.py.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_labels.py [options]

Key outputs
-----------
  <output_dir>/labels_combined.csv          All slides (IgA + registry + non-IgA) with
                                            time, event, stain, source, split.
                                            non-IgA slides always have split="train".
  <val_csv>                                 Flat list of validation slide names
                                            (file_name column); consumed directly
                                            by MIL.py --val_csv.
  <output_dir>/full_data.csv               IgA cohort with all clinical columns.

Parameters
----------
--val_source    IgA | registry | both  (default: both)
                Which cohort(s) contribute slides to the validation set.
--val_frac      Fraction of patients assigned to validation  (default: 0.2)
--n_bins        Quantile strata for time-stratified patient sampling  (default: 4)
--random_state  Random seed  (default: 42)
--val_csv       Full path of the combined validation list for MIL.py --val_csv
                (default: followup_data/survival_validation_files.csv)
--output_dir    Directory for all other CSV outputs  (default: followup_data)
--iga_slides_csv
--iga_followup_csv
--registry_csv  Input file paths (rarely need changing).
"""

import argparse
import math
import os
import re

import pandas as pd

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


def select_val_patients(df, patient_col, time_col, frac, n_bins, random_state):
    """Return patient IDs for the validation set, stratified by time quantile.

    Patients are binned into n_bins equal-frequency strata on time, then
    ceil(frac) of patients is sampled from each stratum.  Stratification
    preserves the event-time distribution across splits and guarantees every
    slide from a given patient lands in the same set.
    """
    patient_df = df.groupby(patient_col)[time_col].first().reset_index()
    patient_df["stratum"] = pd.qcut(
        patient_df[time_col], q=n_bins, labels=False, duplicates="drop"
    )

    def _sample(g):
        n = max(1, math.ceil(frac * len(g)))
        return g.sample(n=n, random_state=random_state)

    return (patient_df.groupby("stratum", group_keys=False).apply(_sample))[
        patient_col
    ].tolist()


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
    df["slide_stem"]    = df["File Location"].apply(extract_file_name)
    df["biopsy_dirname"] = df["Biopsy Number"].apply(biopsy_to_dirname)
    df["file_name"]     = df["biopsy_dirname"] + "/" + df["slide_stem"]
    df["source"] = "IgA"
    return df


def _parse_registry_time_event(df):
    """Add time (days), event, file_name, and biopsy_dirname columns to registry data."""
    df["time"] = (
        df["time_to_event"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    )
    df["event"] = df["Event"].notna().astype(int)
    df["slide_stem"]    = df["ANON_name"].astype(str)
    df["biopsy_dirname"] = df["biop_number"].astype(str).apply(transform_label).apply(biopsy_to_dirname)
    df["file_name"]     = df["biopsy_dirname"] + "/" + df["slide_stem"]
    df.rename(
        columns={"ID_diagnosis": "patient", "Stain": "stain"},
        inplace=True,
    )
    return df


def load_registry_cohort(registry_csv):
    """Return the IgA registry cohort (is_IgA == True) with time, event, file_name."""
    df = pd.read_csv(registry_csv)
    df = _parse_registry_time_event(df[df["is_IgA"] == True].copy())
    df["source"] = "registry"
    return df


def load_non_iga_cohort(registry_csv):
    """Return the non-IgA registry cohort (is_IgA == False) with time, event, file_name.

    Non-IgA slides are always assigned to the training set; they are never
    included in the validation split.
    """
    df = pd.read_csv(registry_csv)
    df = _parse_registry_time_event(df[df["is_IgA"] == False].copy())
    df["source"] = "non_IgA"
    return df


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build survival labels and train/val split CSVs for MIL.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    # Output paths
    parser.add_argument(
        "--val_csv",
        default="followup_data/survival_validation_files.csv",
        help=(
            "Output path for the combined validation slide list consumed by "
            "MIL.py --val_csv."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="followup_data",
        help="Directory for all other CSV outputs.",
    )

    # Input paths
    parser.add_argument(
        "--iga_slides_csv",
        default="followup_data/IgA_slide_data.csv",
        help="Slide metadata for the IgA cohort.",
    )
    parser.add_argument(
        "--iga_followup_csv",
        default="followup_data/IgA_cohort_full_data.csv",
        help="Clinical follow-up data for the IgA cohort.",
    )
    parser.add_argument(
        "--registry_csv",
        default="followup_data/registry_anonymized.csv",
        help="Registry cohort data.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load cohorts ──────────────────────────────────────────────────────────

    iga_df = load_iga_cohort(args.iga_slides_csv, args.iga_followup_csv)
    iga_df["Biopsy_date"] = pd.to_datetime(iga_df["Biopsy_date"], errors="coerce")
    iga_backup = iga_df.copy()  # full IgA cohort before any filtering
    iga_df = iga_df[
        iga_df["Biopsy_date"] >= "2006-01-01"
    ]  # IgA biopsies before 2006 have very short follow-up

    # Slides present in iga_backup but removed by the date filter.
    # Saved so they can be explicitly excluded from feature directories
    # before training (e.g. passed to sync_dirs.py or inspected manually).
    iga_excluded = iga_backup.loc[
        ~iga_backup["file_name"].isin(set(iga_df["file_name"])),
        ["file_name", "Biopsy_date", "Stain"],
    ].drop_duplicates("file_name").sort_values("file_name").reset_index(drop=True)

    registry_df = load_registry_cohort(args.registry_csv)

    # Non-IgA registry — always train, never validated
    non_iga_df = load_non_iga_cohort(args.registry_csv)
    non_iga_df["split"] = "train"

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

    iga_df["split"] = (
        iga_df["PERSON_NR"].isin(iga_val_patients).map({True: "val", False: "train"})
    )
    registry_df["split"] = (
        registry_df["patient"]
        .isin(registry_val_patients)
        .map({True: "val", False: "train"})
    )

    # ── Save outputs ──────────────────────────────────────────────────────────

    # IgA full cohort (all clinical columns, useful for downstream analysis)
    iga_df.to_csv(os.path.join(args.output_dir, "full_data.csv"), index=False)

    # Slides excluded by the date filter — use this list to remove them from
    # feature/label directories before training.
    excluded_path = os.path.join(args.output_dir, "iga_excluded_slides.csv")
    iga_excluded.to_csv(excluded_path, index=False)

    # Combined label file (all three cohorts)
    iga_labels = iga_df[
        ["file_name", "time", "event", "Stain", "source", "split"]
    ].rename(columns={"Stain": "stain"})
    registry_labels = registry_df[
        ["file_name", "time", "event", "stain", "source", "split"]
    ]
    non_iga_labels = non_iga_df[
        ["file_name", "time", "event", "stain", "source", "split"]
    ]
    labels_combined = pd.concat(
        [iga_labels, registry_labels, non_iga_labels], ignore_index=True
    )
    labels_combined.to_csv(
        os.path.join(args.output_dir, "labels_combined.csv"), index=False
    )

    # Per-source validation files (stem of --val_csv + _IgA / _registry)
    val_stem, val_ext = os.path.splitext(args.val_csv)
    iga_df[iga_df["split"] == "val"][["file_name"]].to_csv(
        f"{val_stem}_IgA{val_ext}", index=False, header=False
    )
    registry_df[registry_df["split"] == "val"][["file_name"]].to_csv(
        f"{val_stem}_registry{val_ext}", index=False, header=False
    )

    # Combined validation list — consumed by MIL.py --val_csv
    survival_val = pd.concat(
        [
            iga_df[iga_df["split"] == "val"][["file_name"]],
            registry_df[registry_df["split"] == "val"][["file_name"]],
        ],
        ignore_index=True,
    )
    survival_val.to_csv(args.val_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────

    n_iga_train = (iga_df["split"] == "train").sum()
    n_iga_val = (iga_df["split"] == "val").sum()
    n_reg_train = (registry_df["split"] == "train").sum()
    n_reg_val = (registry_df["split"] == "val").sum()

    print(
        f"val_source={args.val_source}  val_frac={args.val_frac}"
        f"  n_bins={args.n_bins}  random_state={args.random_state}"
    )
    print(f"  IgA      — train: {n_iga_train:>5d}  val: {n_iga_val:>4d}  "
          f"excluded (pre-2006): {len(iga_excluded):>4d}")
    print(f"  Registry — train: {n_reg_train:>5d}  val: {n_reg_val:>4d}")
    print(f"  non-IgA  — train: {len(non_iga_df):>5d}  val:    0  (always train)")
    print(f"  Combined val slides: {len(survival_val)}")
    print(f"\nOutputs:")
    print(f"  {args.val_csv}")
    print(f"  {val_stem}_IgA{val_ext}")
    print(f"  {val_stem}_registry{val_ext}")
    print(f"  {os.path.join(args.output_dir, 'labels_combined.csv')}")
    print(f"  {os.path.join(args.output_dir, 'full_data.csv')}")
    print(f"  {excluded_path}  ({len(iga_excluded)} slides excluded by date filter)")


if __name__ == "__main__":
    main()
