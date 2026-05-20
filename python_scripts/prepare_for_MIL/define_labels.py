import math
import os
import re

import pandas as pd
from itables import init_notebook_mode

init_notebook_mode(connected=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def transform_label(label):
    if pd.isna(label):
        return label
    label = str(label).strip().replace("\xa0", " ")
    match = re.match(r"^B(\d{2})\s+(\d+)$", label)
    if match:
        return f"{match.group(2).lstrip('0')}/{match.group(1)}"
    match = re.match(r"^B(\d{2})(\d+)$", label)
    if match:
        return f"{match.group(2).lstrip('0')}/{match.group(1)}"
    return label


def prepare_slides(df):
    copy_df = df.copy()
    copy_df["Biopsy Number"] = copy_df["Biopsy Number"].astype(str)
    copy_df["Biopsy_number_transformed"] = copy_df["Biopsy Number"].apply(
        transform_label
    )
    return copy_df


def extract_file_name(file_location):
    if pd.isna(file_location):
        return file_location
    path = str(file_location).strip().replace("\xa0", " ")
    basename = re.split(r"[\\\\/]", path)[-1]
    basename = basename.split("?")[0].split("#")[0].strip()
    return os.path.splitext(basename)[0]


def select_validation_patients(df, patient_col, time_col, frac=0.2, n_bins=4, random_state=42):
    """Return the list of patient IDs assigned to the validation set.

    Patients are binned into n_bins quantile strata by time, then ceil(frac)
    of patients is sampled from each stratum, preserving the time-to-event
    distribution.  Selecting at the patient level guarantees that every slide
    from a given patient lands in the same set.
    """
    patient_df = df.groupby(patient_col)[time_col].first().reset_index()
    patient_df["stratum"] = pd.qcut(
        patient_df[time_col], q=n_bins, labels=False, duplicates="drop"
    )

    def sample_stratum(g):
        n = max(1, math.ceil(frac * len(g)))
        return g.sample(n=n, random_state=random_state)

    return (
        patient_df.groupby("stratum", group_keys=False).apply(sample_stratum)
    )[patient_col].tolist()


# ── IgA cohort ────────────────────────────────────────────────────────────────

IgA_slides = prepare_slides(pd.read_csv("followup_data/IgA_slide_data.csv"))
IgA_slides.to_csv("followup_data/IgA_slide_paths.csv", index=False)

IgA_slides = pd.read_csv("followup_data/IgA_slide_paths.csv")[
    ["Biopsy_number_transformed", "File Location", "Slide ID", "Stain"]
]
IgA_slides.rename(columns={"Biopsy_number_transformed": "Biopsy Number"}, inplace=True)

IgA_followup = pd.read_csv("followup_data/IgA_cohort_full_data.csv")
IgA_followup.rename(columns={"Biopsy_nr": "Biopsy Number"}, inplace=True)

iga_df = pd.merge(IgA_slides, IgA_followup, on="Biopsy Number", how="inner")


def calculate_length_follow_up(row):
    if row["RRT_or_death"] == "Yes":
        if (
            pd.notna(row["Year_RRT_or_death"])
            and pd.notna(row["ESKD_year"])
            and pd.notna(row["Biopsy_year"])
        ):
            return min(row["Year_RRT_or_death"], row["ESKD_year"]) - row["Biopsy_year"]
        elif pd.notna(row["Year_RRT_or_death"]) and pd.notna(row["Biopsy_year"]):
            return row["Year_RRT_or_death"] - row["Biopsy_year"]
        elif pd.notna(row["ESKD_year"]) and pd.notna(row["Biopsy_year"]):
            return row["ESKD_year"] - row["Biopsy_year"]
        else:
            return None
    else:
        return row["Length_follow_up"]


iga_df["time_years"] = iga_df.apply(calculate_length_follow_up, axis=1)
iga_df["time"] = iga_df["time_years"] * 365.25  # convert to days
iga_df["event"] = iga_df["RRT_or_death"].apply(lambda x: 1 if x == "Yes" else 0)
iga_df["file_name"] = iga_df["File Location"].apply(extract_file_name)
iga_df["source"] = "IgA"

# legacy outputs (IgA-only, unchanged format)
iga_df.to_csv("followup_data/full_data.csv", index=False)
iga_df[["file_name", "time", "Stain", "event"]].to_csv(
    "followup_data/labels_new.csv", index=False
)


# ── Registry ──────────────────────────────────────────────────────────────────

registry_df = pd.read_csv("followup_data/registry_anonymized.csv")
registry_df = registry_df[registry_df["is_IgA"] == True]

# "62 days" → 62.0
registry_df["time"] = (
    registry_df["time_to_event"]
    .astype(str)
    .str.extract(r"(\d+(?:\.\d+)?)")
    .astype(float)
    .squeeze()
)

# event=1 for any death, treatment or ESKD; event=0 for censored patients
registry_df["event"] = registry_df["Event"].notna().astype(int)
registry_df.rename(
    columns={"ANON_name": "file_name", "ID_diagnosis": "patient", "Stain": "stain"},
    inplace=True,
)
registry_df["source"] = "registry"


# ── Stratified validation split ───────────────────────────────────────────────
# Patients are selected first; every slide belonging to a patient inherits
# the same split, so no patient is split across train and validation.

iga_val_patients = select_validation_patients(iga_df, patient_col="PERSON_NR", time_col="time")
iga_df["split"] = iga_df["PERSON_NR"].isin(iga_val_patients).map({True: "val", False: "train"})

registry_val_patients = select_validation_patients(registry_df, patient_col="patient", time_col="time")
registry_df["split"] = registry_df["patient"].isin(registry_val_patients).map({True: "val", False: "train"})

iga_df[iga_df["split"] == "val"][["file_name"]].to_csv(
    "followup_data/validation_files_IgA.csv", index=False, header=False
)
registry_df[registry_df["split"] == "val"][["file_name"]].to_csv(
    "followup_data/validation_files_registry.csv", index=False, header=False
)


# ── Combined labels ───────────────────────────────────────────────────────────

iga_labels = iga_df[["file_name", "time", "event", "Stain", "source", "split"]].rename(
    columns={"Stain": "stain"}
)
registry_labels = registry_df[["file_name", "time", "event", "stain", "source", "split"]]

labels_combined = pd.concat([iga_labels, registry_labels], ignore_index=True)
labels_combined.to_csv("followup_data/labels_combined.csv", index=False)


# legacy combined validation (IgA-only, unchanged)
prep_val_df = iga_df[["PERSON_NR", "file_name"]].rename(
    columns={"PERSON_NR": "patient"}
)
unique_patients = prep_val_df["patient"].unique()
num_val = int(0.2 * len(unique_patients))
val_patients = pd.Series(unique_patients).sample(n=num_val, random_state=42).tolist()
prep_val_df[prep_val_df["patient"].isin(val_patients)][["file_name"]].to_csv(
    "followup_data/validation_files_new.csv", index=False, header=False
)
