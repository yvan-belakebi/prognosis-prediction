"""
define_classification_labels.py — Create per-slide .npy label files for CKD severity
classification, ready for use with classification_MIL.py.

Source column: CKD_stage_diagnosis in followup_data/IgA_cohort_full_data.csv

CKD stage → integer mapping (0-indexed):
    G1  → 0
    G2  → 1
    G3a → 2
    G3b → 3
    G4  → 4
    G5  → 5

Outputs:
    WSI/IgA/labels_classification/      one .npy per slide (all splits)
    WSI/IgA/labels_classification_val/  .npy files for validation slides only
    followup_data/labels_classification.csv  summary table with split column

Validation split: 20 % of patients, stratified by CKD stage so that class
proportions are preserved. All slides from the same patient stay in the same
split.

Run from the project root:
    python python_scripts/prepare_for_MIL/define_classification_labels.py
"""

import math
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CKD stage → integer mapping
# ---------------------------------------------------------------------------

CKD_STAGE_MAP = {"G1": 0, "G2": 1, "G3a": 2, "G3b": 3, "G4": 4, "G5": 5}

# ---------------------------------------------------------------------------
# Helpers (same biopsy-number normalisation as define_labels.py)
# ---------------------------------------------------------------------------


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


def select_validation_patients(df, patient_col, class_col, frac=0.2, random_state=42):
    """Return patient IDs assigned to the validation set.

    One patient ID is sampled per CKD-stage stratum such that each class
    contributes ceil(frac) of its patients to validation. All slides from a
    given patient land in the same split.
    """
    patient_df = df.groupby(patient_col)[class_col].first().reset_index()

    def sample_stratum(g):
        n = max(1, math.ceil(frac * len(g)))
        return g.sample(n=min(n, len(g)), random_state=random_state)

    return (patient_df.groupby(class_col, group_keys=False).apply(sample_stratum))[
        patient_col
    ].tolist()


# ---------------------------------------------------------------------------
# Load and merge slide paths with follow-up data
# ---------------------------------------------------------------------------

IgA_slides = prepare_slides(pd.read_csv("followup_data/IgA_slide_data.csv"))
IgA_slides = IgA_slides[
    ["Biopsy_number_transformed", "File Location", "Slide ID", "Stain"]
]
IgA_slides.rename(columns={"Biopsy_number_transformed": "Biopsy Number"}, inplace=True)

IgA_followup = pd.read_csv("followup_data/IgA_cohort_full_data.csv")
IgA_followup.rename(columns={"Biopsy_nr": "Biopsy Number"}, inplace=True)

df = pd.merge(IgA_slides, IgA_followup, on="Biopsy Number", how="inner")
df["file_name"] = df["File Location"].apply(extract_file_name)

# ---------------------------------------------------------------------------
# Map CKD stages to integers
# ---------------------------------------------------------------------------

df["ckd_label"] = df["CKD_stage_diagnosis"].map(CKD_STAGE_MAP)

n_missing = df["ckd_label"].isna().sum()
if n_missing > 0:
    unknown = df.loc[df["ckd_label"].isna(), "CKD_stage_diagnosis"].unique().tolist()
    print(
        f"Warning: {n_missing} slides have an unrecognised CKD stage {unknown} and will be skipped."
    )
    df = df.dropna(subset=["ckd_label"])

df["ckd_label"] = df["ckd_label"].astype(int)

# ---------------------------------------------------------------------------
# Patient-level validation split, stratified by CKD stage
# ---------------------------------------------------------------------------

val_patients = select_validation_patients(
    df, patient_col="PERSON_NR", class_col="ckd_label"
)
df["split"] = df["PERSON_NR"].isin(val_patients).map({True: "val", False: "train"})

# ---------------------------------------------------------------------------
# Save CSV summary and validation list
# ---------------------------------------------------------------------------

summary_path = "followup_data/labels_classification.csv"
df[
    ["file_name", "ckd_label", "CKD_stage_diagnosis", "Stain", "PERSON_NR", "split"]
].to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")

val_csv_path = "followup_data/classification_validation_files.csv"
df[df["split"] == "val"][["file_name"]].to_csv(val_csv_path, index=False)
print(f"Validation list saved to {val_csv_path}")

# ---------------------------------------------------------------------------
# Save per-slide .npy files
# ---------------------------------------------------------------------------

TRAIN_DIR = "WSI/IgA/labels_classification"
VAL_DIR = "WSI/IgA/labels_classification_val"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for _, row in df.iterrows():
    label_array = np.array(int(row["ckd_label"]))
    np.save(os.path.join(TRAIN_DIR, f"{row['file_name']}.npy"), label_array)
    if row["split"] == "val":
        np.save(os.path.join(VAL_DIR, f"{row['file_name']}.npy"), label_array)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

stage_order = sorted(CKD_STAGE_MAP, key=CKD_STAGE_MAP.get)
counts = (
    df.groupby(["CKD_stage_diagnosis", "split"])
    .size()
    .unstack(fill_value=0)
    .reindex(stage_order)
)
print(f"\nSlide counts by CKD stage and split:\n{counts}")
print(f"\nTrain slides : {(df['split'] == 'train').sum()}")
print(f"Val slides   : {(df['split'] == 'val').sum()}")
print(f"\nLabel files  → {TRAIN_DIR}/")
print(f"Val labels   → {VAL_DIR}/")
print(
    "\nUsage with classification_MIL.py:\n"
    "  python python_scripts/MIL/classification_MIL.py --model_type abmil \\\n"
    "      --features_paths WSI/IgA/UNI2-h_feats \\\n"
    "      --labels_paths   WSI/IgA/labels_classification \\\n"
    "      --val_features_paths WSI/IgA/UNI2-h_feats_val \\\n"
    "      --val_labels_paths   WSI/IgA/labels_classification_val \\\n"
    "      --n_classes 6"
)
