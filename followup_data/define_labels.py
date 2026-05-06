import pandas as pd
import re
from itables import init_notebook_mode

init_notebook_mode(connected=True)

# Unify the biopsy number format in the slide paths and the follow-up data.

IgA_slides = pd.read_csv("IgA_slide_data.csv")
IgA_followup = pd.read_csv("IgA_cohort_full_data.csv")

test_str = "B12 34"
test_str2 = "B1234"


def transform_label(label):
    if pd.isna(label):
        return label

    label = str(label).strip()  # remove leading/trailing spaces
    label = label.replace("\xa0", " ")  # normalize non-breaking spaces

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


# print(transform_label(test_str))
# print(transform_label(test_str2))

IgA_slides = prepare_slides(IgA_slides)
IgA_slides.to_csv("IgA_slide_paths.csv", index=False)


IgA_slides = pd.read_csv("IgA_slide_paths.csv")
IgA_slides = IgA_slides[
    ["Biopsy_number_transformed", "File Location", "Slide ID", "Stain"]
]
IgA_slides.rename(columns={"Biopsy_number_transformed": "Biopsy Number"}, inplace=True)

IgA_followup = pd.read_csv("IgA_cohort_full_data.csv")
IgA_followup.rename(columns={"Biopsy_nr": "Biopsy Number"}, inplace=True)

df = pd.merge(IgA_slides, IgA_followup, on="Biopsy Number", how="inner")


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


df["time"] = df.apply(lambda row: calculate_length_follow_up(row), axis=1)
interesting_cols = ["ESKD_year", "Year_RRT_or_death", "Length_follow_up", "Biopsy_year"]
df.to_csv("full_data.csv", index=False)
df["RRT_or_death"] = df["RRT_or_death"].apply(lambda x: 1 if x == "Yes" else 0)

df[["File Location", "time", "Stain", "RRT_or_death"]].to_csv("labels.csv", index=False)
