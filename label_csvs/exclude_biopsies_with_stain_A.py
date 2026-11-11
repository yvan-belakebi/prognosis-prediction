import pandas as pd

EXCLUDED = {"Sirius red", "C1q", "C3", "C5-9", "IgA", "IgG", "IgM", "Kappa", "Lambda"}

df = pd.read_csv(
    "label_csvs/labels.csv", dtype={"file_name": str, "biopsy_number": str}
)
bad_biopsies = df.loc[df["stain"].isin(EXCLUDED), "biopsy_number"].unique()
df.loc[~df["biopsy_number"].isin(bad_biopsies), ["file_name"]].to_csv(
    "label_csvs/slides_without_excluded_stains.csv", index=False
)
