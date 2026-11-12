import pandas as pd

INCLUDED = {"Sirius red", "C1q", "C3", "C5-9", "IgA", "IgG", "IgM", "Kappa", "Lambda"}

df = pd.read_csv(
    "label_csvs/labels.csv", dtype={"file_name": str, "biopsy_number": str}
)
good_biopsies = df.loc[df["stain"].isin(INCLUDED), "biopsy_number"].unique()
df.loc[df["biopsy_number"].isin(good_biopsies), ["file_name"]].to_csv(
    "label_csvs/slides_with_selected_stains.csv", index=False
)
