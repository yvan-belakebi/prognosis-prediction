import pandas as pd

df = pd.read_excel(
    "followup_data/raw/registry/KB_with_filenames_new_stain_all_found_fix.xlsx"
)
df = df[df["Stain"] == "Sirius red"]
df = df.loc[
    ~df["Department of Pathology"].isin(["Haukeland Universitetssjukehus"]),
    ["ANON_name"],
]

df_bis = pd.read_excel(
    "followup_data/raw/registry/NKBR_with_filenames_new_stain_all_found_fix.xlsx"
)
df_bis = df_bis[df_bis["Stain"] == "Sirius red"]
df_bis = df_bis.loc[
    ~df_bis["Department of Pathology"].isin(["Haukeland Universitetssjukehus"]),
    ["ANON_name"],
]

full_df = pd.concat([df, df_bis], axis=0)
full_df.to_csv(
    "followup_data/derived/registry/sirius_red_other_hospitals.csv", index=False
)
