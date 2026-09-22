import pandas as pd

df = pd.read_excel(
    f"\\\\ihelse.net\\Forskning\\hbe\\2023-517496\\Database transfer\\Yvan\\To be relocated to nøkkelområdet\\KB_with_filenames_new_stain_all_found_fix.xlsx"
)
df_bis = pd.read_excel(
    f"\\\\ihelse.net\\Forskning\\hbe\\2023-517496\\Database transfer\\Yvan\\To be relocated to nøkkelområdet\\NKBR_with_filenames_new_stain_all_found_fix.xlsx"
)
df_forbidden = pd.read_csv(
    f"\\\\ihelse.net\\Forskning\\hbe\\2023-517496\\Database transfer\\Yvan\\To be relocated to nøkkelområdet\\output_final.csv"
)

df = df[["ANON_name", "Stain", "Department of Pathology"]]
df_bis = df_bis[["ANON_name", "Stain", "Department of Pathology"]]
df_forbidden = df_forbidden[["Name"]].rename(columns={"Name": "ANON_name"})

df_tot = pd.concat([df, df_bis], ignore_index=True)
df_tot = df_tot[~df_tot["ANON_name"].isin(df_forbidden["ANON_name"])]

df_tot.to_csv("registry_slides_untiled.csv", index=False)
