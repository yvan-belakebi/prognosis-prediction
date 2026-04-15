import pandas as pd
import numpy as np

patient_df = pd.read_csv("patient_list.csv", encoding="utf-8")
KB = pd.read_excel("KB.xlsx")
NKBR = pd.read_excel("NKBR.xlsx")
concatenated_df = pd.concat([KB, NKBR], ignore_index=True)
concatenated_df = concatenated_df[["patient_id", "patient_fnr"]]

patient_df = patient_df.rename(columns={"PasientID": "patient_id"})
patient_df["patient_id"] = patient_df["patient_id"].apply(
    lambda x: "%05d" % int(x) if pd.notna(x) else x
)
patient_df = patient_df.merge(
    concatenated_df, left_on="PersNummer", right_on="patient_fnr", how="inner"
)
patient_df = patient_df[["PersNummer", "EtterNavn", "ForNavn"]]
patient_df = patient_df.drop_duplicates()
patient_df["PersNummer"] = (
    patient_df["PersNummer"].astype(float).apply(lambda x: "{:.0f}".format(x))
)
patient_df.to_csv("filtered_patient_list.csv", index=False)
