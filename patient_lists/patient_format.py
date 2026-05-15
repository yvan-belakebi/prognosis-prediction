import pandas as pd
from itables import init_notebook_mode

init_notebook_mode(connected=True)

df = pd.read_csv("patient_list.csv", dtype={"PersNummer": str})
df["PersNummer"] = df["PersNummer"].astype(str).str.zfill(11)
df.to_csv("patient_list_with_leading_zeros.csv", index=False, encoding="utf-8")
