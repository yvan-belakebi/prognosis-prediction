import pandas as pd

full = pd.read_csv("IgA_cohort_full_data.csv", encoding="utf-8")
full["Biopsy_date"] = pd.to_datetime(full["Biopsy_date"], errors="coerce")
full[full["Biopsy_date"] >= "2006-01-01"][["Biopsy_nr", "Biopsy_date"]].to_csv(
    "IgA_cohort_biopsies_after_2006.csv", index=False
)
