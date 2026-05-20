import pandas as pd

registry = pd.read_csv("followup_data/registry_full_data.csv")

# Create mapping of unique PersNummer to unique IDs
persnummer_to_id = {
    persnummer: idx
    for idx, persnummer in enumerate(registry["patient_fnr"].unique(), start=1)
}

# Add new column with anonymized IDs
registry["AnonymizedID"] = registry["patient_fnr"].map(persnummer_to_id)
registry.drop(columns=["patient_fnr"], inplace=True)
registry.to_csv("followup_data/registry_anonymized.csv", index=False)
