import pandas as pd
import matplotlib.pyplot as plt

events = pd.read_excel("followup_data/registry_events.xlsx")
death = pd.read_excel("followup_data/registry_death.xlsx")
diagnosis = pd.read_excel("followup_data/registry_full_data.xlsx")

death.rename(
    columns={
        "FNR": "ID_death",
        "Morsdate": "Death_date",
        "MorsReasonEDTA": "Death_code",
    },
    inplace=True,
)
death = death[["ID_death", "Death_date", "Death_code"]]
events.rename(
    columns={
        "FNR": "ID_event",
        "BehandlingStart": "Treatment_date",
        "Behandling": "Treatment",
    },
    inplace=True,
)
events = events[["ID_event", "Treatment_date", "Treatment"]]
diagnosis.rename(
    columns={
        "PersNummer": "ID_diagnosis",
        "Biopsidato": "Biopsy_date",
        "Diagnoser P1.2013_konklusiv_diagnose": "Diagnosis",
    },
    inplace=True,
)
diagnosis = diagnosis[["ID_diagnosis", "Biopsy_date", "Diagnosis"]]

diagnosis["is_IgA"] = diagnosis["Diagnosis"].str.contains(
    "IgA nefropati", case=False, na=False
)

data = pd.merge(
    diagnosis, events, left_on="ID_diagnosis", right_on="ID_event", how="right"
)

data = pd.merge(data, death, left_on="ID_event", right_on="ID_death", how="left")
data.drop(columns=["ID_event", "ID_death"], inplace=True)
data["Event"] = data["Treatment"]
data["Event_date"] = data["Treatment_date"]
# ensure dates are datetime for correct sorting and time delta calculation
for col in ["Treatment_date", "Biopsy_date", "Death_date"]:
    data[col] = pd.to_datetime(data[col], errors="coerce")


def define_event(row):
    if pd.notna(row["Death_date"]) and pd.notna(row["Treatment_date"]):
        if row["Death_date"] < row["Treatment_date"]:
            row["Event_date"] = row["Death_date"]
            row["Event"] = "Death"
            return row
    elif pd.notna(row["Death_date"]):
        row["Event_date"] = row["Death_date"]
        row["Event"] = "Death"
        return row
    return row


data = data.apply(define_event, axis=1)

# compute time to event and sort by Event_date
data["time_to_event"] = data["Event_date"] - data["Biopsy_date"]
# sort by Event_date and keep the first occurrence per ID_diagnosis
data.sort_values(by="Event_date", inplace=True)
data.drop_duplicates(subset=["ID_diagnosis"], keep="first", inplace=True)
data = data[data["time_to_event"].dt.days > 60]

# plot histogram for IgA cases and save it
plt.figure(figsize=(8, 6))
plt.hist(
    data.loc[data["is_IgA"], "time_to_event"].dt.days.dropna(),
    bins=20,
    color="#4C72B0",
    edgecolor="black",
)
plt.xlabel("Time to event (days)")
plt.ylabel("Count")
plt.title("Distribution of Time to Event for IgA Patients")
plt.tight_layout()
plt.savefig("followup_data/times_to_event.png")
plt.close()

data.to_csv("followup_data/registry_currated.csv", index=False)

wsi_registry_NKBR = pd.read_excel(
    "followup_data/NKBR_with_filenames_new_stain_all_found_fix.xlsx"
)
wsi_registry_KB = pd.read_excel(
    "followup_data/KB_with_filenames_new_stain_all_found_fix.xlsx"
)
wsi_registry = pd.concat(
    [
        wsi_registry_NKBR[["patient_fnr", "biop_number", "ANON_name", "Stain"]],
        wsi_registry_KB[["patient_fnr", "biop_number", "ANON_name", "Stain"]],
    ],
    ignore_index=True,
)
data_registry = pd.merge(
    data, wsi_registry, left_on="ID_diagnosis", right_on="patient_fnr", how="inner"
)
data_registry.to_csv("followup_data/registry_full_data.csv", index=False)
