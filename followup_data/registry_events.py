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
    columns={"FNR": "ID_event", "BehandlingStart": "Event_date", "Behandling": "Event"},
    inplace=True,
)
events = events[["ID_event", "Event_date", "Event"]]
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
# ensure dates are datetime for correct sorting and time delta calculation
for col in ["Event_date", "Biopsy_date"]:
    data[col] = pd.to_datetime(data[col], errors="coerce")

# compute time to event and sort by Event_date
data["time_to_event"] = data["Event_date"] - data["Biopsy_date"]
# sort by Event_date and keep the first occurrence per ID_diagnosis
data.sort_values(by="Event_date", inplace=True)
data.drop_duplicates(subset=["ID_diagnosis"], keep="first", inplace=True)

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
