import pandas as pd
from itables import init_notebook_mode, show

init_notebook_mode(all_interactive=True)
# patient_df = pd.read_csv("followup_data/IgA_cohort_full_data.csv", encoding="utf-8")
patient_df = pd.read_csv(
    "../../followup_data/IgA_cohort_full_data.csv", encoding="utf-8"
)
patient_df.rename(columns={"Mors": "Death", "Mors_year": "Death_year"}, inplace=True)

event_cols = ["ESKD", "Death", "Transplant", "Dialysis"]
patient_df["ESKD"] = patient_df["ESKD"].map({"No": 0, "Yes": 1})
patient_df["Death"] = patient_df["Death"].map({"Nei": 0, "Ja": 1})
patient_df["Dialysis"] = patient_df["Dialysis"].map({"No": 0, "Yes": 1})
patient_df["Transplant"] = patient_df["Transplant"].map({"No": 0, "Yes": 1})


def print_event_occurences(df):
    print(df.value_counts(subset=event_cols))
    print(df["RRT_or_death"].value_counts())
    print(df["ESKD"].value_counts())
    print(df["Death"].value_counts())
    print(df["Transplant"].value_counts())
    print(df["Dialysis"].value_counts())
    try:
        print(df["First_event_RRT_death"].value_counts())
    except KeyError:
        pass


print_event_occurences(patient_df)


def simulate_from_events_distribution(df, event_cols, n_samples=1000):
    event_counts = df.value_counts(subset=event_cols)
    probabilities = event_counts / event_counts.sum()
    probabilities_df = probabilities.reset_index()
    simulated_events = probabilities_df.sample(
        n=n_samples, weights=probabilities_df.iloc[:, -1], replace=True
    )
    return simulated_events


def infer_RRT_or_death(row):
    return 1 - (1 - row["Death"]) * (1 - row["Transplant"]) * (1 - row["Dialysis"])


simulated_df = simulate_from_events_distribution(patient_df, event_cols)
simulated_df["RRT_or_death"] = simulated_df.apply(infer_RRT_or_death, axis=1)
# print_event_occurences(simulated_df)
