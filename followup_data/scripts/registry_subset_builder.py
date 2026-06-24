# %% select long enough time to event and PAS
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

df = pd.read_csv("followup_data/derived/renamed/registry_anonymized.csv", encoding="utf-8")
df = df[df["Stain"] == "PAS"]
df["time_to_event"] = pd.to_timedelta(df["time_to_event"], errors="coerce")
df = df[df["time_to_event"] >= dt.timedelta(days=730)]
df.drop_duplicates(subset=["biop_number"], inplace=True)
df.dropna(subset=["Diagnosis"], inplace=True)

igA_df = df[df["is_IgA"]]
non_igA_df = df[~df["is_IgA"]].sample(n=1000, random_state=0)


def stratified_sample_n(df, column, n):
    return df.groupby(column, group_keys=False).apply(
        lambda x: x.sample(max(1, round(n * len(x) / len(df))))
    )


non_IgA_sub_df = stratified_sample_n(non_igA_df, "Diagnosis", n=1000)

df = pd.concat([igA_df, non_IgA_sub_df], ignore_index=True)

df.to_csv(
    "followup_data/derived/subsets/registry_anonymized_long_TTE_PAS.csv",
    index=False,
    encoding="utf-8",
)
df[df["is_IgA"]]["ANON_name"].to_csv(
    "followup_data/derived/subsets/registry_IgA_long_TTE_PAS_file_names.csv",
    index=False,
    header=False,
    encoding="utf-8",
)
df[~df["is_IgA"]]["ANON_name"].to_csv(
    "followup_data/derived/subsets/registry_non_IgA_long_TTE_PAS_file_names.csv",
    index=False,
    header=False,
    encoding="utf-8",
)
# %%
plt.hist(df[df["is_IgA"]]["time_to_event"].dt.days.dropna(), bins=30)
plt.xlabel("Time to event (days)")
plt.ylabel("Count")
plt.title("Distribution of time_to_event")
plt.tight_layout()
plt.show()

print(len(df[df["is_IgA"]]))
