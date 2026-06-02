import pandas as pd

df = pd.read_csv("labels_combined.csv", encoding="utf-8")
df = df[df["time"] >= 730]
df = df[df["stain"] == "PAS"]
df[["file_name", "time", "stain", "event", "source"]].to_csv(
    "labels.csv", index=False, encoding="utf-8"
)
