import pandas as pd
from pathlib import Path

raw = sorted(Path("data/raw").glob("*.csv"))[0]
df = pd.read_csv(raw)

df = df[df["language"]=="en"].copy()
df["type"] = df["type"].astype(str).str.strip()
df["queue"] = df["queue"].astype(str).str.strip()
df["priority"] = df["priority"].astype(str).str.strip().str.lower()

print("TYPE:\n", df["type"].value_counts(), "\n")
print("QUEUE:\n", df["queue"].value_counts(), "\n")
print("PRIORITY:\n", df["priority"].value_counts(), "\n")