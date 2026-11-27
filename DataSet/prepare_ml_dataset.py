import pandas as pd
from pathlib import Path

# Locate the cleaned dataset
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DataSet" / "phishguard_clean_candidate.csv"
OUTPUT_FILE = BASE_DIR / "DataSet" / "phishguard_ml_sample_10000.csv"

SAMPLES_PER_CLASS = 5000
RANDOM_STATE = 42

df = pd.read_csv(INPUT_FILE)

# First inspect the labels
print("\nLabel distribution in the cleaned dataset:")
print(df["label"].value_counts(dropna=False))

# Do not silently train on unknown or invalid labels
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df = df[df["label"].isin([0, 1])]

# Check that both classes have enough records
counts = df["label"].value_counts()

if len(counts) != 2 or (counts < SAMPLES_PER_CLASS).any():
    raise ValueError("Not enough records in both classes for sampling.")

# Sample across the complete dataset
sample = (
    df.groupby("label", group_keys=False)
      .sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_STATE)
)

# Shuffle the resulting sample
sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

sample.to_csv(OUTPUT_FILE, index=False)

print("\nDataset prepared successfully.")
print("Total rows:", len(sample))
print("\nFinal label distribution:")
print(sample["label"].value_counts())
print("\nSaved to:", OUTPUT_FILE)   