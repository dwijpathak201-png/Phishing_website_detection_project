
from pathlib import Path
import pandas as pd

# Load dataset
DATASET_PATH = Path(__file__).parent / "phishguard_clean_candidate.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# Validate required columns
required = {"url", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

# Validate labels and URLs
print("\nUnique labels:", sorted(df["label"].dropna().unique()))
print("Missing or invalid labels:", (~df["label"].isin([0, 1])).sum())
print("Empty URLs:", (
    df["url"].isna() | df["url"].astype(str).str.strip().eq("")
).sum())
print("Duplicate URLs:", df["url"].duplicated().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset validation completed.")
