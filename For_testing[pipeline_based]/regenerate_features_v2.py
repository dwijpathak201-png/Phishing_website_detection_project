
"""
PhishGuard - Dataset Feature Regeneration V2

Re-extracts URL and domain features using the
corrected extraction pipeline.

Does not perform DNS or webpage requests.
Does not modify the original dataset.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_config import FEATURE_COLUMNS, TARGET_COLUMN

from detection_logic.INFO_POUCH.url_features import (
    extract_url_features
)
from detection_logic.INFO_POUCH.domain_features import (
    extract_domain_features
)


DATA_DIR = PROJECT_ROOT / "For_testing[pipeline_based]"

SOURCE_PATH = DATA_DIR / "phishguard_ml_features.csv"
OUTPUT_PATH = DATA_DIR / "phishguard_ml_features_v2.csv"

if not SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Original dataset not found: {SOURCE_PATH}"
    )

df = pd.read_csv(SOURCE_PATH)

if "url" not in df.columns or TARGET_COLUMN not in df.columns:
    raise ValueError("Required URL or target column is missing.")

updated_rows = []
failed = 0

for index, row in df.iterrows():

    record = row.to_dict()

    try:
        url = str(row["url"])

        url_features = extract_url_features(url)
        domain_features = extract_domain_features(url)

        combined = {
            **url_features,
            **domain_features
        }

        for feature in FEATURE_COLUMNS:
            record[feature] = combined[feature]

        record["extraction_success"] = True
        record["error"] = ""

    except Exception as exc:
        failed += 1

        for feature in FEATURE_COLUMNS:
            record[feature] = None

        record["extraction_success"] = False
        record["error"] = str(exc)

    updated_rows.append(record)


df_v2 = pd.DataFrame(updated_rows)

# Verify that labels were not changed.
if not df_v2[TARGET_COLUMN].equals(df[TARGET_COLUMN]):
    raise ValueError("Target labels changed unexpectedly.")

# Ensure all successful records contain numeric features.
successful = df_v2[
    df_v2["extraction_success"] == True
].copy()

numeric = successful[FEATURE_COLUMNS].apply(
    pd.to_numeric,
    errors="coerce"
)

if numeric.isna().any().any():
    raise ValueError(
        "Some successful rows contain invalid numeric features."
    )

df_v2.to_csv(OUTPUT_PATH, index=False)

print("\n=== PHISHGUARD V2 DATASET ===")
print("Original rows:", len(df))
print("Regenerated rows:", len(df_v2))
print("Successful extractions:", len(successful))
print("Failed extractions:", failed)

print("\nHostname length statistics:")
print(successful["hostname_length"].describe())

print("\nHTTPS feature distribution:")
print(successful["is_https"].value_counts())

print("\nClass distribution:")
print(df_v2[TARGET_COLUMN].value_counts().sort_index())

print("\nSaved to:", OUTPUT_PATH)
print("Original dataset preserved.")
