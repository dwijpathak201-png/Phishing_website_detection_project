#YA WALA TEST 100 URL KA LIYA HA:

from pathlib import Path
import sys
import pandas as pd

# ============================================================
# 1. PROJECT PATH AND IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_config import FEATURE_COLUMNS, TARGET_COLUMN
from detection_logic.INFO_POUCH.url_features import extract_url_features
from detection_logic.INFO_POUCH.domain_features import extract_domain_features


# ============================================================
# 2. LOAD THE CLEANED DATASET
# ============================================================

DATASET_PATH = Path(__file__).parent / "phishguard_clean_candidate.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total dataset rows:", len(df))


# ============================================================
# 3. SELECT A REPRODUCIBLE SAMPLE
# ============================================================

sample = df.sample(n=min(100, len(df)), random_state=42).copy()


# ============================================================
# 4. EXTRACT URL AND DOMAIN FEATURES
# ============================================================

url_features = sample["normalized_url"].apply(extract_url_features)
domain_features = sample["normalized_url"].apply(extract_domain_features)

url_df = pd.DataFrame(url_features.tolist())
domain_df = pd.DataFrame(domain_features.tolist())


# ============================================================
# 5. COMBINE THE RESULTS
# ============================================================

result = pd.concat(
    [
        sample[["normalized_url", TARGET_COLUMN]].reset_index(drop=True),
        url_df.reset_index(drop=True),
        domain_df.reset_index(drop=True),
    ],
    axis=1,
)

print("\nFeature extraction completed!")
print("Sample rows:", len(result))
print("Total columns:", len(result.columns))

print("\nExtracted feature columns:")
print(result.columns.tolist())

print("\nFirst 5 rows:")
print(result.head())


# ============================================================
# 6. SAVE THE EXTRACTED FEATURES
# ============================================================

OUTPUT_PATH = Path(__file__).parent / "feature_sample.csv"
result.to_csv(OUTPUT_PATH, index=False)

print("\nSaved to:", OUTPUT_PATH)


# ============================================================
# 7. VALIDATE THE GENERATED DATASET
# ============================================================

print("\n--- Feature Validation ---")

print("\nMissing values per column:")
print(result.isnull().sum())

print(
    "\nDuplicate normalized URLs:",
    result["normalized_url"].duplicated().sum(),
)

print("\nLabel distribution:")
print(result[TARGET_COLUMN].value_counts())


# ============================================================
# 8. IDENTIFY NUMERIC FEATURES
# ============================================================

metadata_columns = [
    "normalized_url",
    TARGET_COLUMN,
    "detected_keywords",
    "hostname",
    "path",
    "query",
    "subdomain",
    "domain_name",
    "suffix",
    "registered_domain",
]

numeric_feature_columns = [
    column
    for column in result.columns
    if column not in metadata_columns
]

print("\nNon-numeric feature columns:")
print(
    result[numeric_feature_columns]
    .select_dtypes(exclude="number")
    .columns.tolist()
)

print("\nNumeric feature data types:")
print(result[numeric_feature_columns].dtypes)


# ============================================================
# 9. VALIDATE AGAINST CENTRALIZED FEATURE CONFIGURATION
# ============================================================

print("\n--- Configuration Validation ---")

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in result.columns
]

if missing_features:
    print("Missing configured features:", missing_features)
else:
    print("All configured features are present.")

extra_configurable_features = [
    column
    for column in numeric_feature_columns
    if column not in FEATURE_COLUMNS
]

if extra_configurable_features:
    print("Numeric features not in configuration:",
          extra_configurable_features)
else:
    print("No extra numeric features found.")

print("Configured feature count:", len(FEATURE_COLUMNS))
print("Target column exists:", TARGET_COLUMN in result.columns)

# Check whether every configured feature is numeric
non_numeric_configured_features = [
    column
    for column in FEATURE_COLUMNS
    if column in result.columns
    and not pd.api.types.is_numeric_dtype(result[column])
]

if non_numeric_configured_features:
    print(
        "Configured features that are not numeric:",
        non_numeric_configured_features,
    )
else:
    print("All present configured features are numeric.")

print("\nFeature validation completed!")
