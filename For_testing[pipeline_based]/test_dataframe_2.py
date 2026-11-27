#checking the domain and url features:

from pathlib import Path
import sys
import pandas as pd

# 1. Project paths and imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_config import FEATURE_COLUMNS, TARGET_COLUMN
from detection_logic.INFO_POUCH.url_features import extract_url_features
from detection_logic.INFO_POUCH.domain_features import extract_domain_features

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "DataSet"
    / "phishguard_clean_candidate.csv"
)
OUTPUT_PATH = Path(__file__).parent / "phishguard_features_20.csv"
ERROR_PATH = Path(__file__).parent / "feature_extraction_errors_20.csv"

# Start with a small pilot. Change to None after validating it.
MAX_ROWS = 100
BATCH_SIZE = 25

# 2. Load and validate the input
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

required_columns = ["normalized_url", TARGET_COLUMN]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.reset_index(drop=True)
df["source_row_id"] = df.index

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS).copy()

print("Input rows to process:", len(df))

# 3. Load completed row IDs so interrupted runs can resume
if OUTPUT_PATH.exists():
    previous = pd.read_csv(OUTPUT_PATH)
    if "source_row_id" not in previous.columns:
        raise ValueError("Existing output lacks source_row_id.")
    completed_ids = set(
        pd.to_numeric(
            previous["source_row_id"], errors="coerce"
        ).dropna().astype(int)
    )
else:
    completed_ids = set()

pending = df[~df["source_row_id"].isin(completed_ids)]
print("Rows remaining:", len(pending))

# 4. Extract one row at a time
def extract_one(row):
    url = str(row["normalized_url"]).strip()

    record = {
        "source_row_id": int(row["source_row_id"]),
        "normalized_url": url,
        TARGET_COLUMN: row[TARGET_COLUMN],
        "extraction_success": False,
        "error": "",
    }

    try:
        url_data = extract_url_features(url)
        domain_data = extract_domain_features(url)

        combined = {**url_data, **domain_data}

        for feature in FEATURE_COLUMNS:
            if feature not in combined:
                raise ValueError(f"Missing extracted feature: {feature}")

            value = combined[feature]
            if not pd.api.types.is_number(value):
                raise TypeError(
                    f"Feature {feature} is not numeric: {value!r}"
                )

            record[feature] = value

        record["extraction_success"] = True

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record

# 5. Save each batch by appending, rather than rewriting the whole CSV
output_columns = [
    "source_row_id",
    "normalized_url",
    TARGET_COLUMN,
    *FEATURE_COLUMNS,
    "extraction_success",
    "error",
]

batch = []
processed = 0

for _, row in pending.iterrows():
    batch.append(extract_one(row))
    processed += 1

    if len(batch) >= BATCH_SIZE:
        batch_df = pd.DataFrame(batch).reindex(columns=output_columns)
        batch_df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=not OUTPUT_PATH.exists(),
            index=False,
        )

        completed_ids.update(batch_df["source_row_id"].astype(int))
        batch.clear()
        print(f"Processed {processed}/{len(pending)} pending rows")

if batch:
    batch_df = pd.DataFrame(batch).reindex(columns=output_columns)
    batch_df.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=not OUTPUT_PATH.exists(),
        index=False,
    )

# 6. Record failed rows separately
if OUTPUT_PATH.exists():
    saved = pd.read_csv(OUTPUT_PATH)
    errors = saved[
        saved["error"].fillna("").astype(str).ne("")
    ]
    errors.to_csv(ERROR_PATH, index=False)

    print("\nExtraction finished.")
    print("Output:", OUTPUT_PATH)
    print("Rows saved:", len(saved))
    print("Successful rows:", int(saved["extraction_success"].sum()))
    print("Failed rows:", len(errors))
    print("Feature columns missing:",
          [c for c in FEATURE_COLUMNS if c not in saved.columns])
else:
    print("No output was generated.")
