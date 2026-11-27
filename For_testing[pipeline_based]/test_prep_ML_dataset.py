
import sys
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "DataSet" / "phishguard_ml_sample_10000.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "phishguard_ml_features.csv"
ERROR_FILE = OUTPUT_DIR / "phishguard_ml_errors.csv"

# --------------------------------------------------
# 2. IMPORT EXISTING PIPELINE
# --------------------------------------------------

sys.path.insert(0, str(BASE_DIR))

from detection_logic.Centralized_feature_pipelines import extract_all_features
from feature_config import FEATURE_COLUMNS

# --------------------------------------------------
# 3. CONFIGURATION
# --------------------------------------------------

CHECK_LIVE_WEBPAGE = False
CHECKPOINT_SIZE = 100

# Test 10 URLs first.
# Change to None after successful testing.
MAX_ROWS = None

# True = remove previous output and start fresh
START_FRESH = True

# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

if "url" in df.columns:
    URL_COLUMN = "url"
elif "normalized_url" in df.columns:
    URL_COLUMN = "normalized_url"
else:
    raise ValueError("URL column not found.")

if "label" not in df.columns:
    raise ValueError("Label column not found.")

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(subset=["label"])
df = df[df["label"].isin([0, 1])].copy()

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS).copy()

print("\n========== DATASET INFORMATION ==========")
print("Input rows:", len(df))
print("Label distribution:")
print(df["label"].value_counts().sort_index())

# --------------------------------------------------
# 5. PREPARE OUTPUT FILES
# --------------------------------------------------

if START_FRESH:
    for file in [OUTPUT_FILE, ERROR_FILE]:
        if file.exists():
            file.unlink()

completed_ids = set()

if not START_FRESH:
    for file in [OUTPUT_FILE, ERROR_FILE]:
        if file.exists():
            old_df = pd.read_csv(file)

            if "source_row_id" in old_df.columns:
                completed_ids.update(
                    old_df["source_row_id"].tolist()
                )

results = []
errors = []

processed_count = 0
successful_count = 0
failed_count = 0

# --------------------------------------------------
# 6. SAVE CHECKPOINT FUNCTION
# --------------------------------------------------

def save_checkpoint():

    if results:
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            mode="a" if OUTPUT_FILE.exists() else "w",
            header=not OUTPUT_FILE.exists(),
            index=False
        )
        results.clear()

    if errors:
        pd.DataFrame(errors).to_csv(
            ERROR_FILE,
            mode="a" if ERROR_FILE.exists() else "w",
            header=not ERROR_FILE.exists(),
            index=False
        )
        errors.clear()

# --------------------------------------------------
# 7. EXTRACT NUMERIC FEATURES FROM PIPELINE OUTPUT
# --------------------------------------------------

def get_numeric_features(pipeline_output):

    # The centralized pipeline may return a report dict.
    if isinstance(pipeline_output, dict):

        features = {}

        # Preferred location for consolidated features
        numeric_features = pipeline_output.get(
            "numeric_features", {}
        )

        if isinstance(numeric_features, dict):
            features.update(numeric_features)

        # Support nested feature sections
        for section_name in [
            "url_features",
            "domain_features",
            "dns_features",
            "webpage_features"
        ]:
            section = pipeline_output.get(section_name)

            if isinstance(section, dict):
                for key, value in section.items():
                    features.setdefault(key, value)

        # Support features directly in the report
        for feature in FEATURE_COLUMNS:
            if feature in pipeline_output:
                features.setdefault(
                    feature,
                    pipeline_output[feature]
                )

        return features

    # Also support a DataFrame result
    if isinstance(pipeline_output, pd.DataFrame):

        if pipeline_output.empty:
            raise ValueError("Empty feature DataFrame.")

        return pipeline_output.iloc[0].to_dict()

    # Support tuple/list return values if used
    if isinstance(pipeline_output, (tuple, list)):

        combined = {}

        for item in pipeline_output:
            if isinstance(item, (dict, pd.DataFrame)):
                combined.update(get_numeric_features(item))

        return combined

    raise TypeError(
        f"Unexpected pipeline output type: {type(pipeline_output)}"
    )

# --------------------------------------------------
# 8. FEATURE EXTRACTION LOOP
# --------------------------------------------------

for row_number, row in df.iterrows():

    source_row_id = int(row_number)

    if source_row_id in completed_ids:
        continue

    url = str(row[URL_COLUMN]).strip()
    label = int(row["label"])

    try:

        if not url or url.lower() == "nan":
            raise ValueError("Empty URL.")

        # FIX: Do not assume exactly two return values
        pipeline_output = extract_all_features(
            url,
            check_live_webpage=CHECK_LIVE_WEBPAGE
        )

        # Convert pipeline output into a feature dictionary
        feature_values = get_numeric_features(
            pipeline_output
        )

        # Validate the required 20 features
        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in feature_values
        ]

        if missing_features:
            raise ValueError(
                f"Missing ML features: {missing_features}"
            )

        result = {
            "source_row_id": source_row_id,
            "url": url,
            "label": label
        }

        # Extract only configured ML features
        for feature in FEATURE_COLUMNS:

            value = pd.to_numeric(
                feature_values[feature],
                errors="coerce"
            )

            if pd.isna(value):
                raise ValueError(
                    f"Invalid numeric value: {feature}"
                )

            result[feature] = value

        result["extraction_success"] = True
        result["error"] = ""

        results.append(result)
        successful_count += 1

    except Exception as exc:

        error_message = str(exc)

        errors.append({
            "source_row_id": source_row_id,
            "url": url,
            "label": label,
            "error": error_message
        })

        failed_count += 1

        # Display first few errors for easier debugging
        if failed_count <= 3:
            print(
                f"Extraction error at row {source_row_id}: "
                f"{error_message}"
            )

    processed_count += 1

    # Save after each checkpoint
    if processed_count % CHECKPOINT_SIZE == 0:

        save_checkpoint()

        print(
            f"Processed: {processed_count} | "
            f"Successful: {successful_count} | "
            f"Failed: {failed_count}"
        )

    # Stop early if the first 10 attempts all fail.
    # Prevents repeating a code error 10,000 times.
    if (
        processed_count == 10
        and successful_count == 0
    ):
        save_checkpoint()

        raise RuntimeError(
            "All first 10 extractions failed. "
            "Check the printed errors before retrying."
        )

# --------------------------------------------------
# 9. FINAL SAVE
# --------------------------------------------------

save_checkpoint()

# --------------------------------------------------
# 10. FINAL REPORT
# --------------------------------------------------

total_success = (
    len(pd.read_csv(OUTPUT_FILE))
    if OUTPUT_FILE.exists()
    else 0
)

total_failed = (
    len(pd.read_csv(ERROR_FILE))
    if ERROR_FILE.exists()
    else 0
)

print("\n========== EXTRACTION COMPLETE ==========")
print("Input rows:", len(df))
print("Successful rows:", total_success)
print("Failed rows:", total_failed)
print("Configured ML features:", len(FEATURE_COLUMNS))

print("\nOutput file:", OUTPUT_FILE)
print("Error file:", ERROR_FILE)

print("\n========== PROCESS FINISHED ==========")
