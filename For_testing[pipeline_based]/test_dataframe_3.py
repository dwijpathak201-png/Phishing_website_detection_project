#checkin for dns and webpage status: 

from pathlib import Path
import sys
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection_logic.Centralized_feature_pipelines import extract_all_features

DATASET_PATH = PROJECT_ROOT / "DataSet" / "phishguard_clean_candidate.csv"
OUTPUT_PATH = PROJECT_ROOT / "For_testing" / "central_pipeline_100.csv"

SAMPLE_SIZE = 100
CHECK_LIVE_WEBPAGE = True

df = pd.read_csv(DATASET_PATH)

if "normalized_url" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain normalized_url and label.")

# Use the same first 100 URLs consistently on every run.
sample = df.head(SAMPLE_SIZE).copy()

records = []

for index, (_, row) in enumerate(sample.iterrows(), start=1):
    url = str(row["normalized_url"]).strip()

    record = {
        "source_row_id": index - 1,
        "url": url,
        "label": row["label"],
        "pipeline_error": "",
    }

    try:
        report = extract_all_features(
            url,
            check_live_webpage=CHECK_LIVE_WEBPAGE,
        )

        record.update(report.get("numeric_features", {}))

        dns = report.get("dns_features", {})
        webpage = report.get("webpage_features", {})

        record["dns_status"] = dns.get("dns_status", "Not reported")
        record["dns_resolved"] = dns.get("dns_resolved", None)
        record["webpage_status"] = webpage.get(
            "webpage_status", "Not reported"
        )
        record["http_status_code"] = webpage.get("status_code", None)

    except Exception as exc:
        record["pipeline_error"] = f"{type(exc).__name__}: {exc}"

    records.append(record)

    if index % 10 == 0:
        print(f"Processed {index}/{len(sample)} URLs")

    # Avoid sending requests too rapidly.
    time.sleep(0.2)

result = pd.DataFrame(records)
result.to_csv(OUTPUT_PATH, index=False)

print("\nIntegration test finished.")
print("Output:", OUTPUT_PATH)
print("Rows:", len(result))
print("Rows with pipeline exceptions:",
      result["pipeline_error"].ne("").sum())

print("\nDNS status counts:")
print(result["dns_status"].value_counts(dropna=False))

print("\nWebpage status counts:")
print(result["webpage_status"].value_counts(dropna=False))

print("\nHTTP status counts:")
print(result["http_status_code"].value_counts(dropna=False))

print("\nNumeric feature columns:", len(
    result.select_dtypes(include="number").columns
))
print("Numeric feature keys missing:",
      [c for c in [
          "url_length", "num_dots", "num_hyphens", "num_underscores",
          "num_slashes", "num_question_marks", "num_equal_signs",
          "num_at_symbols", "num_percent_symbols", "num_path_segments",
          "path_length", "query_length", "num_query_parameters",
          "hostname_length", "has_ip_address", "is_https",
          "suspicious_keyword_count", "domain_length", "subdomain_count",
          "has_punycode", "has_dns_a_record", "has_dns_mx_record",
          "page_has_title", "page_form_count",
          "page_has_password_field", "page_has_iframe",
          "external_action_forms",
      ] if c not in result.columns])
