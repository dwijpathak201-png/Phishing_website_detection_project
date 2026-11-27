
from pathlib import Path
import sys

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection_logic.Centralized_feature_pipelines import (
    extract_all_features,
)


# ============================================================
# 1. TEST CONFIGURATION
# ============================================================

TEST_URL = "https://example.com"

EXPECTED_POUCHES = [
    "url_features",
    "domain_features",
    "dns_features",
    "webpage_features",
]

EXPECTED_NUMERIC_FEATURES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_percent_symbols",
    "num_path_segments",
    "path_length",
    "query_length",
    "num_query_parameters",
    "hostname_length",
    "has_ip_address",
    "is_https",
    "suspicious_keyword_count",
    "domain_length",
    "subdomain_count",
    "has_punycode",
    "has_dns_a_record",
    "has_dns_mx_record",
    "page_has_title",
    "page_form_count",
    "page_has_password_field",
    "page_has_iframe",
    "external_action_forms",
]


# ============================================================
# 2. TEST THE PIPELINE WITHOUT LIVE WEBPAGE INSPECTION
# ============================================================

print("\n--- Test 1: Pipeline without webpage inspection ---")

report = extract_all_features(
    TEST_URL,
    check_live_webpage=False,
)

# Check that all four pouches are present
missing_pouches = [
    pouch for pouch in EXPECTED_POUCHES
    if pouch not in report
]

assert not missing_pouches, (
    f"Missing pouches: {missing_pouches}"
)

print("All four Info Pouches returned successfully.")

# Check the combined numeric feature dictionary
numeric_features = report["numeric_features"]

missing_features = [
    feature for feature in EXPECTED_NUMERIC_FEATURES
    if feature not in numeric_features
]

assert not missing_features, (
    f"Missing numeric features: {missing_features}"
)

print("All 27 expected numeric feature keys are present.")

# Verify that skipped webpage inspection is represented correctly
webpage_features = report["webpage_features"]

assert webpage_features["page_title"] == "Inspection Skipped"
assert webpage_features["page_has_title"] is None
assert webpage_features["page_form_count"] is None

print("Skipped webpage inspection is represented correctly.")

print("\nDNS features:")
print(report["dns_features"])

print("\nTest 1 passed!")


# ============================================================
# 3. TEST LIVE WEBPAGE INSPECTION
# ============================================================

print("\n--- Test 2: Pipeline with live webpage inspection ---")
print("This test requires internet access and may take some time.")

try:
    live_report = extract_all_features(
        TEST_URL,
        check_live_webpage=True,
    )

    live_webpage_features = live_report["webpage_features"]

    # Check that webpage analysis returned its expected keys
    expected_webpage_keys = [
        "page_has_title",
        "page_form_count",
        "page_has_password_field",
        "page_has_iframe",
        "external_action_forms",
    ]

    missing_webpage_keys = [
        key for key in expected_webpage_keys
        if key not in live_webpage_features
    ]

    assert not missing_webpage_keys, (
        f"Missing webpage keys: {missing_webpage_keys}"
    )

    print("Webpage analyzer returned all expected feature keys.")
    print("Webpage status:",
          live_webpage_features.get("webpage_status"))
    print("Page title:",
          live_webpage_features.get("page_title"))
    print("HTTP status code:",
          live_webpage_features.get("status_code"))

    print("\nTest 2 passed: webpage feature keys verified.")

except Exception as error:
    print("\nTest 2 failed with an exception:")
    print(type(error).__name__, "-", error)
    print(
        "Inspect webpage_analysis.py and its error handling "
        "before treating live inspection as verified."
    )


print("\n--- Integration test run completed ---")
