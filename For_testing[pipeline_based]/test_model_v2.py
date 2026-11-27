
"""
PhishGuard V2 - Standalone Prediction Test

Tests the saved V2 model with freshly extracted
features. Does not modify V1 or perform web requests.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from detection_logic.INFO_POUCH.url_features import (
    extract_url_features
)
from detection_logic.INFO_POUCH.domain_features import (
    extract_domain_features
)

MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "saved_model"
    / "phishguard_model_v2.joblib"
)

package = joblib.load(MODEL_PATH)

model = package["model"]
feature_columns = package["feature_columns"]

print("\n=== PHISHGUARD V2 PREDICTION TEST ===")
print("Model:", package["model_name"])
print("Features expected:", len(feature_columns))

test_urls = [
    "https://example.com",
    "http://example.com/login",
    "https://github.com",
    "example.com/account/verify",
]

for url in test_urls:

    url_features = extract_url_features(url)
    domain_features = extract_domain_features(url)

    combined = {
        **url_features,
        **domain_features
    }

    missing = [
        feature for feature in feature_columns
        if feature not in combined
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    input_data = pd.DataFrame([{
        feature: combined[feature]
        for feature in feature_columns
    }])

    input_data = input_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if input_data.isna().any().any():
        raise ValueError(
            f"Invalid numeric values for: {url}"
        )

    predicted_class = int(
        model.predict(input_data)[0]
    )

    probabilities = model.predict_proba(
        input_data
    )[0]

    class_one_index = list(
        model.classes_
    ).index(1)

    class_one_probability = float(
        probabilities[class_one_index]
    )

    print("\nURL:", url)
    print("Hostname:", url_features["hostname"])
    print("Predicted numerical class:", predicted_class)
    print(
        "Model probability for class 1:",
        f"{class_one_probability:.4f}"
    )

print("\nV2 prediction compatibility test completed.")
