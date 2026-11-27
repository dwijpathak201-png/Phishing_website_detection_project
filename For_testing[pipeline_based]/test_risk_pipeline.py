
"""
PhishGuard V3 - Hybrid Risk Pipeline Test.

Connects:
1. Centralized feature extraction
2. Random Forest V3 prediction
3. Hybrid risk scoring

Live webpage inspection is disabled.
Class 1 is provisionally treated as phishing
for development testing only.
"""

import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection_logic.Centralized_feature_pipelines import (
    extract_all_features
)

from Models.predict_v3 import predict_v3

from detection_logic.risk_scoring import (
    evaluate_risk
)


def analyze_url_v3(url):

    print("\n====================================")
    print("PHISHGUARD V3 HYBRID ANALYSIS")
    print("====================================")

    print("\n[1/3] Extracting URL features")

    report = extract_all_features(
        url,
        check_live_webpage=False
    )

    numeric_features = report["numeric_features"]

    print("URL:", url)
    print("Features extracted:", len(numeric_features))

    print("\n[2/3] Running Random Forest V3")

    prediction = predict_v3(numeric_features)

    print(
        "Predicted class:",
        prediction["predicted_label"]
    )

    print(
        "P(class 1):",
        round(
            prediction["probability_label_1"],
            4
        )
    )

    print("\n[3/3] Calculating hybrid risk")

    # DEVELOPMENT ASSUMPTION:
    # Class 1 = phishing.
    # This mapping must be verified from
    # the original dataset source before release.

    risk_result = evaluate_risk(
        report=report,
        ml_prediction=prediction["predicted_label"],
        ml_probability=prediction["probability_label_1"],
        phishing_label=1
    )

    print("\n========== FINAL ASSESSMENT ==========")

    pprint(risk_result)

    print(
        "\nWARNING: Risk scoring is provisional "
        "until class mapping and thresholds "
        "are validated."
    )

    return {
        "url": url,
        "features": report,
        "prediction": prediction,
        "risk_assessment": risk_result
    }


if __name__ == "__main__":

    test_urls = [
        "https://example.com",
        "https://github.com"
    ]

    for test_url in test_urls:
        analyze_url_v3(test_url)

    print("\nV3 hybrid integration test complete.")
