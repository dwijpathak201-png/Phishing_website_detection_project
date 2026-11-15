
"""
PhishGuard - Trained Model Prediction
"""

from pathlib import Path

import joblib
import pandas as pd

# Path to trained model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "saved_model"
    / "phishguard_model.joblib"
)

# Load the saved model package
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

package = joblib.load(MODEL_PATH)

model = package["model"]
FEATURE_COLUMNS = package["feature_columns"]


def predict_phishing(features):
    """
    Accepts a dictionary of extracted URL features.

    Returns:
        predicted_label
        probability_label_1
    """

    # Check required features
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in features
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    # Arrange features in training order
    input_data = pd.DataFrame(
        [{col: features[col] for col in FEATURE_COLUMNS}]
    )

    # Ensure numerical input
    input_data = input_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if input_data.isna().any().any():
        raise ValueError(
            "Feature values contain invalid or missing numbers."
        )

    # Predict class
    prediction = int(model.predict(input_data)[0])

    # Probability for numerical class 1
    class_labels = list(model.classes_)
    class_1_index = class_labels.index(1)

    probability = float(
        model.predict_proba(input_data)[0][class_1_index]
    )

    return {
        "predicted_label": prediction,
        "probability_label_1": probability
    }
