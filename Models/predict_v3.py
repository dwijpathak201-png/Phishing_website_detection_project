
"""
PhishGuard V3 Prediction Module.

Loads the corrected Random Forest V3.
Does not modify the existing V1/V2 models.
"""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "saved_model"
    / "phishguard_model_v3.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V3 model not found: {MODEL_PATH}"
    )

package = joblib.load(MODEL_PATH)

model = package["model"]
FEATURE_COLUMNS = package["feature_columns"]


def predict_v3(features: dict):
    """
    Predict numerical class using V3 features.
    """

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in features
    ]

    if missing:
        raise ValueError(
            f"Missing ML features: {missing}"
        )

    input_data = pd.DataFrame([{
        column: features[column]
        for column in FEATURE_COLUMNS
    }])

    input_data = input_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if input_data.isna().any().any():
        raise ValueError(
            "Invalid numerical feature values."
        )

    predicted_label = int(
        model.predict(input_data)[0]
    )

    class_labels = list(model.classes_)

    if 1 not in class_labels:
        raise ValueError(
            "Expected numerical class 1 is missing."
        )

    class_one_index = class_labels.index(1)

    probability_class_1 = float(
        model.predict_proba(input_data)[0][class_one_index]
    )

    return {
        "model_version": "v3",
        "predicted_label": predicted_label,
        "probability_label_1": probability_class_1,
        "label_meaning_verified": True,
    }
