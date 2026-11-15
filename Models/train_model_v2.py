
"""PhishGuard - V2 model training with corrected features."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_config import FEATURE_COLUMNS, TARGET_COLUMN

DATASET_PATH = (
    PROJECT_ROOT
    / "For_testing[pipeline_based]"
    / "phishguard_ml_features_v2.csv"
)

MODEL_DIR = Path(__file__).resolve().parent / "saved_model"

MODEL_PATH = MODEL_DIR / "phishguard_model_v2.joblib"
METADATA_PATH = MODEL_DIR / "training_metadata_v2.json"

RANDOM_STATE = 42

# The original dataset does not contain reliable
# protocol information. Exclude is_https from ML.
# It remains available for separate rule analysis.

V2_FEATURE_COLUMNS = [
    name for name in FEATURE_COLUMNS
    if name != "is_https"
]


# --------------------------------------------------
# 2. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y, predictions,
                pos_label=1,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, predictions,
                pos_label=1,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y, predictions,
                pos_label=1,
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions,
            labels=[0, 1]
        ).tolist(),
    }


def main():

    print("\n========== PHISHGUARD V2 TRAINING ==========")
    print("Dataset:", DATASET_PATH)

    # --------------------------------------------------
    # 3. LOAD AND CHECK DATASET
    # --------------------------------------------------

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            "V2 dataset not found. "
            "Run regenerate_features_v2.py first."
        )

    if "is_https" not in FEATURE_COLUMNS:
        raise ValueError(
            "Expected is_https in feature_config.py."
        )

    if not V2_FEATURE_COLUMNS:
        raise ValueError("No training features available.")

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("Original V2 rows:", len(df))

    url_column = (
        "url" if "url" in df.columns
        else "normalized_url"
    )

    required_columns = [
        url_column,
        TARGET_COLUMN,
        "extraction_success",
        *V2_FEATURE_COLUMNS,
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    # Do not silently discard failed extraction rows.
    # The previous failures were all class 1.

    successful = (
        df["extraction_success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    failed_count = int((~successful).sum())

    print("Successful extractions:", int(successful.sum()))
    print("Failed extractions:", failed_count)

    if failed_count:
        if "error" in df.columns:
            print(
                df.loc[~successful, "error"]
                .value_counts()
                .head(10)
                .to_string()
            )

        raise ValueError(
            "Some extractions still failed. "
            "Fix them and regenerate the V2 dataset "
            "before training."
        )

    # --------------------------------------------------
    # 4. CLEAN AND VALIDATE
    # --------------------------------------------------

    df[url_column] = (
        df[url_column]
        .astype("string")
        .str.strip()
    )

    if (
        df[url_column].isna().any()
        or df[url_column].eq("").any()
    ):
        raise ValueError("Missing or empty URLs found.")

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "Target labels must contain only 0 and 1."
        )

    for feature in V2_FEATURE_COLUMNS:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df[V2_FEATURE_COLUMNS] = (
        df[V2_FEATURE_COLUMNS].replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    if df[V2_FEATURE_COLUMNS].isna().any().any():
        bad_features = (
            df[V2_FEATURE_COLUMNS]
            .columns[
                df[V2_FEATURE_COLUMNS].isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            f"Invalid numeric features: {bad_features}"
        )

    conflicts = (
        df.groupby(url_column)[TARGET_COLUMN]
        .nunique()
    )

    if conflicts.gt(1).any():
        raise ValueError(
            "Identical URLs have conflicting labels."
        )

    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=[url_column]
    ).copy()

    print(
        "Duplicate URLs removed:",
        before_dedup - len(df)
    )

    if len(df) < 1000 or df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Insufficient data or missing target class."
        )

    print("\n========== DATASET SUMMARY ==========")
    print("Clean rows:", len(df))
    print("Training features:", len(V2_FEATURE_COLUMNS))
    print(
        "Class distribution:\n",
        df[TARGET_COLUMN].value_counts().sort_index()
    )

    constant_features = [
        feature
        for feature in V2_FEATURE_COLUMNS
        if df[feature].nunique() <= 1
    ]

    if constant_features:
        print(
            "WARNING - Constant features:",
            constant_features
        )

    # --------------------------------------------------
    # 5. DATA SPLITTING
    # --------------------------------------------------

    X = df[V2_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_dev
    )

    print("\n========== DATA SPLITTING ==========")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))

    # --------------------------------------------------
    # 6. INITIALIZE MODELS
    # --------------------------------------------------

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
    }

    # --------------------------------------------------
    # 7. TRAIN AND COMPARE
    # --------------------------------------------------

    validation_results = {}

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model, X_val, y_val
        )

        validation_results[name] = metrics

        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        print(
            f"Precision: {metrics['precision']:.4f}"
        )
        print(
            f"Recall: {metrics['recall']:.4f}"
        )
        print(
            f"F1-score: {metrics['f1_score']:.4f}"
        )

    best_validation_model = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["f1_score"]
        )
    )

    print("\nBest validation model:", best_validation_model)

    # Keep Random Forest as the explicit V2 baseline.
    # Other classifiers remain comparison models.

    # --------------------------------------------------
    # 8. FINAL RANDOM FOREST TRAINING
    # --------------------------------------------------

    rf_model = models["Random Forest"]

    # Train on development set: training + validation.
    rf_model.fit(X_dev, y_dev)

    # Evaluate once on untouched test data.
    test_metrics = evaluate_model(
        rf_model, X_test, y_test
    )

    test_predictions = rf_model.predict(X_test)

    print("\n========== RANDOM FOREST V2 TEST ==========")

    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:
        print(
            f"{metric_name}: "
            f"{test_metrics[metric_name]:.4f}"
        )

    print("\nConfusion Matrix:")
    print(
        np.array(test_metrics["confusion_matrix"])
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # 9. FEATURE IMPORTANCE
    # --------------------------------------------------

    feature_importances = dict(
        sorted(
            zip(
                V2_FEATURE_COLUMNS,
                [
                    float(value)
                    for value
                    in rf_model.feature_importances_
                ]
            ),
            key=lambda item: item[1],
            reverse=True
        )
    )

    print("\n========== TOP 10 FEATURES ==========")

    for feature, importance in list(
        feature_importances.items()
    )[:10]:
        print(
            f"{feature:<28} {importance:.4f}"
        )

    # --------------------------------------------------
    # 10. SAVE V2 MODEL SEPARATELY
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_package = {
        "model": rf_model,
        "model_name": "Random Forest",
        "model_version": "v2",
        "feature_columns": list(V2_FEATURE_COLUMNS),
        "positive_class": 1,
        "label_meaning_verified": False,
    }

    joblib.dump(
        model_package,
        MODEL_PATH
    )

    # --------------------------------------------------
    # 11. SAVE TRAINING METADATA
    # --------------------------------------------------

    metadata = {
        "project": "PhishGuard",
        "model_version": "v2",

        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "selected_model": "Random Forest",
        "best_validation_model": best_validation_model,

        "dataset_path": str(DATASET_PATH),
        "total_records": int(len(df)),

        "feature_count": len(V2_FEATURE_COLUMNS),
        "features": list(V2_FEATURE_COLUMNS),
        "excluded_features": ["is_https"],

        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "label_meaning_verified": False,

        "label_distribution": {
            str(key): int(value)
            for key, value
            in y.value_counts().items()
        },

        "dataset_split": {
            "training": len(X_train),
            "validation": len(X_val),
            "testing": len(X_test),
        },

        "validation_results": validation_results,
        "final_test_results": test_metrics,
        "feature_importances": feature_importances,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\n========== TRAINING COMPLETE ==========")
    print("V2 model:", MODEL_PATH)
    print("V2 metadata:", METADATA_PATH)
    print("V1 artifacts were not overwritten.")
    print(
        "NOTE: Verify class 1's meaning "
        "against the original dataset source."
    )


if __name__ == "__main__":
    main()
