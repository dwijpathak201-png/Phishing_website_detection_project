
"""
PhishGuard - V3 Machine Learning Training

Models:
1. Random Forest
2. Decision Tree
3. Logistic Regression

Dataset:
phishguard_ml_features_v3.csv

Data split:
60% Training
20% Validation
20% Testing

Outputs:
phishguard_model_v3.joblib
training_metadata_v3.json

Existing V1 and V2 models are preserved.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_config import FEATURE_COLUMNS, TARGET_COLUMN


DATASET_PATH = (
    PROJECT_ROOT
    / "For_testing[pipeline_based]"
    / "phishguard_ml_features_v3.csv"
)

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "saved_model"
)

MODEL_PATH = (
    MODEL_DIR / "phishguard_model_v3.joblib"
)

METADATA_PATH = (
    MODEL_DIR / "training_metadata_v3.json"
)

RANDOM_STATE = 42

# HTTPS was not available reliably in the source data.
# Exclude it from the ML model, but keep it available
# to the rule-based risk assessment system.

V3_FEATURE_COLUMNS = [
    feature
    for feature in FEATURE_COLUMNS
    if feature != "is_https"
]


# ==================================================
# 2. MODEL EVALUATION FUNCTION
# ==================================================

def evaluate_model(model, X_data, y_data):

    predictions = model.predict(X_data)

    return {

        "accuracy": float(
            accuracy_score(y_data, predictions)
        ),

        "precision": float(
            precision_score(
                y_data,
                predictions,
                pos_label=1,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_data,
                predictions,
                pos_label=1,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_data,
                predictions,
                pos_label=1,
                zero_division=0
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_data,
            predictions,
            labels=[0, 1]
        ).tolist()
    }


# ==================================================
# 3. MAIN TRAINING FUNCTION
# ==================================================

def main():

    print("\n========== PHISHGUARD V3 TRAINING ==========")
    print("Dataset:", DATASET_PATH)

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"V3 dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    print("\nOriginal dataset rows:", len(df))

    if "url" in df.columns:
        URL_COLUMN = "url"

    elif "normalized_url" in df.columns:
        URL_COLUMN = "normalized_url"

    else:
        raise ValueError("URL column not found.")

    required_columns = (
        [URL_COLUMN, TARGET_COLUMN, "extraction_success"]
        + V3_FEATURE_COLUMNS
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------
    # VERIFY EXTRACTION SUCCESS
    # --------------------------------------------------

    success_mask = (
        df["extraction_success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    successful_count = int(success_mask.sum())
    failed_count = int((~success_mask).sum())

    print("\nSuccessful extractions:", successful_count)
    print("Failed extractions:", failed_count)

    if failed_count > 0:

        if "error" in df.columns:
            print(
                df.loc[~success_mask, "error"]
                .value_counts()
                .head(10)
                .to_string()
            )

        raise ValueError(
            "V3 dataset contains failed extractions. "
            "Correct them before training."
        )

    # --------------------------------------------------
    # CLEAN URLS AND LABELS
    # --------------------------------------------------

    df[URL_COLUMN] = (
        df[URL_COLUMN]
        .astype("string")
        .str.strip()
    )

    if (
        df[URL_COLUMN].isna().any()
        or df[URL_COLUMN].eq("").any()
    ):
        raise ValueError(
            "Dataset contains missing or empty URLs."
        )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "Labels must be 0 or 1."
        )

    # --------------------------------------------------
    # VALIDATE NUMERIC FEATURES
    # --------------------------------------------------

    for feature in V3_FEATURE_COLUMNS:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df[V3_FEATURE_COLUMNS] = (
        df[V3_FEATURE_COLUMNS].replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    invalid_columns = [
        col for col in V3_FEATURE_COLUMNS
        if df[col].isna().any()
    ]

    if invalid_columns:
        raise ValueError(
            f"Invalid numeric features: {invalid_columns}"
        )

    # V3 should normalize an empty URL path to "/".
    if "path_length" in df.columns:

        empty_path_count = int(
            df["path_length"].eq(0).sum()
        )

        print(
            "Unnormalized empty paths:",
            empty_path_count
        )

        if empty_path_count > 0:
            raise ValueError(
                "Some URLs still have path_length=0. "
                "Regenerate the dataset using "
                "the corrected V3 URL extractor."
            )

    # --------------------------------------------------
    # CHECK DUPLICATES
    # --------------------------------------------------

    conflicts = (
        df.groupby(URL_COLUMN)[TARGET_COLUMN]
        .nunique()
    )

    if conflicts.gt(1).any():
        raise ValueError(
            "Identical URLs have conflicting labels."
        )

    original_count = len(df)

    df = df.drop_duplicates(
        subset=[URL_COLUMN]
    ).copy()

    print(
        "Duplicate URLs removed:",
        original_count - len(df)
    )

    if len(df) < 1000:
        raise ValueError(
            "Insufficient data for training."
        )

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Both target classes are required."
        )

    # --------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------

    print("\n========== DATASET SUMMARY ==========")

    print("Valid rows:", len(df))
    print("Feature count:", len(V3_FEATURE_COLUMNS))

    print("\nClass distribution:")
    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # FEATURES AND LABELS
    # --------------------------------------------------

    X = df[V3_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    # --------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_dev
        )
    )

    print("\n========== DATA SPLITTING ==========")

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Testing rows:", len(X_test))

    # --------------------------------------------------
    # INITIALIZE CLASSIFIERS
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
        ])
    }

    # --------------------------------------------------
    # TRAIN THREE MODELS
    # --------------------------------------------------

    validation_results = {}

    for name, model in models.items():

        print(
            f"\n========== TRAINING {name} ==========",
            flush=True
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_val,
            y_val
        )

        validation_results[name] = metrics

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )
        print(
            f"Precision: {metrics['precision']:.4f}"
        )
        print(
            f"Recall:    {metrics['recall']:.4f}"
        )
        print(
            f"F1-score:  {metrics['f1_score']:.4f}"
        )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    print("\n========== MODEL COMPARISON ==========")

    for name, metrics in validation_results.items():

        print(
            f"{name:<22} "
            f"Accuracy={metrics['accuracy']:.4f} "
            f"F1={metrics['f1_score']:.4f}"
        )

    best_validation_model = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["f1_score"]
        )
    )

    print(
        "\nBest validation model:",
        best_validation_model
    )

    # Keep Random Forest as V3 baseline,
    # even if another model wins validation.

    # --------------------------------------------------
    # RETRAIN RANDOM FOREST
    # --------------------------------------------------

    print("\n========== FINAL RANDOM FOREST ==========")

    best_model = models["Random Forest"]

    # Train on the entire development partition.
    best_model.fit(X_dev, y_dev)

    # Evaluate on untouched test partition.
    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    predictions = best_model.predict(X_test)

    print("\n========== RANDOM FOREST V3 TEST ==========")

    print(
        f"Accuracy:  {test_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision: {test_metrics['precision']:.4f}"
    )
    print(
        f"Recall:    {test_metrics['recall']:.4f}"
    )
    print(
        f"F1-score:  {test_metrics['f1_score']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    feature_importances = dict(
        sorted(
            zip(
                V3_FEATURE_COLUMNS,
                [
                    float(value)
                    for value in best_model.feature_importances_
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
    # SAVE TRAINED MODEL
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_package = {

        "model": best_model,

        "model_name": "Random Forest",

        "model_version": "v3",

        "feature_columns": list(
            V3_FEATURE_COLUMNS
        ),

        "positive_class": 1,

        "label_meaning_verified": False
    }

    joblib.dump(
        model_package,
        MODEL_PATH
    )

    print("\nV3 model saved successfully.")

    # --------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------

    metadata = {

        "project": "PhishGuard",

        "model_version": "v3",

        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "selected_model": "Random Forest",

        "best_validation_model": best_validation_model,

        "dataset_path": str(DATASET_PATH),

        "total_records": len(df),

        "feature_count": len(V3_FEATURE_COLUMNS),

        "features": list(V3_FEATURE_COLUMNS),

        "excluded_features": ["is_https"],

        "target_column": TARGET_COLUMN,

        "positive_class": 1,

        "label_meaning_verified": False,

        "label_distribution": {
            str(key): int(value)
            for key, value in y.value_counts().items()
        },

        "dataset_split": {
            "training": len(X_train),
            "validation": len(X_val),
            "testing": len(X_test)
        },

        "validation_results": validation_results,

        "final_test_results": test_metrics,

        "feature_importances": feature_importances
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    print("\n========================================")
    print("PHISHGUARD V3 TRAINING COMPLETED")
    print("========================================")

    print("Model:", MODEL_PATH)
    print("Metadata:", METADATA_PATH)

    print("\nV1 and V2 files remain unchanged.")
    print("Remember to verify the original class labels.")

    print("========================================")


# ==================================================
# 4. EXECUTE TRAINING
# ==================================================

if __name__ == "__main__":
    main()
