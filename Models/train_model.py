
"""
PHISHGUARD - MACHINE LEARNING TRAINING

Models:
1. Random Forest
2. Decision Tree
3. Logistic Regression

Data Split:
60% Training
20% Validation
20% Testing

Outputs:
- phishguard_model.joblib
- training_metadata.json
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
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from feature_config import FEATURE_COLUMNS, TARGET_COLUMN

# Automatically search for the dataset
DATASET_FILES = list(
    BASE_DIR.rglob("phishguard_ml_features.csv")
)

if not DATASET_FILES:
    raise FileNotFoundError(
        "phishguard_ml_features.csv was not found "
        "inside the project directory."
    )

if len(DATASET_FILES) > 1:
    print("\nMultiple matching datasets were found:")
    for file in DATASET_FILES:
        print(file)

    raise ValueError(
        "Keep one intended training dataset or "
        "specify its path explicitly."
    )

DATASET_PATH = DATASET_FILES[0]

# Save models relative to this script
MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "phishguard_model.joblib"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

RANDOM_STATE = 42

print("\n========== PHISHGUARD ML ==========")
print("Dataset found:", DATASET_PATH)

# ==================================================
# 2. LOAD DATASET
# ==================================================

df = pd.read_csv(DATASET_PATH)

print("\nOriginal dataset rows:", len(df))

# Identify URL column
if "url" in df.columns:
    URL_COLUMN = "url"
elif "normalized_url" in df.columns:
    URL_COLUMN = "normalized_url"
else:
    raise ValueError("URL column not found.")

# Check configured features and label
required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# ==================================================
# 3. CLEAN AND VALIDATE DATASET
# ==================================================

# Keep successful extraction records
if "extraction_success" in df.columns:

    df = df[
        df["extraction_success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    ].copy()

# Clean URL values
df[URL_COLUMN] = (
    df[URL_COLUMN]
    .astype("string")
    .str.strip()
)

df = df.dropna(subset=[URL_COLUMN])
df = df[df[URL_COLUMN] != ""].copy()

# Validate labels
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

df = df[
    df[TARGET_COLUMN].isin([0, 1])
].copy()

# Convert features to numeric
for feature in FEATURE_COLUMNS:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

# Replace infinite values with missing values
df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows with missing features
df = df.dropna(subset=FEATURE_COLUMNS).copy()

# Check for conflicting labels
conflicting_urls = (
    df.groupby(URL_COLUMN)[TARGET_COLUMN]
    .nunique()
)

if (conflicting_urls > 1).any():
    raise ValueError(
        "Conflicting labels exist for identical URLs."
    )

# Remove duplicate URLs
df = df.drop_duplicates(
    subset=[URL_COLUMN]
).copy()

# Check sample size
if len(df) < 1000:
    raise ValueError(
        "Dataset is too small. Complete the "
        "10,000-URL extraction first."
    )

if df[TARGET_COLUMN].nunique() != 2:
    raise ValueError(
        "Both classes (0 and 1) are required."
    )

print("\n========== DATASET SUMMARY ==========")
print("Valid rows:", len(df))
print("Feature count:", len(FEATURE_COLUMNS))
print("Label distribution:")
print(df[TARGET_COLUMN].value_counts().sort_index())

# ==================================================
# 4. FEATURES AND TARGET
# ==================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].astype(int)

# ==================================================
# 5. STRATIFIED SPLITTING
# ==================================================

# 80% development, 20% testing
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Overall:
# Training = 60%
# Validation = 20%
# Testing = 20%
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_dev
)

print("\n========== DATA SPLITTING ==========")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# ==================================================
# 6. INITIALIZE MODELS
# ==================================================

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

# ==================================================
# 7. EVALUATION FUNCTION
# ==================================================

def evaluate_model(model, X_data, y_data):

    predictions = model.predict(X_data)

    cm = confusion_matrix(
        y_data,
        predictions,
        labels=[0, 1]
    )

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

        "confusion_matrix": cm.tolist()
    }

# ==================================================
# 8. TRAIN ALL THREE MODELS
# ==================================================

validation_results = {}

for name, model in models.items():

    print(f"\n========== TRAINING {name} ==========")

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

# ==================================================
# 9. MODEL COMPARISON
# ==================================================

print("\n========== MODEL COMPARISON ==========")

for name, metrics in validation_results.items():

    print(
        f"{name:<22} "
        f"Accuracy={metrics['accuracy']:.4f} "
        f"F1={metrics['f1_score']:.4f}"
    )

# Select model using validation F1-score
BEST_MODEL_NAME = max(
    validation_results,
    key=lambda name: validation_results[name]["f1_score"]
)

best_model = models[BEST_MODEL_NAME]

print("\nSelected model:", BEST_MODEL_NAME)

# ==================================================
# 10. RETRAIN SELECTED MODEL
# ==================================================

# Retrain using training + validation records
best_model.fit(X_dev, y_dev)

# Evaluate on untouched test set
test_metrics = evaluate_model(
    best_model,
    X_test,
    y_test
)

test_predictions = best_model.predict(X_test)

print("\n========== FINAL TEST RESULTS ==========")
print("Best model:", BEST_MODEL_NAME)

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

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )
)

# ==================================================
# 11. FEATURE IMPORTANCE
# ==================================================

feature_importances = {}

if hasattr(best_model, "feature_importances_"):

    importance_values = best_model.feature_importances_

    feature_importances = dict(
        sorted(
            zip(
                FEATURE_COLUMNS,
                [float(v) for v in importance_values]
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

else:
    print(
        "\nSelected model does not provide "
        "tree-based feature importance."
    )

# ==================================================
# 12. SAVE BEST MODEL
# ==================================================

model_package = {

    "model": best_model,

    "model_name": BEST_MODEL_NAME,

    "feature_columns": list(FEATURE_COLUMNS),

    "positive_class": 1

}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\nModel saved successfully.")

# ==================================================
# 13. SAVE TRAINING METADATA
# ==================================================

metadata = {

    "project": "PhishGuard",

    "trained_at": datetime.now(
        timezone.utc
    ).isoformat(),

    "selected_model": BEST_MODEL_NAME,

    "dataset_path": str(DATASET_PATH),

    "total_records": len(df),

    "feature_count": len(FEATURE_COLUMNS),

    "features": list(FEATURE_COLUMNS),

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

# ==================================================
# 14. FINAL OUTPUT
# ==================================================

print("\n========================================")
print("PHISHGUARD TRAINING COMPLETED")
print("========================================")

print("Selected model:", BEST_MODEL_NAME)
print("Model location:", MODEL_PATH)
print("Metadata location:", METADATA_PATH)

print("========================================")
