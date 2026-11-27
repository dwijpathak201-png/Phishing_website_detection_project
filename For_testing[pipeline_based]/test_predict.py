
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from Models.predict import predict_phishing

DATASET_PATH = (
    BASE_DIR
    / "For_testing[pipeline_based]"
    / "phishguard_ml_features.csv"
)

# Read one previously extracted URL
df = pd.read_csv(DATASET_PATH, nrows=1)

features = df.iloc[0].to_dict()

result = predict_phishing(features)

print("\n========== PREDICTION TEST ==========")
print("Actual label:", features["label"])
print("Predicted label:", result["predicted_label"])
print(
    "Probability of label 1:",
    round(result["probability_label_1"], 4)
)
