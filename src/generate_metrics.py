from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "hospital_readmission_dataset.csv"
)

MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"

METRICS_DIR.mkdir(exist_ok=True)


# ==================================================
# LOAD DATASET
# ==================================================

print("Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

target_column = "readmitted_30_days"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found."
    )

df = df.drop_duplicates()
df = df.dropna(subset=[target_column])

X = df.drop(columns=[target_column])
y = df[target_column]


# ==================================================
# SAME TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# EVALUATE SAVED MODELS
# ==================================================

model_files = [
    "logistic_regression_without_l2.joblib",
    "logistic_regression_with_l2.joblib"
]

all_metrics = []
classification_reports = {}

for model_file in model_files:

    model_path = MODELS_DIR / model_file

    if not model_path.exists():
        print(f"\nModel not found: {model_file}")
        continue

    print(f"\nEvaluating {model_file}...")

    model = joblib.load(model_path)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    confusion = confusion_matrix(
        y_test,
        predictions
    )

    true_negative, false_positive, false_negative, true_positive = (
        confusion.ravel()
    )

    model_metrics = {
        "model": model_file,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "test_samples": int(len(y_test))
    }

    all_metrics.append(model_metrics)

    classification_reports[model_file] = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    print(json.dumps(model_metrics, indent=4))


# ==================================================
# SAVE METRICS FILES
# ==================================================

json_path = METRICS_DIR / "model_metrics.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        all_metrics,
        file,
        indent=4
    )

csv_path = METRICS_DIR / "model_metrics.csv"

pd.DataFrame(all_metrics).to_csv(
    csv_path,
    index=False
)

reports_path = METRICS_DIR / "classification_reports.json"

with open(reports_path, "w", encoding="utf-8") as file:
    json.dump(
        classification_reports,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("METRICS GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"\nCreated: {json_path}")
print(f"Created: {csv_path}")
print(f"Created: {reports_path}")