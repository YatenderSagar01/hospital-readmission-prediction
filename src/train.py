"""Case Study 2: Hospital readmission prediction.

Workflow:
1. Data loading
2. Data cleaning/pre-processing
3. ML model
4. Training without L2 and with L2 regularization
5. ROC-AUC evaluation
6. False-negative analysis
"""

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def make_dataset(n_samples=5000, random_state=42):
    """Create a reproducible educational dataset if no CSV is supplied."""

    rng = np.random.default_rng(random_state)

    data = pd.DataFrame({
        "age": rng.integers(18, 90, n_samples),
        "heart_rate": rng.normal(80, 15, n_samples),
        "systolic_bp": rng.normal(125, 20, n_samples),
        "previous_visits": rng.poisson(2, n_samples),
        "num_diagnoses": rng.integers(1, 8, n_samples),
        "length_of_stay": rng.integers(1, 15, n_samples),
        "emergency_admission": rng.integers(0, 2, n_samples),
    })

    score = (
        0.03 * data["age"]
        + 0.35 * data["previous_visits"]
        + 0.25 * data["num_diagnoses"]
        + 0.08 * data["length_of_stay"]
        + 0.7 * data["emergency_admission"]
        + rng.normal(0, 2, n_samples)
        - 4.5
    )

    probability = 1 / (1 + np.exp(-score))
    data["readmitted_30_days"] = (
        rng.random(n_samples) < probability
    ).astype(int)

    return data


def load_and_clean_data():
    """Load and clean the hospital readmission dataset."""

    data_dir = ROOT / "data"

    # Find the first CSV file inside data/
    csv_files = list(data_dir.glob("*.csv"))

    if csv_files:
        csv_path = csv_files[0]
        print(f"Loading dataset: {csv_path}")
        data = pd.read_csv(csv_path)
    else:
        print("No CSV found. Creating synthetic dataset.")
        data = make_dataset()

    # Clean column names
    data.columns = data.columns.str.strip()

    # Remove duplicate rows
    data = data.drop_duplicates().copy()

    # Replace infinite values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Support both possible target names
    if "readmitted_30_days" in data.columns:
        target = "readmitted_30_days"
    elif "readmitted" in data.columns:
        target = "readmitted"
    else:
        raise ValueError(
            "Target column not found. Expected "
            "'readmitted_30_days' or 'readmitted'. "
            f"Available columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target])
    y = pd.to_numeric(data[target], errors="coerce")

    # Remove rows where target is missing
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    print(f"Target column: {target}")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")

    return X, y


def create_preprocessor(X):
    """Create preprocessing for numeric and categorical columns."""

    numeric_columns = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return preprocessor


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train and evaluate one model."""

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    metrics = {
        "model": name,
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "accuracy": float(
            (predictions == y_test).mean()
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }

    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"False negatives: {fn}")
    print(f"False positives: {fp}")
    print(f"True negatives: {tn}")
    print(f"True positives: {tp}")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return metrics


def main():
    # 1-2. Data loading and preprocessing
    X, y = load_and_clean_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = create_preprocessor(X)

    # 3-4. Logistic Regression without L2 and with L2
    without_l2 = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=2000
            )
        ),
    ])

    with_l2 = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000
            )
        ),
    ])

    # 5-6. ROC-AUC and false-negative analysis
    results = [
        evaluate_model(
            "without_l2",
            without_l2,
            X_train,
            X_test,
            y_train,
            y_test
        ),
        evaluate_model(
            "with_l2",
            with_l2,
            X_train,
            X_test,
            y_train,
            y_test
        ),
    ]

    # Save complete metrics
    with open(
        RESULTS_DIR / "case_study_2_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    # Save comparison table
    comparison = pd.DataFrame([
        {
            "model": item["model"],
            "roc_auc": item["roc_auc"],
            "accuracy": item["accuracy"],
            "false_negatives": item["false_negatives"],
            "false_positives": item["false_positives"],
            "true_negatives": item["true_negatives"],
            "true_positives": item["true_positives"],
        }
        for item in results
    ])

    comparison.to_csv(
        RESULTS_DIR / "case_study_2_comparison.csv",
        index=False
    )

    print(
        f"\nResults saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()