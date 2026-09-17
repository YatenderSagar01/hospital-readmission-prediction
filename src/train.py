"""Case Study 2: hospital readmission prediction.

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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_dataset(n_samples=5000, random_state=42):
    """Create a reproducible educational dataset when no real dataset is supplied."""
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
    data["readmitted"] = (rng.random(n_samples) < probability).astype(int)
    return data


def load_and_clean_data():
    """Load a CSV if supplied; otherwise create the reproducible demo dataset."""
    csv_candidates = [ROOT / "data" / "hospital_readmission.csv", ROOT / "data" / "hospital_readmission_dataset.csv"]
    csv_path = next((path for path in csv_candidates if path.exists()), None)

    if csv_path:
        data = pd.read_csv(csv_path)
    else:
        data = make_dataset()

    data = data.drop_duplicates().copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    target = "readmitted"
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is missing from the dataset.")

    X = data.drop(columns=[target])
    y = data[target].astype(int)
    return X, y


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float((predictions == y_test).mean()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }
    print(f"\n{name}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"False negatives: {fn}")
    print(classification_report(y_test, predictions, zero_division=0))
    return metrics


def main():
    # 1-2. Data load and cleaning
    X, y = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3-4. Same model architecture, trained without and with L2.
    common_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    without_l2 = Pipeline(common_steps + [
        ("model", LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000))
    ])
    with_l2 = Pipeline(common_steps + [
        ("model", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000))
    ])

    # 5-6. ROC-AUC and false-negative analysis
    results = [
        evaluate_model("without_l2", without_l2, X_train, X_test, y_train, y_test),
        evaluate_model("with_l2", with_l2, X_train, X_test, y_train, y_test),
    ]

    with open(RESULTS_DIR / "case_study_2_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    pd.DataFrame([
        {
            "model": item["model"],
            "roc_auc": item["roc_auc"],
            "accuracy": item["accuracy"],
            "false_negatives": item["false_negatives"],
            "false_positives": item["false_positives"],
        }
        for item in results
    ]).to_csv(RESULTS_DIR / "case_study_2_comparison.csv", index=False)

    print(f"\nSaved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
