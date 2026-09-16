"""Hospital readmission prediction using Logistic Regression with L2 regularization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)


def make_dataset(n_samples=5000, random_state=42):
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


def main():
    data = make_dataset()
    X = data.drop(columns="readmitted")
    y = data["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=1000
        )),
    ])
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification report:\n", classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Hospital Readmission Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve - Hospital Readmission")
    plt.tight_layout()
    plt.show()

    coefficients = model.named_steps["logistic_regression"].coef_[0]
    importance = pd.DataFrame({"Feature": X.columns, "Coefficient": coefficients})
    importance["Absolute Importance"] = importance["Coefficient"].abs()
    print("\nFeature coefficients:\n", importance.sort_values("Absolute Importance", ascending=False))


if __name__ == "__main__":
    main()
