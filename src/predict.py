"""Predict hospital readmission using a saved Logistic Regression pipeline."""

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hospital_readmission_with_l2.joblib"


def predict_readmission(patient_data: dict) -> int:
    """Return 1 for predicted readmission within 30 days, otherwise 0."""
    model = joblib.load(MODEL_PATH)
    patient_df = pd.DataFrame([patient_data])
    return int(model.predict(patient_df)[0])


if __name__ == "__main__":
    print("Load a patient record and call predict_readmission(...) to get a prediction.")
