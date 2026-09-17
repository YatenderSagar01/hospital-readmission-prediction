from pathlib import Path
import warnings

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ==================================================
# 1. DATA LOAD
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "hospital_readmission_dataset.csv"
)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    print("=" * 60)
    print("1. DATA LOAD")
    print("=" * 60)

    print(f"Checking dataset at:\n{DATA_PATH}")

    # Check whether the real dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "\nReal dataset was not found.\n"
            f"Please place the dataset here:\n{DATA_PATH}"
        )

    # Check whether the file is empty
    if DATA_PATH.stat().st_size == 0:
        raise ValueError("The dataset file is empty.")

    # Load the real dataset
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    print("\nReal dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset columns:")
    print(df.columns.tolist())

    return df


# ==================================================
# 2. PRE-PROCESSING / DATA CLEANING
# ==================================================

def clean_data(df):
    print("\n" + "=" * 60)
    print("2. PRE-PROCESSING / DATA CLEANING")
    print("=" * 60)

    target_column = "readmitted_30_days"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicate_count}")

    df = df.drop_duplicates()

    # Remove rows where target value is missing
    missing_target_count = df[target_column].isna().sum()

    print(
        f"Rows with missing target values: "
        f"{missing_target_count}"
    )

    df = df.dropna(subset=[target_column])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"\nCleaned dataset shape: {df.shape}")

    print("\nTarget distribution:")
    print(y.value_counts())

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    print("\nData cleaning and preprocessing completed.")

    return X, y, preprocessor


# ==================================================
# 3. ML MODEL
# ==================================================

def create_models():
    print("\n" + "=" * 60)
    print("3. ML MODEL")
    print("=" * 60)

    models = {
        "Logistic Regression Without L2": LogisticRegression(
            penalty=None,
            max_iter=2000,
            random_state=42
        ),

        "Logistic Regression With L2": LogisticRegression(
            penalty="l2",
            max_iter=2000,
            random_state=42
        )
    }

    print("Models created:")
    for model_name in models:
        print(f"- {model_name}")

    return models


# ==================================================
# 4. TRAINING WITH AND WITHOUT L2
# ==================================================

def train_and_evaluate(
    model_name,
    model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\n" + "=" * 60)
    print(f"4. TRAINING: {model_name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)

    # Probabilities for ROC-AUC
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # ROC-AUC
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    print("\nAccuracy:")
    print(round(accuracy, 4))

    print("\nROC-AUC:")
    print(round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nFalse Negative:")
    print(false_negative)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # Save trained model
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = MODELS_DIR / filename

    joblib.dump(
        pipeline,
        model_path
    )

    print(f"Model saved at: {model_path}")

    return pipeline


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():
    # 1. Load real dataset
    df = load_data()

    # 2. Clean and preprocess data
    X, y, preprocessor = clean_data(df)

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining data size:", len(X_train))
    print("Testing data size:", len(X_test))

    # 3. Create ML models
    models = create_models()

    # 4. Train and evaluate both models
    for model_name, model in models.items():
        train_and_evaluate(
            model_name,
            model,
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test
        )

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated model files:")
    for model_file in MODELS_DIR.glob("*.joblib"):
        print(model_file.name)


if __name__ == "__main__":
    main()