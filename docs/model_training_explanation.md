# Hospital Readmission Prediction

## Model Training Explanation

This document explains how the saved `.joblib` models in the `models/` folder were trained.

## 1. Project Objective

The objective is to predict whether a patient will be readmitted to the hospital within 30 days.

The target column is:

```text
readmitted_30_days
```

Target meaning:

- `0`: Patient was not readmitted within 30 days.
- `1`: Patient was readmitted within 30 days.

## 2. Data Loading

The training program loads the real CSV dataset from the `data/` folder. It does not generate fake or randomly created data.

```python
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "hospital_readmission_dataset.csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError("Real dataset was not found.")

if DATA_PATH.stat().st_size == 0:
    raise ValueError("The dataset file is empty.")

df = pd.read_csv(DATA_PATH)
```

## 3. Data Cleaning

Duplicate rows are removed. Rows with a missing target value are also removed.

```python
df = df.drop_duplicates()
df = df.dropna(subset=["readmitted_30_days"])
```

The features and target are separated:

```python
X = df.drop(columns=["readmitted_30_days"])
y = df["readmitted_30_days"]
```

- `X` contains the input features.
- `y` contains the output class.

## 4. Pre-processing

The program identifies numerical and categorical columns.

### Numerical preprocessing

Numerical missing values are replaced with the median, and the values are standardized using `StandardScaler`.

```python
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
```

### Categorical preprocessing

Categorical missing values are replaced with the most frequent value. Categorical values are converted into numerical values using One-Hot Encoding.

```python
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
```

The numerical and categorical steps are combined using `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)
```

## 5. Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Parameter explanation:

- `test_size=0.20`: 20% of the data is used for testing.
- `random_state=42`: Makes the split reproducible.
- `stratify=y`: Preserves the class distribution in both sets.

## 6. Machine Learning Models

Two Logistic Regression models are trained.

### Model A: Logistic Regression Without L2

```python
model_without_l2 = LogisticRegression(
    penalty=None,
    max_iter=2000,
    random_state=42
)
```

`penalty=None` means that no regularization penalty is applied.

### Model B: Logistic Regression With L2

```python
model_with_l2 = LogisticRegression(
    penalty="l2",
    max_iter=2000,
    random_state=42
)
```

L2 regularization penalizes large model coefficients. It helps control model complexity and may reduce overfitting.

## 7. Complete Training Pipeline

The preprocessing steps and model are combined in one `Pipeline`:

```python
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)
```

The model is trained using:

```python
pipeline.fit(X_train, y_train)
```

The same preprocessing is automatically applied during training, testing, and future predictions.

## 8. Predictions

Class predictions are generated using:

```python
predictions = pipeline.predict(X_test)
```

Prediction probabilities are generated for ROC-AUC:

```python
probabilities = pipeline.predict_proba(X_test)[:, 1]
```

The `[:, 1]` selects the probability of class `1`, meaning readmission within 30 days.

## 9. Evaluation Metrics

### Accuracy

```python
accuracy = accuracy_score(y_test, predictions)
```

Accuracy is the percentage of correctly classified test samples.

### ROC-AUC

```python
roc_auc = roc_auc_score(y_test, probabilities)
```

ROC-AUC measures how well the model separates the two classes.

### Confusion Matrix

```python
matrix = confusion_matrix(y_test, predictions)
```

The confusion matrix contains:

- True Negative: Actual `0`, predicted `0`.
- False Positive: Actual `0`, predicted `1`.
- False Negative: Actual `1`, predicted `0`.
- True Positive: Actual `1`, predicted `1`.

### False Negative

```python
true_negative, false_positive, false_negative, true_positive = matrix.ravel()

print("False Negative:", false_negative)
```

A false negative occurs when a patient is actually readmitted but the model predicts that the patient will not be readmitted.

## 10. Saving the Models as `.joblib`

After training, the complete pipeline is saved using Joblib:

```python
import joblib

joblib.dump(
    pipeline,
    "models/logistic_regression_without_l2.joblib"
)
```

For the L2 model:

```python
joblib.dump(
    pipeline,
    "models/logistic_regression_with_l2.joblib"
)
```

The `.joblib` file stores the complete trained pipeline, including:

- Data preprocessing steps.
- Encoders and scalers.
- The trained Logistic Regression model.
- Learned model parameters.

## 11. Loading a Saved Model

A saved model can be loaded without training again:

```python
model = joblib.load(
    "models/logistic_regression_with_l2.joblib"
)
```

It can then be used for prediction:

```python
prediction = model.predict(new_patient_data)
```

The new patient data must contain the same feature columns used during training.

## 12. Generated Files

```text
models/
├── logistic_regression_without_l2.joblib
└── logistic_regression_with_l2.joblib
```

## 13. Complete Workflow

```text
Real Dataset
    ↓
Data Loading
    ↓
Data Cleaning
    ↓
Numerical and Categorical Pre-processing
    ↓
Train-Test Split
    ↓
Logistic Regression Without L2
    ↓
Logistic Regression With L2
    ↓
Model Predictions
    ↓
Accuracy, ROC-AUC and False Negative
    ↓
Save Trained Pipelines as .joblib Files
```

## 14. Source Code Reference

The actual implementation is available in:

```text
src/train.py
```

The generated evaluation results are available in:

```text
metrics/
```
