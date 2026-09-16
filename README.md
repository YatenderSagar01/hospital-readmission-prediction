# Hospital Readmission Prediction

## Case Study 1: 30-Day Hospital Readmission Prediction

This project predicts whether a patient will be readmitted to the hospital within 30 days using Logistic Regression with L2 regularization.

### Objective

Use patient-level features such as diagnosis-related information, vital signs, previous visits, and hospital stay information to estimate 30-day readmission risk.

### Model

- Logistic Regression
- L2 regularization
- Feature standardization

### Evaluation

The project reports:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve

### Clinical Error Trade-off

A **false negative** means a patient who is actually at high risk is classified as low risk. This can result in a missed opportunity for follow-up care and preventive intervention.

A **false positive** means a lower-risk patient is classified as high risk. This can lead to additional monitoring, unnecessary follow-up, and increased use of healthcare resources.

The operating threshold should therefore be selected using clinical priorities and the relative costs of these two types of errors rather than relying on accuracy alone.

### Data

The included training script supports a self-contained synthetic dataset so the repository can run without exposing private patient information. The generated features represent plausible clinical variables but are **not real patient records** and must not be interpreted as evidence of clinical performance.

### Project Structure

```text
hospital-readmission-prediction/
├── data/
│   └── README.md
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

### Run

```bash
pip install -r requirements.txt
python src/train.py
```

### Notes

For a real clinical project, replace the synthetic data section with an appropriately licensed and de-identified dataset, document the cohort definition, handle missing values and categorical variables, and validate the model on an independent patient population.
