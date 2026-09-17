# Trained Models

The trained Logistic Regression models are binary `.joblib` files and must be generated locally from `src/train.py` before committing them to GitHub.

Expected files:

- `hospital_readmission_without_l2.joblib`
- `hospital_readmission_with_l2.joblib`

Run:

```bash
python src/train.py
```

The models will be created automatically in this directory.