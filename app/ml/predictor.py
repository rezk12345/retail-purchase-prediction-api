import os
import joblib
import pandas as pd

ML_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(ML_DIR, "model.joblib"))
model_columns = joblib.load(os.path.join(ML_DIR, "model_columns.joblib"))
cat_cols = joblib.load(os.path.join(ML_DIR, "cat_cols.joblib"))


def predict(data: dict) -> tuple[int, float]:
    df = pd.DataFrame([data])

    df["ratio"] = df["current_price"] / df["regular_price"]
    df["discount"] = 1 - df["ratio"]
    df["retailweek"] = pd.to_datetime(df["retailweek"])
    df["month"] = df["retailweek"].dt.month
    df["quarter"] = df["retailweek"].dt.quarter
    df = df.drop(columns=["retailweek", "ratio"])

    df = pd.get_dummies(df, columns=cat_cols)
    df = df.reindex(columns=model_columns, fill_value=0)

    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(probability >= 0.5)
    return prediction, probability