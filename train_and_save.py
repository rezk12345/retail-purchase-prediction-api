import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "full_gen_data.csv")
ML_DIR = os.path.join(BASE_DIR, "app", "ml")
os.makedirs(ML_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()
df = df.dropna()

df["discount"] = 1 - df["ratio"]
df["retailweek"] = pd.to_datetime(df["retailweek"])
df["month"] = df["retailweek"].dt.month
df["quarter"] = df["retailweek"].dt.quarter
df_model = df.drop(columns=["article", "article.1", "customer_id", "retailweek", "ratio"])

cat_cols = ["country", "productgroup", "category", "style", "sizes", "gender"]
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

X = df_model.drop(columns=["label"])
y = df_model["label"]
model_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_res.value_counts().to_dict())

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
rf_for_ensemble = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

ensemble = VotingClassifier(
    estimators=[("lr", lr_pipe), ("rf", rf_for_ensemble)],
    voting="soft",
)
ensemble.fit(X_train_res, y_train_res)

y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1]))

joblib.dump(ensemble, os.path.join(ML_DIR, "model.joblib"))
joblib.dump(model_columns, os.path.join(ML_DIR, "model_columns.joblib"))
joblib.dump(cat_cols, os.path.join(ML_DIR, "cat_cols.joblib"))

print(f"Saved model files to {ML_DIR}")