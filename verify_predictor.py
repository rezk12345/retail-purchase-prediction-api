import pandas as pd
from app.ml.predictor import predict, model, model_columns, cat_cols

df = pd.read_csv("full_gen_data.csv").drop_duplicates().dropna()

fields = [
    "country", "productgroup", "category", "style", "sizes", "gender",
    "sales", "regular_price", "current_price", "cost", "promo1", "promo2",
    "retailweek", "rgb_r_main_col", "rgb_g_main_col", "rgb_b_main_col",
    "rgb_r_sec_col", "rgb_g_sec_col", "rgb_b_sec_col",
]

nb = df.copy()
nb["discount"] = 1 - nb["ratio"]
nb["retailweek"] = pd.to_datetime(nb["retailweek"])
nb["month"] = nb["retailweek"].dt.month
nb["quarter"] = nb["retailweek"].dt.quarter
nb = nb.drop(columns=["article", "article.1", "customer_id", "retailweek", "ratio", "label"])
nb = pd.get_dummies(nb, columns=cat_cols, drop_first=True)
nb = nb.reindex(columns=model_columns, fill_value=0)

sample = df.sample(300, random_state=0)
notebook_preds = model.predict(nb.loc[sample.index])

api_preds = []
for _, row in sample.iterrows():
    prediction, probability = predict({f: row[f] for f in fields})
    api_preds.append(prediction)

api_preds = pd.Series(api_preds, index=sample.index)
labels = sample["label"]

print("API matches notebook:", round((api_preds == notebook_preds).mean() * 100, 1), "%")
print("API correct vs real label:", round((api_preds == labels).mean() * 100, 1), "%")
print("Label 0 rows predicted as Purchase:", int(((labels == 0) & (api_preds == 1)).sum()), "out of", int((labels == 0).sum()))
print("Label 1 rows predicted as Purchase:", int(((labels == 1) & (api_preds == 1)).sum()), "out of", int((labels == 1).sum()))