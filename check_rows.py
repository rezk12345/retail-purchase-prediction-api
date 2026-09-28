import json
import pandas as pd

df = pd.read_csv("full_gen_data.csv")

fields = [
    "country", "productgroup", "category", "style", "sizes", "gender",
    "sales", "regular_price", "current_price", "cost", "promo1", "promo2",
    "retailweek", "rgb_r_main_col", "rgb_g_main_col", "rgb_b_main_col",
    "rgb_r_sec_col", "rgb_g_sec_col", "rgb_b_sec_col",
]

for label in [1, 0]:
    row = df[df["label"] == label].sample(1, random_state=1).iloc[0]
    data = {f: row[f] for f in fields}
    data["retailweek"] = str(pd.to_datetime(row["retailweek"]).date())
    print(f"----- Real row with label = {label} -----")
    print(json.dumps(data, indent=2, default=lambda x: x.item()))