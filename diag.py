import pandas as pd
from src.data import load

df = load("data/raw/jewlery.csv")
vc = df.user_id.value_counts()
print(vc.head(5))
print("share of rows from top user:", round(vc.iloc[0] / len(df), 3))
print()
pc = df.product_id.value_counts()
print(pc.head(10))
print("top-10 products share of rows:", round(pc.head(10).sum() / len(df), 3))
print()
print(df.groupby(df.event_time.dt.to_period("M")).size().to_string())
print()
print(df.head(5).to_string())