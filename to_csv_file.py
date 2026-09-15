import pandas as pd

df = pd.read_csv("banana_labels.csv")

print(df.head())
print("\nColumns:")
print(df.columns.tolist())

print("\nTotal rows:", len(df))

print(
    "Labeled rows:",
    df["estimated_days_left"].notna().sum()
)

print("\nFirst 10 targets:")
print(df["estimated_days_left"].head(10))