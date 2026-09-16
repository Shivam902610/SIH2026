import pandas as pd

FILE = "data/dataset/ml_dataset.csv"

df = pd.read_csv(FILE)

print("\n===== ML DATASET CHECK =====")

print("Shape:", df.shape)

print("\nColumns:")
for col in df.columns:
    print("-", col)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTrains:")
print(df["train_number"].value_counts())

print("\nTarget statistics:")
print(df["target_arrival_delay"].describe())

print("\nSample data:")
print(df.head(10))

print("\n===== CHECK COMPLETE =====")