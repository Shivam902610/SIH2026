import pandas as pd

# Load the master dataset
input_file = "data/dataset/train_12919_history.csv"

df = pd.read_csv(input_file)

print("Missing station_sequence before fix:",
      df["station_sequence"].isna().sum())


# Create a station_code -> station_sequence mapping
# using rows where sequence is already available
sequence_map = (
    df.dropna(subset=["station_sequence"])
      .drop_duplicates("station_code")
      .set_index("station_code")["station_sequence"]
      .to_dict()
)


# Fill missing sequence numbers using station_code
df["station_sequence"] = df["station_sequence"].fillna(
    df["station_code"].map(sequence_map)
)


print("Missing station_sequence after fix:",
      df["station_sequence"].isna().sum())


# Save the fixed dataset
df.to_csv(input_file, index=False)

print("\nDataset fixed successfully!")
print("Saved to:", input_file)

print("\nRows:", len(df))
print("Columns:", len(df.columns))