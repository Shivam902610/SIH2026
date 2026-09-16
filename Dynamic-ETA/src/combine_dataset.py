import pandas as pd
import glob
import os


# ============================================
# Find all processed CSV files
# ============================================

files = glob.glob("data/processed/*.csv")

print("================================")
print("CLEANING TRAIN DATA")
print("================================")

print("\nCSV files found:", len(files))

for file in files:
    print("-", file)


# ============================================
# Read all CSV files
# ============================================

dataframes = []

for file in files:

    df = pd.read_csv(file)

    # ========================================
    # Check station_sequence column
    # ========================================

    if "station_sequence" not in df.columns:

        print(
            "\nAdding missing station_sequence:",
            file
        )

        # This old CSV doesn't have the column.
        # Its rows are already in route order.
        df["station_sequence"] = range(
            1,
            len(df) + 1
        )

    else:

        # Column exists, but may contain NaN
        if df["station_sequence"].isna().any():

            print(
                "\nFixing missing station_sequence:",
                file
            )

            df["station_sequence"] = range(
                1,
                len(df) + 1
            )

    dataframes.append(df)


# ============================================
# Combine all journeys
# ============================================

master_df = pd.concat(
    dataframes,
    ignore_index=True
)


# ============================================
# Remove exact duplicate rows
# ============================================

duplicates = master_df.duplicated().sum()

print("\nDuplicate rows found:", duplicates)

if duplicates > 0:
    master_df = master_df.drop_duplicates()


# ============================================
# Sort dataset
# ============================================

master_df = master_df.sort_values(
    [
        "train_number",
        "journey_date",
        "station_sequence"
    ]
).reset_index(drop=True)


# ============================================
# Save clean dataset
# ============================================

os.makedirs(
    "data/dataset",
    exist_ok=True
)

output_file = (
    "data/dataset/train_history_clean.csv"
)

master_df.to_csv(
    output_file,
    index=False
)


# ============================================
# Dataset summary
# ============================================

print("\n================================")
print("CLEAN DATASET CREATED")
print("================================")

print(
    "Total rows:",
    len(master_df)
)

print(
    "Total columns:",
    len(master_df.columns)
)

print(
    "Different trains:",
    master_df["train_number"].nunique()
)

print(
    "Different journeys:",
    master_df[
        ["train_number", "journey_date"]
    ].drop_duplicates().shape[0]
)


# ============================================
# Rows per train
# ============================================

print("\nRows per train:")

print(
    master_df["train_number"]
    .value_counts()
)


# ============================================
# Missing values
# ============================================

print("\nMissing values:")

print(
    master_df.isnull().sum()
)


# ============================================
# Final sequence check
# ============================================

print(
    "\nMissing station_sequence:",
    master_df["station_sequence"].isna().sum()
)


print("\nSaved to:")

print(output_file)

print("\n================================")
print("CLEANING COMPLETE")
print("================================")