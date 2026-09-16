import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "train_history_clean.csv"
)

EDA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "eda"
)

os.makedirs(EDA_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRAIN DELAY - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# BASIC CLEANING
# ============================================================

df["train_number"] = df["train_number"].astype(str)

df["journey_date"] = pd.to_datetime(
    df["journey_date"],
    errors="coerce"
)

df["arrival_delay"] = pd.to_numeric(
    df["arrival_delay"],
    errors="coerce"
)

df["departure_delay"] = pd.to_numeric(
    df["departure_delay"],
    errors="coerce"
)

df["station_sequence"] = pd.to_numeric(
    df["station_sequence"],
    errors="coerce"
)


# ============================================================
# RAW DATA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RAW DATASET SUMMARY")
print("=" * 60)

print("\nUnique trains:")
print(df["train_number"].nunique())

print("\nUnique journeys:")
print(
    df[
        ["train_number", "journey_date"]
    ].drop_duplicates().shape[0]
)

print("\nStation records:")
print(len(df))

print("\nDate range:")

print(
    df["journey_date"].min().date(),
    "to",
    df["journey_date"].max().date()
)


# ============================================================
# EDA DELAY FILTER
# ============================================================
#
# We found obvious timestamp anomalies in the raw historical
# data, producing impossible-looking delay values such as
# 51,731 minutes.
#
# These observations are excluded ONLY from EDA.
#
# The master historical dataset is NOT modified.
#
# The upper threshold is 600 minutes (10 hours), which is
# deliberately generous for train delays while excluding the
# clearly corrupted multi-day timestamp values.
# ============================================================

EDA_MAX_DELAY = 600

arrival_df_raw = df[
    df["arrival_delay"].notna()
].copy()

arrival_df = arrival_df_raw[
    arrival_df_raw["arrival_delay"].between(
        -60,
        EDA_MAX_DELAY
    )
].copy()

excluded_count = (
    len(arrival_df_raw)
    - len(arrival_df)
)

print("\n" + "=" * 60)
print("EDA DELAY QUALITY CHECK")
print("=" * 60)

print(
    "\nArrival-delay observations before filtering:",
    len(arrival_df_raw)
)

print(
    "Arrival-delay observations used for EDA:",
    len(arrival_df)
)

print(
    "Excluded suspicious observations:",
    excluded_count
)

print(
    "\nEDA delay range:",
    f"{arrival_df['arrival_delay'].min():.0f}",
    "to",
    f"{arrival_df['arrival_delay'].max():.0f}",
    "minutes"
)


# ============================================================
# 1. DELAY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("ARRIVAL DELAY DISTRIBUTION")
print("=" * 60)

print(
    arrival_df["arrival_delay"].describe()
)


plt.figure(figsize=(10, 6))

plt.hist(
    arrival_df["arrival_delay"],
    bins=30
)

plt.axvline(
    arrival_df["arrival_delay"].mean(),
    linestyle="--",
    label="Mean delay"
)

plt.xlabel("Arrival delay (minutes)")
plt.ylabel("Number of station records")
plt.title("Distribution of Train Arrival Delays")
plt.legend()

plt.tight_layout()

delay_distribution_file = os.path.join(
    EDA_DIR,
    "delay_distribution.png"
)

plt.savefig(
    delay_distribution_file,
    dpi=150
)

plt.close()

print(
    "\nSaved:",
    delay_distribution_file
)


# ============================================================
# 2. AVERAGE DELAY BY TRAIN
# ============================================================

train_delay = (
    arrival_df
    .groupby("train_number")["arrival_delay"]
    .agg(
        average_delay="mean",
        median_delay="median",
        station_records="count"
    )
    .sort_values(
        "average_delay",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("ARRIVAL DELAY BY TRAIN")
print("=" * 60)

print(
    train_delay.round(2)
)


plt.figure(figsize=(10, 6))

plt.bar(
    train_delay.index,
    train_delay["average_delay"]
)

plt.xlabel("Train number")
plt.ylabel("Average arrival delay (minutes)")
plt.title("Average Arrival Delay by Train")

plt.tight_layout()

train_delay_file = os.path.join(
    EDA_DIR,
    "average_delay_by_train.png"
)

plt.savefig(
    train_delay_file,
    dpi=150
)

plt.close()

print(
    "\nSaved:",
    train_delay_file
)


# ============================================================
# 3. DELAY BY STATION
# ============================================================

station_delay = (
    arrival_df
    .groupby(
        [
            "station_code",
            "station_name"
        ]
    )["arrival_delay"]
    .agg(
        average_delay="mean",
        median_delay="median",
        observations="count"
    )
)

station_delay_filtered = (
    station_delay[
        station_delay["observations"] >= 3
    ]
    .sort_values(
        "average_delay",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("STATIONS WITH HIGHEST AVERAGE ARRIVAL DELAY")
print("=" * 60)

print(
    station_delay_filtered
    .head(15)
    .round(2)
)


top_station_delay = (
    station_delay_filtered
    .head(15)
    .sort_values(
        "average_delay"
    )
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_station_delay.index.get_level_values(
        "station_code"
    ),
    top_station_delay["average_delay"]
)

plt.xlabel("Average arrival delay (minutes)")
plt.ylabel("Station code")
plt.title(
    "Stations with Highest Average Arrival Delay"
)

plt.tight_layout()

station_delay_file = os.path.join(
    EDA_DIR,
    "delay_by_station.png"
)

plt.savefig(
    station_delay_file,
    dpi=150
)

plt.close()

print(
    "\nSaved:",
    station_delay_file
)


# ============================================================
# 4. DELAY PROPAGATION
# ============================================================

propagation_df = df[
    df["departure_delay"].notna()
    & df["arrival_delay"].notna()
].copy()

# Remove obviously corrupted values from both sides.

propagation_df = propagation_df[
    propagation_df["departure_delay"].between(
        -60,
        EDA_MAX_DELAY
    )
    &
    propagation_df["arrival_delay"].between(
        -60,
        EDA_MAX_DELAY
    )
].copy()

propagation_df = propagation_df.sort_values(
    [
        "train_number",
        "journey_date",
        "station_sequence"
    ]
)

propagation_df["next_arrival_delay"] = (
    propagation_df
    .groupby(
        [
            "train_number",
            "journey_date"
        ]
    )["arrival_delay"]
    .shift(-1)
)

propagation_df = propagation_df[
    propagation_df["next_arrival_delay"].notna()
].copy()

propagation_df = propagation_df[
    propagation_df["next_arrival_delay"].between(
        -60,
        EDA_MAX_DELAY
    )
].copy()

propagation_df["delay_change"] = (
    propagation_df["next_arrival_delay"]
    - propagation_df["departure_delay"]
)


print("\n" + "=" * 60)
print("DELAY PROPAGATION")
print("=" * 60)

print(
    "\nUsable consecutive-station records:",
    len(propagation_df)
)

print(
    "\nAverage change in delay:",
    f"{propagation_df['delay_change'].mean():.2f}",
    "minutes"
)

print(
    "\nMedian change:",
    f"{propagation_df['delay_change'].median():.2f}",
    "minutes"
)

correlation = propagation_df[
    [
        "departure_delay",
        "next_arrival_delay"
    ]
].corr().iloc[0, 1]

print(
    "\nCorrelation between current departure delay"
    "\nand next station arrival delay:"
)

print(
    f"{correlation:.4f}"
)


plt.figure(figsize=(10, 6))

plt.scatter(
    propagation_df["departure_delay"],
    propagation_df["next_arrival_delay"],
    alpha=0.35
)

plt.xlabel(
    "Current station departure delay (minutes)"
)

plt.ylabel(
    "Next station arrival delay (minutes)"
)

plt.title(
    "Delay Propagation Between Consecutive Stations"
)

plt.tight_layout()

propagation_file = os.path.join(
    EDA_DIR,
    "delay_propagation.png"
)

plt.savefig(
    propagation_file,
    dpi=150
)

plt.close()

print(
    "\nSaved:",
    propagation_file
)


# ============================================================
# 5. DAY OF WEEK
# ============================================================

arrival_df["day_of_week"] = (
    arrival_df["journey_date"].dt.day_name()
)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_delay = (
    arrival_df
    .groupby("day_of_week")["arrival_delay"]
    .agg(
        average_delay="mean",
        median_delay="median",
        observations="count"
    )
    .reindex(day_order)
)

print("\n" + "=" * 60)
print("ARRIVAL DELAY BY DAY OF WEEK")
print("=" * 60)

print(
    day_delay.round(2)
)


plt.figure(figsize=(10, 6))

plt.bar(
    day_delay.index,
    day_delay["average_delay"]
)

plt.xlabel("Day of week")
plt.ylabel("Average arrival delay (minutes)")
plt.title("Average Arrival Delay by Day of Week")

plt.xticks(rotation=30)

plt.tight_layout()

day_delay_file = os.path.join(
    EDA_DIR,
    "delay_by_day_of_week.png"
)

plt.savefig(
    day_delay_file,
    dpi=150
)

plt.close()

print(
    "\nSaved:",
    day_delay_file
)


# ============================================================
# 6. JOURNEY COVERAGE
# ============================================================

journeys_by_train = (
    df[
        [
            "train_number",
            "journey_date"
        ]
    ]
    .drop_duplicates()
    .groupby("train_number")
    .size()
    .sort_values(
        ascending=False
    )
)

print("\n" + "=" * 60)
print("HISTORICAL JOURNEYS BY TRAIN")
print("=" * 60)

print(
    journeys_by_train
)


plt.figure(figsize=(10, 6))

plt.bar(
    journeys_by_train.index,
    journeys_by_train.values
)

plt.xlabel("Train number")
plt.ylabel("Number of historical journeys")
plt.title("Historical Journey Coverage by Train")

plt.tight_layout()

journey_count_file = os.path.join(
    EDA_DIR,
    "journeys_by_train.png"
)

plt.savefig(
    journey_count_file,
    dpi=150
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print("\nGenerated files:")

for filename in [
    "delay_distribution.png",
    "average_delay_by_train.png",
    "delay_by_station.png",
    "delay_propagation.png",
    "delay_by_day_of_week.png",
    "journeys_by_train.png"
]:

    print(
        "-",
        os.path.join(
            EDA_DIR,
            filename
        )
    )

print("\n" + "=" * 60)