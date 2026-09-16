import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "ml_dataset_enhanced.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "feature_importance.csv"
)


# ============================================================
# FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "train_number",
    "current_station",
    "next_station"
]

NUMERICAL_FEATURES = [
    "station_sequence",
    "current_departure_delay",
    "previous_departure_delay",
    "delay_change",
    "next_station_sequence",
    "scheduled_travel_time",
    "route_progress",
    "stations_remaining",
    "historical_station_delay",
    "historical_train_delay",
    "scheduled_departure_hour",
    "day_of_week"
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)

TARGET = "target_arrival_delay"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

df = pd.read_csv(
    DATA_FILE
)

print(
    "\nDataset loaded."
)

print(
    "Rows:",
    len(df)
)


# ============================================================
# PREPARE X / Y
# ============================================================

X = df[FEATURES]

y = df[TARGET]


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES
        )
    ],
    remainder="passthrough"
)


X_encoded = preprocessor.fit_transform(
    X
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print(
    "\nTraining Random Forest..."
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_encoded,
    y
)


print(
    "Random Forest trained."
)


# ============================================================
# GET FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importances = (
    model.feature_importances_
)


importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances
    }
)


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# GROUP ONE-HOT FEATURES
# ============================================================
#
# A categorical feature such as train_number creates several
# encoded columns.
#
# For the hackathon, we also want to know the total importance
# of the original feature.
#
# ============================================================

grouped_importance = []


for feature in FEATURES:

    if feature in CATEGORICAL_FEATURES:

        matching = importance_df[
            importance_df["feature"]
            .str.contains(
                f"__{feature}_",
                regex=False
            )
        ]

        total_importance = (
            matching["importance"].sum()
        )

    else:

        matching = importance_df[
            importance_df["feature"]
            .str.endswith(
                f"__{feature}"
            )
        ]

        total_importance = (
            matching["importance"].sum()
        )

    grouped_importance.append(
        {
            "feature": feature,
            "importance": total_importance
        }
    )


grouped_df = pd.DataFrame(
    grouped_importance
)


grouped_df = (
    grouped_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# PRINT INDIVIDUAL ENCODED FEATURES
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TOP ENCODED FEATURES"
)

print(
    "=" * 60
)

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# PRINT ORIGINAL FEATURE IMPORTANCE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "GROUPED FEATURE IMPORTANCE"
)

print(
    "=" * 60
)

print(
    grouped_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

grouped_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n" + "=" * 60
)

print(
    "FEATURE IMPORTANCE ANALYSIS COMPLETE"
)

print(
    "=" * 60
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)