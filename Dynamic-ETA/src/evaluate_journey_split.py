import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "ml_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("JOURNEY-LEVEL MODEL EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nML dataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# FEATURES AND TARGET
# ============================================================

FEATURES = [
    "train_number",
    "current_station",
    "next_station",
    "station_sequence",
    "current_departure_delay",
    "previous_departure_delay",
    "next_station_sequence",
    "scheduled_travel_time",
    "day_of_week"
]

TARGET = "target_arrival_delay"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# JOURNEY-LEVEL SPLIT
# ============================================================
#
# IMPORTANT:
# We split using complete journeys instead of individual rows.
#
# This prevents records from the same journey appearing in both
# training and testing data.
# ============================================================

journey_ids = (
    df["train_number"].astype(str)
    + "_"
    + df["journey_date"].astype(str)
)

unique_journeys = journey_ids.unique()

print("\nTotal journeys:", len(unique_journeys))

# Use the last 20% of journeys chronologically as test data.
#
# First sort journeys by train number and journey date so that
# the evaluation is closer to a real historical -> future setup.

journey_info = (
    df[["train_number", "journey_date"]]
    .drop_duplicates()
    .copy()
)

journey_info["train_number"] = journey_info["train_number"].astype(str)
journey_info["journey_date"] = pd.to_datetime(
    journey_info["journey_date"]
)

journey_info = journey_info.sort_values(
    ["journey_date", "train_number"]
)

journey_info["journey_id"] = (
    journey_info["train_number"]
    + "_"
    + journey_info["journey_date"].dt.strftime("%Y-%m-%d")
)

sorted_journeys = journey_info["journey_id"].tolist()

test_count = max(1, int(len(sorted_journeys) * 0.20))

test_journeys = sorted_journeys[-test_count:]
train_journeys = sorted_journeys[:-test_count]

train_mask = journey_ids.isin(train_journeys)
test_mask = journey_ids.isin(test_journeys)

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print("\nJourney-level split:")
print("Training journeys:", len(train_journeys))
print("Testing journeys :", len(test_journeys))

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))

print("\nTest journeys:")

for journey in test_journeys:
    print("-", journey)


# ============================================================
# PREPROCESSING
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
    "next_station_sequence",
    "scheduled_travel_time",
    "day_of_week"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            "passthrough",
            NUMERICAL_FEATURES
        )
    ]
)


# ============================================================
# MODEL 1 - LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1 - LINEAR REGRESSION")
print("=" * 60)

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print(f"MAE  : {linear_mae:.2f} minutes")
print(f"RMSE : {linear_rmse:.2f} minutes")
print(f"R²   : {linear_r2:.4f}")


# ============================================================
# MODEL 2 - RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2 - RANDOM FOREST")
print("=" * 60)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print(f"MAE  : {rf_mae:.2f} minutes")
print(f"RMSE : {rf_rmse:.2f} minutes")
print(f"R²   : {rf_r2:.4f}")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("JOURNEY-LEVEL MODEL COMPARISON")
print("=" * 60)

print(f"\nLinear Regression")
print(f"MAE  : {linear_mae:.2f} minutes")
print(f"RMSE : {linear_rmse:.2f} minutes")
print(f"R²   : {linear_r2:.4f}")

print(f"\nRandom Forest")
print(f"MAE  : {rf_mae:.2f} minutes")
print(f"RMSE : {rf_rmse:.2f} minutes")
print(f"R²   : {rf_r2:.4f}")


# ============================================================
# PER-TRAIN EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("TEST PERFORMANCE BY TRAIN")
print("=" * 60)

test_results = df[test_mask].copy()

test_results["linear_prediction"] = linear_predictions
test_results["rf_prediction"] = rf_predictions

for train_number in sorted(
    test_results["train_number"].unique()
):

    train_data = test_results[
        test_results["train_number"] == train_number
    ]

    actual = train_data[TARGET]

    linear_pred = train_data["linear_prediction"]

    rf_pred = train_data["rf_prediction"]

    train_linear_mae = mean_absolute_error(
        actual,
        linear_pred
    )

    train_rf_mae = mean_absolute_error(
        actual,
        rf_pred
    )

    journeys = train_data["journey_date"].nunique()

    print(
        f"\nTrain {train_number}"
    )

    print(
        f"Journeys tested : {journeys}"
    )

    print(
        f"Rows tested     : {len(train_data)}"
    )

    print(
        f"Linear MAE      : {train_linear_mae:.2f} minutes"
    )

    print(
        f"Random Forest MAE: {train_rf_mae:.2f} minutes"
    )


# ============================================================
# BASELINE COMPARISON
# ============================================================
#
# A very important baseline:
#
# "Next station delay = current station departure delay"
#
# This represents a simple non-ML approach.
# ============================================================

print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

baseline_predictions = test_results[
    "current_departure_delay"
]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print(
    f"\nNaive delay propagation baseline MAE:"
    f" {baseline_mae:.2f} minutes"
)

print(
    "\nThis baseline assumes the current delay simply"
    "\ncontinues unchanged to the next station."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION SUMMARY")
print("=" * 60)

print(
    f"\nTotal journeys       : {len(sorted_journeys)}"
)

print(
    f"Training journeys    : {len(train_journeys)}"
)

print(
    f"Testing journeys     : {len(test_journeys)}"
)

print(
    f"Training rows        : {len(X_train)}"
)

print(
    f"Testing rows         : {len(X_test)}"
)

print(
    f"\nLinear Regression MAE : {linear_mae:.2f} minutes"
)

print(
    f"Random Forest MAE     : {rf_mae:.2f} minutes"
)

print(
    f"Naive Baseline MAE    : {baseline_mae:.2f} minutes"
)

print("\n" + "=" * 60)
print("JOURNEY-LEVEL EVALUATION COMPLETE")
print("=" * 60)