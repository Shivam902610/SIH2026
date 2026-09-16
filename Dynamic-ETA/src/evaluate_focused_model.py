import os
import numpy as np
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


# ============================================================
# FOCUSED FEATURES
# ============================================================
#
# These are the features that have a direct relationship with
# the next station's arrival delay.
#
# We intentionally remove:
#
# - historical_station_delay
# - historical_train_delay
# - route_progress
# - stations_remaining
# - scheduled_departure_hour
# - day_of_week
#
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
    "scheduled_travel_time"
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
print("FOCUSED MODEL - LEAVE-ONE-JOURNEY-OUT EVALUATION")
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
# PREPARE JOURNEY DATE
# ============================================================

df["journey_date"] = pd.to_datetime(
    df["journey_date"]
)

df["journey_key"] = (
    df["train_number"].astype(str)
    + "_"
    + df["journey_date"]
    .dt.strftime("%Y-%m-%d")
)


# ============================================================
# JOURNEYS
# ============================================================

journeys = (
    df[
        [
            "journey_key",
            "journey_date"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "journey_date"
    )
    .reset_index(drop=True)
)


print(
    "Total journeys:",
    len(journeys)
)


# ============================================================
# STORAGE
# ============================================================

actual_values = []

linear_predictions = []

rf_predictions = []

baseline_predictions = []


# ============================================================
# LEAVE-ONE-JOURNEY-OUT
# ============================================================

for index, journey_info in enumerate(
    journeys.itertuples(index=False),
    start=1
):

    test_journey = (
        journey_info.journey_key
    )


    print(
        f"\rEvaluating journey "
        f"{index}/{len(journeys)}: "
        f"{test_journey}",
        end=""
    )


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_df = df[
        df["journey_key"]
        != test_journey
    ]

    test_df = df[
        df["journey_key"]
        == test_journey
    ]


    if test_df.empty:
        continue


    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ]


    # ========================================================
    # LINEAR REGRESSION
    # ========================================================

    linear_preprocessor = (
        ColumnTransformer(
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
    )


    linear_model = Pipeline(
        steps=[
            (
                "preprocessor",
                linear_preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )


    linear_model.fit(
        X_train,
        y_train
    )


    linear_pred = (
        linear_model.predict(
            X_test
        )
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_preprocessor = (
        ColumnTransformer(
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
    )


    rf_model = Pipeline(
        steps=[
            (
                "preprocessor",
                rf_preprocessor
            ),
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


    rf_model.fit(
        X_train,
        y_train
    )


    rf_pred = (
        rf_model.predict(
            X_test
        )
    )


    # ========================================================
    # NAIVE BASELINE
    # ========================================================

    baseline_pred = (
        test_df[
            "current_departure_delay"
        ].values
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    actual_values.extend(
        y_test.values
    )

    linear_predictions.extend(
        linear_pred
    )

    rf_predictions.extend(
        rf_pred
    )

    baseline_predictions.extend(
        baseline_pred
    )


print("\n")


# ============================================================
# ARRAYS
# ============================================================

actual_values = np.array(
    actual_values
)

linear_predictions = np.array(
    linear_predictions
)

rf_predictions = np.array(
    rf_predictions
)

baseline_predictions = np.array(
    baseline_predictions
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


linear_metrics = calculate_metrics(
    actual_values,
    linear_predictions
)

rf_metrics = calculate_metrics(
    actual_values,
    rf_predictions
)

baseline_metrics = calculate_metrics(
    actual_values,
    baseline_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("FOCUSED MODEL RESULTS")
print("=" * 60)


print(
    "\nLinear Regression"
)

print(
    f"MAE  : {linear_metrics[0]:.2f} minutes"
)

print(
    f"RMSE : {linear_metrics[1]:.2f} minutes"
)

print(
    f"R²   : {linear_metrics[2]:.4f}"
)


print(
    "\nRandom Forest"
)

print(
    f"MAE  : {rf_metrics[0]:.2f} minutes"
)

print(
    f"RMSE : {rf_metrics[1]:.2f} minutes"
)

print(
    f"R²   : {rf_metrics[2]:.4f}"
)


print(
    "\nNaive Delay Propagation"
)

print(
    f"MAE  : {baseline_metrics[0]:.2f} minutes"
)

print(
    f"RMSE : {baseline_metrics[1]:.2f} minutes"
)

print(
    f"R²   : {baseline_metrics[2]:.4f}"
)


# ============================================================
# COMPARISON
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "COMPARISON"
)

print(
    "=" * 60
)


print(
    f"\nCurrent working model:"
)

print(
    "Linear Regression MAE = 5.94 minutes"
)


print(
    f"\nFocused Linear Regression:"
)

print(
    f"MAE = {linear_metrics[0]:.2f} minutes"
)


print(
    f"\nFocused Random Forest:"
)

print(
    f"MAE = {rf_metrics[0]:.2f} minutes"
)


print(
    "\nImprovement over naive baseline:"
)

print(
    f"Focused Linear: "
    f"{baseline_metrics[0] - linear_metrics[0]:.2f} minutes"
)

print(
    f"Focused Random Forest: "
    f"{baseline_metrics[0] - rf_metrics[0]:.2f} minutes"
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EVALUATION COMPLETE"
)

print(
    "=" * 60
)

print(
    "\nNo existing model files were modified."
)