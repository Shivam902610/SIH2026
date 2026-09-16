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
print("ENHANCED MODEL - LEAVE-ONE-JOURNEY-OUT EVALUATION")
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

print(
    "Journeys:",
    df["journey_date"]
    .nunique()
)


# ============================================================
# SORT JOURNEYS
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

journeys = (
    df[
        [
            "journey_key",
            "journey_date"
        ]
    ]
    .drop_duplicates()
    .sort_values("journey_date")
    .reset_index(drop=True)
)

print(
    "\nTotal journeys:",
    len(journeys)
)


# ============================================================
# STORAGE FOR PREDICTIONS
# ============================================================

linear_predictions = []
random_forest_predictions = []
actual_values = []
baseline_predictions = []

journey_results = []


# ============================================================
# LEAVE-ONE-JOURNEY-OUT
# ============================================================

for index, journey_info in enumerate(
    journeys.itertuples(index=False),
    start=1
):

    test_journey_key = (
        journey_info.journey_key
    )

    test_date = (
        journey_info.journey_date
    )


    print(
        f"\rEvaluating journey "
        f"{index}/{len(journeys)}: "
        f"{test_journey_key}",
        end=""
    )


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------
    #
    # The complete journey is held out.
    #
    # This prevents rows from the same journey appearing
    # in both training and testing.
    #
    # --------------------------------------------------------

    train_df = df[
        df["journey_key"]
        != test_journey_key
    ].copy()

    test_df = df[
        df["journey_key"]
        == test_journey_key
    ].copy()


    if len(test_df) == 0:
        continue


    # --------------------------------------------------------
    # FEATURES
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


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

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


    # ========================================================
    # LINEAR REGRESSION
    # ========================================================

    linear_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
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


    linear_pred = linear_model.predict(
        X_test
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_preprocessor = ColumnTransformer(
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


    rf_pred = rf_model.predict(
        X_test
    )


    # ========================================================
    # NAIVE BASELINE
    # ========================================================
    #
    # Simplest prediction:
    #
    # "The delay at the current station will remain the same
    # at the next station."
    #
    # ========================================================

    baseline_pred = (
        test_df[
            "current_departure_delay"
        ]
        .values
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    actual_values.extend(
        y_test.values
    )

    linear_predictions.extend(
        linear_pred
    )

    random_forest_predictions.extend(
        rf_pred
    )

    baseline_predictions.extend(
        baseline_pred
    )


    # --------------------------------------------------------
    # JOURNEY METRICS
    # --------------------------------------------------------

    linear_mae = mean_absolute_error(
        y_test,
        linear_pred
    )

    rf_mae = mean_absolute_error(
        y_test,
        rf_pred
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )


    journey_results.append(
        {
            "journey": test_journey_key,
            "train_number": test_df[
                "train_number"
            ].iloc[0],
            "date": test_date.strftime(
                "%Y-%m-%d"
            ),
            "rows": len(test_df),
            "linear_mae": linear_mae,
            "rf_mae": rf_mae,
            "baseline_mae": baseline_mae
        }
    )


print("\n")


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

actual_values = np.array(
    actual_values
)

linear_predictions = np.array(
    linear_predictions
)

random_forest_predictions = np.array(
    random_forest_predictions
)

baseline_predictions = np.array(
    baseline_predictions
)


# ============================================================
# METRICS FUNCTION
# ============================================================

def print_metrics(
    name,
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


    print(
        f"\n{name}"
    )

    print(
        "-" * 40
    )

    print(
        f"MAE  : {mae:.2f} minutes"
    )

    print(
        f"RMSE : {rmse:.2f} minutes"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return mae, rmse, r2


# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)


linear_metrics = print_metrics(
    "Linear Regression",
    actual_values,
    linear_predictions
)


rf_metrics = print_metrics(
    "Random Forest",
    actual_values,
    random_forest_predictions
)


baseline_metrics = print_metrics(
    "Naive Delay Propagation",
    actual_values,
    baseline_predictions
)


# ============================================================
# IMPROVEMENT
# ============================================================

linear_improvement = (
    baseline_metrics[0]
    - linear_metrics[0]
)

rf_improvement = (
    baseline_metrics[0]
    - rf_metrics[0]
)


print(
    "\n" + "=" * 60
)

print(
    "IMPROVEMENT OVER BASELINE"
)

print(
    "=" * 60
)


print(
    f"\nLinear Regression:"
)

print(
    f"Baseline MAE : "
    f"{baseline_metrics[0]:.2f}"
)

print(
    f"Model MAE    : "
    f"{linear_metrics[0]:.2f}"
)

print(
    f"Difference   : "
    f"{linear_improvement:.2f} minutes"
)


print(
    f"\nRandom Forest:"
)

print(
    f"Baseline MAE : "
    f"{baseline_metrics[0]:.2f}"
)

print(
    f"Model MAE    : "
    f"{rf_metrics[0]:.2f}"
)

print(
    f"Difference   : "
    f"{rf_improvement:.2f} minutes"
)


# ============================================================
# PER-JOURNEY RESULTS
# ============================================================

results_df = pd.DataFrame(
    journey_results
)


print(
    "\n" + "=" * 60
)

print(
    "PER-JOURNEY RESULTS"
)

print(
    "=" * 60
)


print(
    results_df[
        [
            "journey",
            "rows",
            "linear_mae",
            "rf_mae",
            "baseline_mae"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "enhanced_evaluation_results.csv"
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL SUMMARY"
)

print(
    "=" * 60
)

print(
    f"\nTotal test journeys: "
    f"{len(journeys)}"
)

print(
    f"Total test samples: "
    f"{len(actual_values)}"
)

print(
    f"\nLinear Regression MAE: "
    f"{linear_metrics[0]:.2f} minutes"
)

print(
    f"Random Forest MAE: "
    f"{rf_metrics[0]:.2f} minutes"
)

print(
    f"Naive Baseline MAE: "
    f"{baseline_metrics[0]:.2f} minutes"
)

print(
    "\nResults saved to:"
)

print(
    RESULT_FILE
)

print(
    "\n" + "=" * 60
)