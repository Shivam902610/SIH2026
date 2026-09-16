import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATH
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
print("LEAVE-ONE-JOURNEY-OUT EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nML dataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# FEATURES / TARGET
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


# ============================================================
# CREATE JOURNEY ID
# ============================================================

df["journey_id"] = (
    df["train_number"].astype(str)
    + "_"
    + df["journey_date"].astype(str)
)

journeys = df["journey_id"].unique()

print("\nTotal usable journeys:", len(journeys))


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

linear_predictions_all = []
rf_predictions_all = []
actual_values_all = []
baseline_predictions_all = []

journey_results = []


# ============================================================
# LEAVE-ONE-JOURNEY-OUT
# ============================================================

for i, test_journey in enumerate(journeys):

    print(
        f"\nEvaluating journey {i + 1}/{len(journeys)}: "
        f"{test_journey}"
    )

    train_df = df[df["journey_id"] != test_journey].copy()
    test_df = df[df["journey_id"] == test_journey].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]


    # ========================================================
    # PREPROCESSOR
    # ========================================================

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

    linear_model.fit(X_train, y_train)

    linear_predictions = linear_model.predict(X_test)


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_preprocessor = ColumnTransformer(
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

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)


    # ========================================================
    # BASELINE
    # ========================================================

    baseline_predictions = test_df[
        "current_departure_delay"
    ].values


    # ========================================================
    # STORE PREDICTIONS
    # ========================================================

    actual_values_all.extend(
        y_test.values
    )

    linear_predictions_all.extend(
        linear_predictions
    )

    rf_predictions_all.extend(
        rf_predictions
    )

    baseline_predictions_all.extend(
        baseline_predictions
    )


    # ========================================================
    # JOURNEY METRICS
    # ========================================================

    linear_journey_mae = mean_absolute_error(
        y_test,
        linear_predictions
    )

    rf_journey_mae = mean_absolute_error(
        y_test,
        rf_predictions
    )

    baseline_journey_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    journey_results.append(
        {
            "journey_id": test_journey,
            "rows": len(test_df),
            "linear_mae": linear_journey_mae,
            "random_forest_mae": rf_journey_mae,
            "baseline_mae": baseline_journey_mae
        }
    )


# ============================================================
# CONVERT RESULTS
# ============================================================

actual_values_all = pd.Series(
    actual_values_all
)

linear_predictions_all = pd.Series(
    linear_predictions_all
)

rf_predictions_all = pd.Series(
    rf_predictions_all
)

baseline_predictions_all = pd.Series(
    baseline_predictions_all
)

journey_results_df = pd.DataFrame(
    journey_results
)


# ============================================================
# OVERALL METRICS
# ============================================================

linear_mae = mean_absolute_error(
    actual_values_all,
    linear_predictions_all
)

linear_rmse = mean_squared_error(
    actual_values_all,
    linear_predictions_all
) ** 0.5

linear_r2 = r2_score(
    actual_values_all,
    linear_predictions_all
)


rf_mae = mean_absolute_error(
    actual_values_all,
    rf_predictions_all
)

rf_rmse = mean_squared_error(
    actual_values_all,
    rf_predictions_all
) ** 0.5

rf_r2 = r2_score(
    actual_values_all,
    rf_predictions_all
)


baseline_mae = mean_absolute_error(
    actual_values_all,
    baseline_predictions_all
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("OVERALL LEAVE-ONE-JOURNEY-OUT RESULTS")
print("=" * 60)

print("\nLinear Regression")
print(f"MAE  : {linear_mae:.2f} minutes")
print(f"RMSE : {linear_rmse:.2f} minutes")
print(f"R²   : {linear_r2:.4f}")

print("\nRandom Forest")
print(f"MAE  : {rf_mae:.2f} minutes")
print(f"RMSE : {rf_rmse:.2f} minutes")
print(f"R²   : {rf_r2:.4f}")

print("\nNaive Delay Propagation Baseline")
print(f"MAE  : {baseline_mae:.2f} minutes")


# ============================================================
# AVERAGE JOURNEY MAE
# ============================================================

average_linear_journey_mae = (
    journey_results_df["linear_mae"].mean()
)

average_rf_journey_mae = (
    journey_results_df["random_forest_mae"].mean()
)

average_baseline_journey_mae = (
    journey_results_df["baseline_mae"].mean()
)

print("\n" + "=" * 60)
print("AVERAGE ERROR PER JOURNEY")
print("=" * 60)

print(
    f"\nLinear Regression       : "
    f"{average_linear_journey_mae:.2f} minutes"
)

print(
    f"Random Forest           : "
    f"{average_rf_journey_mae:.2f} minutes"
)

print(
    f"Naive Baseline          : "
    f"{average_baseline_journey_mae:.2f} minutes"
)


# ============================================================
# BEST / WORST JOURNEYS
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION JOURNEY RESULTS")
print("=" * 60)

sorted_results = journey_results_df.sort_values(
    "linear_mae"
)

print("\nLowest MAE journeys:")

print(
    sorted_results[
        [
            "journey_id",
            "rows",
            "linear_mae",
            "random_forest_mae",
            "baseline_mae"
        ]
    ].head(5).to_string(index=False)
)

print("\nHighest MAE journeys:")

print(
    sorted_results[
        [
            "journey_id",
            "rows",
            "linear_mae",
            "random_forest_mae",
            "baseline_mae"
        ]
    ].tail(5).to_string(index=False)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"\nUsable journeys evaluated : {len(journeys)}"
)

print(
    f"Total test rows            : "
    f"{len(actual_values_all)}"
)

print(
    f"\nLinear Regression MAE      : "
    f"{linear_mae:.2f} minutes"
)

print(
    f"Random Forest MAE          : "
    f"{rf_mae:.2f} minutes"
)

print(
    f"Naive Baseline MAE         : "
    f"{baseline_mae:.2f} minutes"
)

improvement = baseline_mae - linear_mae

print(
    f"\nLinear vs baseline         : "
    f"{improvement:.2f} minutes MAE difference"
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)