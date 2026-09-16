import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "ml_dataset_enhanced.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dynamic_delay_change_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING ENHANCED ML DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"Rows loaded: {len(df)}")


# ============================================================
# CREATE DYNAMIC TARGET
# ============================================================

# Instead of predicting the complete delay at the next station,
# predict how much the delay is expected to CHANGE.
#
# Example:
#
# Current delay = 247 min
# Historical/current conditions suggest +3 min change
#
# Predicted next delay = 250 min

df["target_delay_change"] = (
    df["target_arrival_delay"]
    - df["current_departure_delay"]
)


# ============================================================
# FEATURES
# ============================================================

categorical_features = [
    "train_number",
    "current_station",
    "next_station"
]

numeric_features = [
    "station_sequence",
    "current_departure_delay",
    "previous_departure_delay",
    "delay_change",
    "next_station_sequence",
    "scheduled_travel_time"
]

features = categorical_features + numeric_features

target = "target_delay_change"


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=features + [target]
).copy()

print(f"Usable rows: {len(df)}")

print()
print("Target: delay change")
print(df[target].describe())


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

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


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# LEAVE-ONE-JOURNEY-OUT EVALUATION
# ============================================================

print()
print("=" * 60)
print("LEAVE-ONE-JOURNEY-OUT EVALUATION")
print("=" * 60)

journeys = (
    df[
        [
            "train_number",
            "journey_date"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["journey_date", "train_number"]
    )
)


linear_predictions = []
rf_predictions = []
actual_values = []

for _, journey in journeys.iterrows():

    train_number = journey["train_number"]
    journey_date = journey["journey_date"]

    test_mask = (
        (df["train_number"] == train_number)
        &
        (df["journey_date"] == journey_date)
    )

    train_df = df[
        ~test_mask
    ]

    test_df = df[
        test_mask
    ]

    if len(test_df) == 0:
        continue

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # Linear Regression
    linear_model.fit(
        X_train,
        y_train
    )

    linear_pred = linear_model.predict(
        X_test
    )

    # Random Forest
    random_forest_model.fit(
        X_train,
        y_train
    )

    rf_pred = random_forest_model.predict(
        X_test
    )

    linear_predictions.extend(
        linear_pred
    )

    rf_predictions.extend(
        rf_pred
    )

    actual_values.extend(
        y_test
    )


# ============================================================
# METRICS
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


linear_mae = mean_absolute_error(
    actual_values,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        linear_predictions
    )
)

linear_r2 = r2_score(
    actual_values,
    linear_predictions
)


rf_mae = mean_absolute_error(
    actual_values,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        rf_predictions
    )
)

rf_r2 = r2_score(
    actual_values,
    rf_predictions
)


print()
print("Linear Regression")
print("-" * 40)
print(f"MAE  : {linear_mae:.2f} min")
print(f"RMSE : {linear_rmse:.2f} min")
print(f"R²   : {linear_r2:.4f}")


print()
print("Random Forest")
print("-" * 40)
print(f"MAE  : {rf_mae:.2f} min")
print(f"RMSE : {rf_rmse:.2f} min")
print(f"R²   : {rf_r2:.4f}")


# ============================================================
# SELECT MODEL
# ============================================================

if linear_mae <= rf_mae:

    best_model = linear_model

    print()
    print("Selected model: Linear Regression")

else:

    best_model = random_forest_model

    print()
    print("Selected model: Random Forest")


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================

print()
print("=" * 60)
print("TRAINING FINAL DYNAMIC MODEL")
print("=" * 60)

X = df[features]
y = df[target]

best_model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)


print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Path: {MODEL_PATH}"
)

print()
print("Features:")
for feature in features:
    print(f"  - {feature}")

print()
print("Target:")
print("  target_delay_change")

print()
print("Example:")
print(
    "Current delay + predicted delay change = predicted next-station delay"
)

print("=" * 60)