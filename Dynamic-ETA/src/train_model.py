import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/dataset/ml_dataset.csv"
MODEL_DIR = "models"

LINEAR_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "linear_regression_eta.pkl"
)

RANDOM_FOREST_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "random_forest_eta.pkl"
)

BEST_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_eta_model.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("DYNAMIC ETA - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print()
print("ML dataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# FEATURES AND TARGET
# ============================================================

target_column = "target_arrival_delay"


features = [
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

X = df[features]
y = df[target_column]


print()
print("Features:")
print(features)

print()
print("Target:")
print(target_column)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "train_number",
    "current_station",
    "next_station"
]


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_features = [
    "station_sequence",
    "current_departure_delay",
    "previous_departure_delay",
    "next_station_sequence",
    "scheduled_travel_time",
    "day_of_week"
]


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
            numerical_features
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# FUNCTION TO EVALUATE MODEL
# ============================================================

def evaluate_model(model, model_name):

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"MAE  : {mae:.2f} minutes")
    print(f"RMSE : {rmse:.2f} minutes")
    print(f"R²   : {r2:.4f}")

    return model, mae, rmse, r2


# ============================================================
# MODEL 1 - LINEAR REGRESSION
# ============================================================

linear_pipeline = Pipeline(

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


linear_model, linear_mae, linear_rmse, linear_r2 = evaluate_model(
    linear_pipeline,
    "MODEL 1 - LINEAR REGRESSION"
)


# ============================================================
# MODEL 2 - RANDOM FOREST
# ============================================================

random_forest_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
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


random_forest_model, rf_mae, rf_rmse, rf_r2 = evaluate_model(
    random_forest_pipeline,
    "MODEL 2 - RANDOM FOREST"
)


# ============================================================
# COMPARE MODELS
# ============================================================

print()
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print()
print(
    f"Linear Regression MAE : "
    f"{linear_mae:.2f} minutes"
)

print(
    f"Random Forest MAE     : "
    f"{rf_mae:.2f} minutes"
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

if rf_mae < linear_mae:

    best_model = random_forest_model
    best_model_name = "Random Forest"
    best_mae = rf_mae

else:

    best_model = linear_model
    best_model_name = "Linear Regression"
    best_mae = linear_mae


print()
print("Best model:", best_model_name)
print(f"Best MAE: {best_mae:.2f} minutes")


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    linear_model,
    LINEAR_MODEL_FILE
)

joblib.dump(
    random_forest_model,
    RANDOM_FOREST_MODEL_FILE
)

joblib.dump(
    best_model,
    BEST_MODEL_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("MODELS SAVED")
print("=" * 60)

print()
print("Linear Regression:")
print(LINEAR_MODEL_FILE)

print()
print("Random Forest:")
print(RANDOM_FOREST_MODEL_FILE)

print()
print("Best model:")
print(BEST_MODEL_FILE)

print()
print("=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)