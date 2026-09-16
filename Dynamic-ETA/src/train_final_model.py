import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "focused_eta_model.pkl"
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
    "scheduled_travel_time"
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)

TARGET = "target_arrival_delay"


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TRAINING FINAL ETA MODEL")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

print("\nDataset loaded.")

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "Journeys:",
    df["journey_date"].nunique()
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].copy()


print(
    "\nTraining features:"
)

for feature in FEATURES:
    print(
        f"  - {feature}"
    )


# ============================================================
# PREPROCESSOR
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


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
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


# ============================================================
# TRAIN
# ============================================================

print(
    "\nTraining Linear Regression..."
)

model.fit(
    X,
    y
)

print(
    "Training complete."
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# VERIFY
# ============================================================

print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)


print(
    "\nModel file exists:",
    os.path.exists(
        MODEL_FILE
    )
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL MODEL INFORMATION"
)

print(
    "=" * 60
)

print(
    "\nModel type:"
)

print(
    "Linear Regression"
)


print(
    "\nTraining samples:"
)

print(
    len(X)
)


print(
    "\nTarget:"
)

print(
    TARGET
)


print(
    "\nFeatures:"
)

for feature in FEATURES:
    print(
        f"  {feature}"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL ETA MODEL READY"
)

print(
    "=" * 60
)

print(
    "\nExisting best_eta_model.pkl was NOT modified."
)