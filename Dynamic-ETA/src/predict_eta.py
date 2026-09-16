import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/best_eta_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print("=" * 60)
print("DYNAMIC ETA - PREDICTION TEST")
print("=" * 60)


# ============================================================
# EXAMPLE CURRENT TRAIN STATUS
# ============================================================

# We will use a real example from our historical dataset.

train_number = 12919

current_station = "INDB"

next_station = "DWX"

current_departure_delay = 13.0

previous_departure_delay = 4.0

station_sequence = 7

next_station_sequence = 13

scheduled_travel_time = 36.0

# September 12, 2026 was a Saturday
day_of_week = 5


# ============================================================
# CREATE INPUT
# ============================================================

input_data = pd.DataFrame({

    "train_number": [train_number],

    "current_station": [current_station],

    "next_station": [next_station],

    "station_sequence": [station_sequence],

    "current_departure_delay": [
        current_departure_delay
    ],

    "previous_departure_delay": [
        previous_departure_delay
    ],

    "next_station_sequence": [
        next_station_sequence
    ],

    "scheduled_travel_time": [
        scheduled_travel_time
    ],

    "day_of_week": [
        day_of_week
    ]
})


# ============================================================
# DISPLAY INPUT
# ============================================================

print()
print("Current train information:")
print()

print(f"Train: {train_number}")
print(f"Current station: {current_station}")
print(f"Next station: {next_station}")
print(
    f"Current departure delay: "
    f"{current_departure_delay} minutes"
)

print(
    f"Previous departure delay: "
    f"{previous_departure_delay} minutes"
)

print(
    f"Scheduled travel time: "
    f"{scheduled_travel_time} minutes"
)


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(input_data)

predicted_delay = prediction[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 60)
print("ETA PREDICTION")
print("=" * 60)

print()
print(
    f"Predicted arrival delay at "
    f"{next_station}: "
    f"{predicted_delay:.2f} minutes"
)

print()

print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)