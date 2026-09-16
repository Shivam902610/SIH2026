import json
import joblib
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

LIVE_FILE = "data/live_response.json"
MODEL_FILE = "models/best_eta_model.pkl"


# ============================================================
# LOAD LIVE TRAIN DATA
# ============================================================

with open(LIVE_FILE, "r", encoding="utf-8") as f:
    response_data = json.load(f)


# Check API response
if not response_data.get("success"):
    raise ValueError("Live train data is not successful.")


data = response_data["data"]

route = data["route"]


# ============================================================
# BASIC TRAIN INFORMATION
# ============================================================

train_number = str(data["trainNumber"])
train_name = data["trainName"]
journey_date = data["startDate"]

current_location = data["currentLocation"]
next_halt = data["nextHalt"]


# ============================================================
# CURRENT AND NEXT STATION
# ============================================================

current_sequence = current_location["sequence"]
next_sequence = next_halt["sequence"]

current_station = current_location["stationCode"]
next_station = next_halt["stationCode"]


# ============================================================
# FIND STATIONS IN ROUTE
# ============================================================

current_record = None
previous_record = None
next_record = None


for station in route:

    # Current station
    if station["sequence"] == current_sequence:
        current_record = station

    # Find the latest previous halt
    if (
        station["sequence"] < current_sequence
        and station.get("isHalt")
    ):
        if (
            previous_record is None
            or station["sequence"] > previous_record["sequence"]
        ):
            previous_record = station

    # Next station
    if station["sequence"] == next_sequence:
        next_record = station


# ============================================================
# SAFETY CHECKS
# ============================================================

if current_record is None:
    raise ValueError(
        f"Current station {current_station} "
        "was not found in route."
    )

if next_record is None:
    raise ValueError(
        f"Next station {next_station} "
        "was not found in route."
    )


# ============================================================
# CURRENT DEPARTURE DELAY
# ============================================================

current_departure_delay = current_record.get(
    "delayDeparture"
)

# If departure delay is unavailable,
# use current location delay
if current_departure_delay is None:
    current_departure_delay = current_location.get(
        "delayMinutes"
    )


# ============================================================
# PREVIOUS DEPARTURE DELAY
# ============================================================

previous_departure_delay = None

if previous_record is not None:
    previous_departure_delay = previous_record.get(
        "delayDeparture"
    )

# If previous delay is unavailable,
# use current delay as fallback
if previous_departure_delay is None:
    previous_departure_delay = current_departure_delay


# ============================================================
# SCHEDULED TRAVEL TIME
# ============================================================

scheduled_departure = current_record.get(
    "scheduledDeparture"
)

scheduled_arrival = next_record.get(
    "scheduledArrival"
)

if scheduled_departure is None:
    raise ValueError(
        "Scheduled departure time is missing."
    )

if scheduled_arrival is None:
    raise ValueError(
        "Scheduled arrival time for next station is missing."
    )


scheduled_departure = pd.to_datetime(
    scheduled_departure
)

scheduled_arrival = pd.to_datetime(
    scheduled_arrival
)

scheduled_travel_time = (
    scheduled_arrival - scheduled_departure
).total_seconds() / 60


# ============================================================
# DAY OF WEEK
# ============================================================

day_of_week = pd.to_datetime(
    journey_date
).dayofweek


# ============================================================
# DISPLAY LIVE TRAIN INFORMATION
# ============================================================

print("=" * 60)
print("DYNAMIC ETA - LIVE ML PREDICTION")
print("=" * 60)

print("\nLive train:")
print("Train:", train_number)
print("Train name:", train_name)
print("Journey date:", journey_date)

print("\nCurrent state:")
print("Current station:", current_station)
print("Next station:", next_station)
print(
    "Current departure delay:",
    current_departure_delay,
    "minutes"
)
print(
    "Previous departure delay:",
    previous_departure_delay,
    "minutes"
)
print(
    "Scheduled travel time:",
    scheduled_travel_time,
    "minutes"
)
print("Station sequence:", current_sequence)
print("Next station sequence:", next_sequence)
print("Day of week:", day_of_week)


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# CREATE INPUT FOR ML MODEL
# ============================================================

input_data = pd.DataFrame([
    {
        "train_number": train_number,
        "current_station": current_station,
        "next_station": next_station,
        "station_sequence": current_sequence,
        "current_departure_delay": current_departure_delay,
        "previous_departure_delay": previous_departure_delay,
        "next_station_sequence": next_sequence,
        "scheduled_travel_time": scheduled_travel_time,
        "day_of_week": day_of_week
    }
])


# ============================================================
# MAKE ML PREDICTION
# ============================================================

predicted_delay = model.predict(input_data)[0]


# ============================================================
# CALCULATE EXPECTED ARRIVAL TIME
# ============================================================

predicted_arrival = (
    scheduled_arrival
    + pd.to_timedelta(
        predicted_delay,
        unit="minutes"
    )
)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("ETA PREDICTION")
print("=" * 60)

print(
    f"\nPredicted arrival delay at "
    f"{next_station}: "
    f"{predicted_delay:.2f} minutes"
)

print(
    f"\nScheduled arrival at "
    f"{next_station}: "
    f"{scheduled_arrival.strftime('%d-%m-%Y %I:%M %p')}"
)

print(
    f"Expected arrival at "
    f"{next_station}: "
    f"{predicted_arrival.strftime('%d-%m-%Y %I:%M %p')}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)