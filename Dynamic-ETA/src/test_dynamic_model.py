import os
import joblib
import requests
import pandas as pd

from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("RAILRADAR_API_KEY")

TRAIN_NUMBER = "12919"

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dynamic_delay_change_model.pkl"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not API_KEY:

    print(
        "ERROR: RAILRADAR_API_KEY not found in .env"
    )

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING DYNAMIC MODEL")
print("=" * 60)

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully")


# ============================================================
# GET LIVE DATA
# ============================================================

print()
print("=" * 60)
print(
    f"GETTING LIVE DATA FOR TRAIN {TRAIN_NUMBER}"
)
print("=" * 60)

url = (
    f"https://api.railradar.in/"
    f"v1/trains/{TRAIN_NUMBER}/live"
)

headers = {
    "Authorization":
        f"Bearer {API_KEY}",

    "Accept":
        "application/json"
}

response = requests.get(
    url,
    headers=headers,
    timeout=20
)

print(
    "RailRadar status:",
    response.status_code
)

if response.status_code != 200:

    print(response.text)

    raise SystemExit


data = response.json()["data"]


# ============================================================
# GET MAIN OBJECTS
# ============================================================

route = data.get(
    "route",
    []
)

current_location = data.get(
    "currentLocation"
)

next_halt = data.get(
    "nextHalt"
)


if not current_location:

    print(
        "ERROR: Current location unavailable"
    )

    raise SystemExit


if not next_halt:

    print(
        "ERROR: Next halt unavailable"
    )

    raise SystemExit


# ============================================================
# CURRENT STATION
# ============================================================

current_sequence = current_location.get(
    "sequence"
)

current_code = current_location.get(
    "stationCode"
)


current_item = None

for item in route:

    if item.get(
        "sequence"
    ) == current_sequence:

        current_item = item

        break


if current_item is None:

    print(
        "ERROR: Current station not found in route"
    )

    raise SystemExit


# ============================================================
# NEXT HALT
# ============================================================

next_code = next_halt.get(
    "stationCode"
)

next_sequence = next_halt.get(
    "sequence"
)


next_item = None

for item in route:

    if (
        item.get("stationCode")
        == next_code
        and
        item.get("sequence")
        == next_sequence
    ):

        next_item = item

        break


if next_item is None:

    print(
        "ERROR: Next halt not found in route"
    )

    raise SystemExit


# ============================================================
# CURRENT DELAY
# ============================================================

current_delay = current_item.get(
    "delayDeparture"
)

if current_delay is None:

    current_delay = current_location.get(
        "delayMinutes"
    )

if current_delay is None:

    current_delay = 0


current_delay = float(
    current_delay
)


# ============================================================
# PREVIOUS HALT
# ============================================================

previous_record = None

for item in route:

    sequence = item.get(
        "sequence"
    )

    if (
        sequence is not None
        and
        sequence < current_sequence
        and
        item.get("isHalt")
    ):

        if (
            previous_record is None
            or
            sequence >
            previous_record.get("sequence")
        ):

            previous_record = item


# ============================================================
# PREVIOUS DELAY
# ============================================================

if previous_record:

    previous_delay = previous_record.get(
        "delayDeparture"
    )

else:

    previous_delay = None


if previous_delay is None:

    previous_delay = current_delay


previous_delay = float(
    previous_delay
)


# ============================================================
# OBSERVED DELAY CHANGE
# ============================================================

delay_change = (
    current_delay
    -
    previous_delay
)


# ============================================================
# SCHEDULED TRAVEL TIME
# ============================================================

scheduled_departure = (
    current_item.get(
        "scheduledDeparture"
    )
)

scheduled_arrival = (
    next_item.get(
        "scheduledArrival"
    )
)


print()
print(
    "Current scheduled departure:",
    scheduled_departure
)

print(
    "Next scheduled arrival:",
    scheduled_arrival
)


if (
    scheduled_departure is None
    or
    scheduled_arrival is None
):

    print()
    print(
        "WARNING: Scheduled time missing."
    )

    print(
        "The model can still be tested, "
        "but ETA cannot be calculated."
    )

    scheduled_travel_time = 0

else:

    departure_time = pd.to_datetime(
        scheduled_departure
    )

    arrival_time = pd.to_datetime(
        scheduled_arrival
    )

    scheduled_travel_time = (
        arrival_time
        -
        departure_time
    ).total_seconds() / 60


# ============================================================
# ML INPUT
# ============================================================

input_data = pd.DataFrame(
    [
        {

            "train_number":
                TRAIN_NUMBER,

            "current_station":
                current_code,

            "next_station":
                next_code,

            "station_sequence":
                current_sequence,

            "current_departure_delay":
                current_delay,

            "previous_departure_delay":
                previous_delay,

            "delay_change":
                delay_change,

            "next_station_sequence":
                next_sequence,

            "scheduled_travel_time":
                scheduled_travel_time
        }
    ]
)


# ============================================================
# PREDICT DELAY CHANGE
# ============================================================

predicted_delay_change = model.predict(
    input_data
)[0]

predicted_delay_change = float(
    predicted_delay_change
)


# ============================================================
# PREDICT NEXT-STATION DELAY
# ============================================================

predicted_next_delay = (
    current_delay
    +
    predicted_delay_change
)

predicted_next_delay = max(
    0,
    predicted_next_delay
)


# ============================================================
# CALCULATE ETA
# ============================================================

expected_arrival = None

if scheduled_arrival is not None:

    scheduled_arrival_time = pd.to_datetime(
        scheduled_arrival
    )

    expected_arrival = (
        scheduled_arrival_time
        +
        pd.Timedelta(
            minutes=predicted_next_delay
        )
    )


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 60)
print("LIVE TRAIN STATE")
print("=" * 60)

print(
    f"Train             : "
    f"{TRAIN_NUMBER} - "
    f"{data.get('trainName')}"
)

print(
    f"Journey date      : "
    f"{data.get('startDate')}"
)

print(
    f"Current station   : "
    f"{current_code} - "
    f"{current_item.get('stationName')}"
)

print(
    f"Next station      : "
    f"{next_code} - "
    f"{next_item.get('stationName')}"
)

print(
    f"Current delay     : "
    f"{current_delay:.2f} min"
)

print(
    f"Previous delay    : "
    f"{previous_delay:.2f} min"
)

print(
    f"Observed delay Δ  : "
    f"{delay_change:.2f} min"
)

print(
    f"Scheduled travel  : "
    f"{scheduled_travel_time:.2f} min"
)


print()
print("=" * 60)
print("DYNAMIC ML PREDICTION")
print("=" * 60)

print(
    f"Predicted delay Δ : "
    f"{predicted_delay_change:.2f} min"
)

print(
    f"Current delay     : "
    f"{current_delay:.2f} min"
)

print(
    f"Predicted next    : "
    f"{predicted_next_delay:.2f} min"
)

if expected_arrival:

    print(
        f"Expected arrival  : "
        f"{expected_arrival.isoformat()}"
    )

else:

    print(
        "Expected arrival  : unavailable"
    )


print()
print("=" * 60)
print("CALCULATION")
print("=" * 60)

print(
    f"{current_delay:.2f}"
    f" + "
    f"{predicted_delay_change:.2f}"
    f" = "
    f"{predicted_next_delay:.2f}"
    f" min delay"
)

print("=" * 60)