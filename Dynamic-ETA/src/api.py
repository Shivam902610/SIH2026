import os
import math
import joblib
import requests
import pandas as pd

from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.replay import (
    get_available_journeys,
    create_replay
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

RAILRADAR_API_KEY = os.getenv(
    "RAILRADAR_API_KEY"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DYNAMIC_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dynamic_delay_change_model.pkl"
)

OLD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "focused_eta_model.pkl"
)


# ============================================================
# LOAD DYNAMIC MODEL
# ============================================================

try:

    dynamic_model = joblib.load(
        DYNAMIC_MODEL_PATH
    )

    print("=" * 60)
    print("DYNAMIC ML MODEL LOADED")
    print("=" * 60)
    print(
        f"Model: {DYNAMIC_MODEL_PATH}"
    )

except Exception as e:

    dynamic_model = None

    print("=" * 60)
    print("WARNING: DYNAMIC MODEL FAILED TO LOAD")
    print("=" * 60)
    print(e)


# ============================================================
# LOAD OLD MODEL AS FALLBACK
# ============================================================

try:

    fallback_model = joblib.load(
        OLD_MODEL_PATH
    )

    print(
        f"Fallback model: {OLD_MODEL_PATH}"
    )

except Exception as e:

    fallback_model = None

    print(
        "WARNING: Fallback model failed to load"
    )

    print(e)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dynamic Train ETA API",
    description=(
        "ML-based dynamic ETA prediction "
        "for Indian Railways"
    ),
    version="2.1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# JSON SAFE VALUE
# ============================================================

def clean_value(value):

    if value is None:

        return None

    try:

        if pd.isna(value):

            return None

    except Exception:

        pass

    if isinstance(value, float):

        if (
            math.isnan(value)
            or
            math.isinf(value)
        ):

            return None

    if hasattr(value, "item"):

        try:

            value = value.item()

        except Exception:

            pass

    return value


# ============================================================
# VALIDATE TRAIN NUMBER
# ============================================================

def validate_train_number(
    train_number
):

    train_number = str(
        train_number
    ).strip()

    if not train_number.isdigit():

        raise HTTPException(
            status_code=400,
            detail=(
                "Train number must contain only digits"
            )
        )

    if len(train_number) != 5:

        raise HTTPException(
            status_code=400,
            detail=(
                "Train number must be 5 digits"
            )
        )

    return train_number


# ============================================================
# GET LIVE TRAIN DATA
# ============================================================

def get_live_train(
    train_number
):

    if not RAILRADAR_API_KEY:

        raise HTTPException(
            status_code=500,
            detail=(
                "RAILRADAR_API_KEY not found in .env"
            )
        )

    url = (
        "https://api.railradar.in/"
        f"v1/trains/{train_number}/live"
    )

    headers = {
        "Authorization":
            f"Bearer {RAILRADAR_API_KEY}",

        "Accept":
            "application/json"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

    except requests.RequestException as e:

        raise HTTPException(
            status_code=503,
            detail=(
                f"Unable to contact RailRadar: {str(e)}"
            )
        )

    if response.status_code != 200:

        try:

            error_data = response.json()

        except Exception:

            error_data = response.text

        raise HTTPException(
            status_code=response.status_code,
            detail=error_data
        )

    try:

        result = response.json()

    except Exception:

        raise HTTPException(
            status_code=502,
            detail=(
                "RailRadar returned invalid JSON"
            )
        )

    if not result.get(
        "success",
        False
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "RailRadar returned unsuccessful data"
            )
        )

    return result["data"]


# ============================================================
# FIND ROUTE ITEM
# ============================================================

def find_route_item(
    route,
    station_code,
    sequence
):

    for item in route:

        if (
            item.get("stationCode")
            == station_code
            and
            item.get("sequence")
            == sequence
        ):

            return item

    return None


# ============================================================
# FIND PREVIOUS HALT
# ============================================================

def find_previous_halt(
    route,
    current_sequence
):

    previous = None

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
                previous is None
                or
                sequence >
                previous.get("sequence")
            ):

                previous = item

    return previous


# ============================================================
# EXTRACT LIVE STATE
# ============================================================

def extract_train_state(
    data
):

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

    if not route:

        raise HTTPException(
            status_code=404,
            detail="Train route unavailable"
        )

    if not current_location:

        raise HTTPException(
            status_code=404,
            detail="Current train location unavailable"
        )

    if not next_halt:

        raise HTTPException(
            status_code=404,
            detail="Next halt unavailable"
        )

    # --------------------------------------------------------
    # CURRENT STATION
    # --------------------------------------------------------

    current_code = current_location.get(
        "stationCode"
    )

    current_sequence = current_location.get(
        "sequence"
    )

    current_item = find_route_item(
        route,
        current_code,
        current_sequence
    )

    if current_item is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Current station not found in route"
            )
        )

    # --------------------------------------------------------
    # NEXT STATION
    # --------------------------------------------------------

    next_code = next_halt.get(
        "stationCode"
    )

    next_sequence = next_halt.get(
        "sequence"
    )

    next_item = find_route_item(
        route,
        next_code,
        next_sequence
    )

    if next_item is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Next station not found in route"
            )
        )

    # --------------------------------------------------------
    # CURRENT DELAY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREVIOUS HALT
    # --------------------------------------------------------

    previous_item = find_previous_halt(
        route,
        current_sequence
    )

    if previous_item:

        previous_delay = previous_item.get(
            "delayDeparture"
        )

    else:

        previous_delay = None

    if previous_delay is None:

        previous_delay = current_delay

    previous_delay = float(
        previous_delay
    )

    # --------------------------------------------------------
    # OBSERVED DELAY CHANGE
    # --------------------------------------------------------

    delay_change = (
        current_delay
        -
        previous_delay
    )

    # --------------------------------------------------------
    # SCHEDULED TIMES
    # --------------------------------------------------------

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

    if (
        scheduled_departure is None
        or
        scheduled_arrival is None
    ):

        scheduled_travel_time = 0

    else:

        try:

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

        except Exception:

            scheduled_travel_time = 0

    # --------------------------------------------------------
    # DAY OF WEEK
    # --------------------------------------------------------

    journey_date = data.get(
        "startDate"
    )

    if journey_date:

        try:

            day_of_week = datetime.strptime(
                journey_date,
                "%Y-%m-%d"
            ).weekday()

        except Exception:

            day_of_week = 0

    else:

        day_of_week = 0

    return {

        "current_station":
            current_code,

        "current_station_name":
            current_item.get(
                "stationName"
            ),

        "next_station":
            next_code,

        "next_station_name":
            next_item.get(
                "stationName"
            ),

        "station_sequence":
            current_sequence,

        "next_station_sequence":
            next_sequence,

        "current_delay":
            current_delay,

        "previous_delay":
            previous_delay,

        "delay_change":
            delay_change,

        "scheduled_travel_time":
            scheduled_travel_time,

        "scheduled_arrival":
            scheduled_arrival,

        "scheduled_departure":
            scheduled_departure,

        "journey_date":
            journey_date,

        "day_of_week":
            day_of_week
    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Dynamic Train ETA API is running",

        "status":
            "online",

        "model":
            "Dynamic Delay Change Model"
    }


# ============================================================
# LIVE ETA PREDICTION
# ============================================================

@app.get(
    "/predict/{train_number}"
)
def predict_eta(
    train_number: str
):

    train_number = validate_train_number(
        train_number
    )

    if dynamic_model is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Dynamic ML model is not loaded"
            )
        )

    # --------------------------------------------------------
    # GET LIVE DATA
    # --------------------------------------------------------

    data = get_live_train(
        train_number
    )

    # --------------------------------------------------------
    # EXTRACT STATE
    # --------------------------------------------------------

    state = extract_train_state(
        data
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {

                "train_number":
                    train_number,

                "current_station":
                    state[
                        "current_station"
                    ],

                "next_station":
                    state[
                        "next_station"
                    ],

                "station_sequence":
                    state[
                        "station_sequence"
                    ],

                "current_departure_delay":
                    state[
                        "current_delay"
                    ],

                "previous_departure_delay":
                    state[
                        "previous_delay"
                    ],

                "delay_change":
                    state[
                        "delay_change"
                    ],

                "next_station_sequence":
                    state[
                        "next_station_sequence"
                    ],

                "scheduled_travel_time":
                    state[
                        "scheduled_travel_time"
                    ]
            }
        ]
    )

    # --------------------------------------------------------
    # PREDICT DELAY CHANGE
    # --------------------------------------------------------

    try:

        predicted_delay_change = (
            dynamic_model.predict(
                input_data
            )[0]
        )

        predicted_delay_change = float(
            predicted_delay_change
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Dynamic prediction failed: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # PREDICT NEXT-STATION DELAY
    # --------------------------------------------------------

    current_delay = state[
        "current_delay"
    ]

    predicted_next_delay = (
        current_delay
        +
        predicted_delay_change
    )

    predicted_next_delay = max(
        0,
        predicted_next_delay
    )

    # --------------------------------------------------------
    # CALCULATE ETA
    # --------------------------------------------------------

    scheduled_arrival = state[
        "scheduled_arrival"
    ]

    expected_arrival = None

    if scheduled_arrival:

        try:

            scheduled_time = pd.to_datetime(
                scheduled_arrival
            )

            expected_time = (
                scheduled_time
                +
                pd.Timedelta(
                    minutes=predicted_next_delay
                )
            )

            expected_arrival = (
                expected_time.isoformat()
            )

        except Exception:

            expected_arrival = None

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "train_number":
            train_number,

        "train_name":
            data.get(
                "trainName"
            ),

        "journey_date":
            state[
                "journey_date"
            ],

        "current_station":
            state[
                "current_station"
            ],

        "current_station_name":
            state[
                "current_station_name"
            ],

        "next_station":
            state[
                "next_station"
            ],

        "next_station_name":
            state[
                "next_station_name"
            ],

        "current_delay_minutes":
            round(
                current_delay,
                2
            ),

        "previous_delay_minutes":
            round(
                state[
                    "previous_delay"
                ],
                2
            ),

        "observed_delay_change_minutes":
            round(
                state[
                    "delay_change"
                ],
                2
            ),

        "predicted_delay_change_minutes":
            round(
                predicted_delay_change,
                2
            ),

        "predicted_delay_minutes":
            round(
                predicted_next_delay,
                2
            ),

        "scheduled_travel_time_minutes":
            round(
                state[
                    "scheduled_travel_time"
                ],
                2
            ),

        "scheduled_arrival":
            scheduled_arrival,

        "expected_arrival":
            expected_arrival,

        "model":
            "Random Forest - Dynamic Delay Change"
    }


# ============================================================
# REPLAY - AVAILABLE JOURNEYS
# ============================================================

@app.get(
    "/replay/journeys"
)
def replay_journeys():

    try:

        journeys = get_available_journeys()

        return {

            "success":
                True,

            "total_journeys":
                len(journeys),

            "journeys":
                journeys
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to load journeys: {str(e)}"
            )
        )


# ============================================================
# REPLAY - HISTORICAL JOURNEY
# ============================================================

@app.get(
    "/replay/{train_number}/{journey_date}"
)
def replay_train(
    train_number: str,
    journey_date: str
):

    train_number = validate_train_number(
        train_number
    )

    try:

        datetime.strptime(
            journey_date,
            "%Y-%m-%d"
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Journey date must be YYYY-MM-DD"
            )
        )

    try:

        replay = create_replay(
            train_number,
            journey_date
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to create replay: {str(e)}"
            )
        )

    if replay is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical journey found "
                f"for train {train_number} "
                f"on {journey_date}"
            )
        )

    return replay