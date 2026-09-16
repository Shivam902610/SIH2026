import os
import math
import pandas as pd


# ============================================================
# FILE LOCATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "train_history_clean.csv"
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_history():
    """
    Load the cleaned historical train dataset.
    """

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    return df


# ============================================================
# CONVERT VALUES TO JSON-SAFE VALUES
# ============================================================

def clean_value(value):
    """
    Convert pandas / NumPy values into JSON-safe Python values.

    NaN, +inf and -inf are converted to None because
    standard JSON does not support them.
    """

    # Missing pandas value
    if pd.isna(value):
        return None

    # Float infinity / NaN protection
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    # NumPy numeric types
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return value


def clean_record(record):
    """
    Make every value in a dictionary JSON-safe.
    """

    cleaned = {}

    for key, value in record.items():
        cleaned[key] = clean_value(value)

    return cleaned


# ============================================================
# GET AVAILABLE JOURNEYS
# ============================================================

def get_available_journeys():
    """
    Return every unique train + journey-date combination.

    Important:
    A journey is identified by BOTH train number and date.
    """

    df = load_history()

    journeys = (
        df[
            [
                "train_number",
                "journey_date"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "train_number",
                "journey_date"
            ]
        )
    )

    result = []

    for _, row in journeys.iterrows():

        result.append(
            {
                "train_number": clean_value(row["train_number"]),
                "journey_date": str(row["journey_date"])
            }
        )

    return result


# ============================================================
# GET ONE HISTORICAL JOURNEY
# ============================================================

def get_journey(train_number, journey_date):
    """
    Return all historical station records for one journey.
    """

    df = load_history()

    # Convert train number to string for safe comparison
    df["train_number"] = df["train_number"].astype(str)

    train_number = str(train_number)

    # Select requested journey
    journey = df[
        (df["train_number"] == train_number)
        &
        (df["journey_date"].astype(str) == str(journey_date))
    ].copy()

    if journey.empty:
        return None

    # Sort by station sequence
    if "station_sequence" in journey.columns:
        journey = journey.sort_values("station_sequence")

    # Convert dataframe records to dictionaries
    records = journey.to_dict(orient="records")

    # Make every value JSON safe
    records = [
        clean_record(record)
        for record in records
    ]

    return records


# ============================================================
# CREATE REPLAY DATA
# ============================================================

def create_replay(train_number, journey_date):
    """
    Create a replay of a historical train journey.

    Each station record represents one point in the
    historical journey that can later be played step-by-step
    in the frontend.
    """

    records = get_journey(
        train_number,
        journey_date
    )

    if records is None:
        return None

    # --------------------------------------------------------
    # Create replay steps
    # --------------------------------------------------------

    replay_steps = []

    for index, record in enumerate(records):

        step = {
            "step": index + 1,

            "station_sequence": clean_value(
                record.get("station_sequence")
            ),

            "station_code": clean_value(
                record.get("station_code")
            ),

            "station_name": clean_value(
                record.get("station_name")
            ),

            "scheduled_arrival": clean_value(
                record.get("scheduled_arrival")
            ),

            "actual_arrival": clean_value(
                record.get("actual_arrival")
            ),

            "arrival_delay": clean_value(
                record.get("arrival_delay")
            ),

            "scheduled_departure": clean_value(
                record.get("scheduled_departure")
            ),

            "actual_departure": clean_value(
                record.get("actual_departure")
            ),

            "departure_delay": clean_value(
                record.get("departure_delay")
            ),
        }

        replay_steps.append(step)

    # --------------------------------------------------------
    # Return complete replay object
    # --------------------------------------------------------

    return {
        "success": True,

        "train_number": str(train_number),

        "journey_date": str(journey_date),

        "total_steps": len(replay_steps),

        "steps": replay_steps
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HISTORICAL TRAIN REPLAY")
    print("=" * 60)

    journeys = get_available_journeys()

    print("\nAvailable journeys:", len(journeys))

    for journey in journeys[:10]:
        print(journey)

    print("\n" + "=" * 60)

    # Test journey
    test_train = "12919"
    test_date = "2026-09-10"

    replay = create_replay(
        test_train,
        test_date
    )

    if replay is None:

        print(
            f"No journey found for "
            f"{test_train} on {test_date}"
        )

    else:

        print(
            f"Train: {replay['train_number']}"
        )

        print(
            f"Journey date: {replay['journey_date']}"
        )

        print(
            f"Replay steps: {replay['total_steps']}"
        )

        print("\nFirst 3 steps:")

        for step in replay["steps"][:3]:
            print(step)