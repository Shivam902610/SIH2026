import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("RAILRADAR_API_KEY")

if not API_KEY:
    raise ValueError(
        "RAILRADAR_API_KEY not found. Check your .env file."
    )


# ============================================================
# TRAIN PRIORITY
# ============================================================

TRAINS = [
    "12002",
    "12952",
    "12951",
    "12920",
    "12919"
]


# ============================================================
# DATES
# ============================================================

DATES = [
    "2026-08-10",
    "2026-08-11",
    "2026-08-12",
    "2026-08-13",
    "2026-08-14",
    "2026-08-15",
    "2026-08-16",
    "2026-08-17",
    "2026-08-18",
    "2026-08-19"
]


# ============================================================
# SAFETY SETTINGS
# ============================================================

# FIRST TEST:
# Only make ONE API request.
#
# After we confirm that the route is being saved correctly,
# change this back to 8.

MAX_REQUESTS = 8

REQUEST_DELAY_SECONDS = 7

OUTPUT_DIR = os.path.join(
    "data",
    "processed"
)

BASE_URL = "https://api.railradar.in/v1/trains"


# ============================================================
# SETUP
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def output_file(train_number, journey_date):
    """
    Return the CSV filename for a journey.
    """

    return os.path.join(
        OUTPUT_DIR,
        f"{train_number}_{journey_date}.csv"
    )


def already_collected(train_number, journey_date):
    """
    Check whether this journey already exists.
    """

    return os.path.exists(
        output_file(
            train_number,
            journey_date
        )
    )


def save_journey(train_number, journey_date, data):
    """
    Extract route data and save it as CSV.
    """

    route = data.get(
        "route",
        []
    )

    if not route:
        return 0

    rows = []

    for station in route:

        rows.append({
            "train_number": train_number,
            "journey_date": journey_date,
            "station_sequence": station.get("sequence"),
            "station_code": station.get("stationCode"),
            "station_name": station.get("stationName"),
            "scheduled_arrival": station.get("scheduledArrival"),
            "actual_arrival": station.get("actualArrival"),
            "arrival_delay": station.get("delayArrival"),
            "scheduled_departure": station.get("scheduledDeparture"),
            "actual_departure": station.get("actualDeparture"),
            "departure_delay": station.get("delayDeparture"),
            "status": station.get("status")
        })

    df = pd.DataFrame(rows)

    filename = output_file(
        train_number,
        journey_date
    )

    df.to_csv(
        filename,
        index=False
    )

    return len(df)


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("DYNAMIC ETA - MULTI TRAIN HISTORICAL COLLECTOR")
print("=" * 60)

print(
    f"Trains: {len(TRAINS)}"
)

print(
    f"Dates per train: {len(DATES)}"
)

print(
    f"Maximum requests this run: {MAX_REQUESTS}"
)

print(
    f"Delay between requests: "
    f"{REQUEST_DELAY_SECONDS} seconds"
)


# ============================================================
# FIND MISSING JOURNEYS
# ============================================================

missing_journeys = []

for train_number in TRAINS:

    for journey_date in DATES:

        if not already_collected(
            train_number,
            journey_date
        ):

            missing_journeys.append(
                (
                    train_number,
                    journey_date
                )
            )


total_possible = (
    len(TRAINS) *
    len(DATES)
)

already_count = (
    total_possible -
    len(missing_journeys)
)


print(
    f"\nTotal possible journeys "
    f"in this date range: {total_possible}"
)

print(
    f"Already collected "
    f"in this date range: {already_count}"
)

print(
    f"New API requests needed: "
    f"{len(missing_journeys)}"
)


if not missing_journeys:

    print(
        "\nNo new journeys to collect."
    )

    raise SystemExit


# ============================================================
# COLLECTION
# ============================================================

attempts = 0
successes = 0
failures = 0
no_route = 0


for train_number, journey_date in missing_journeys:

    if attempts >= MAX_REQUESTS:

        print("\n")
        print("=" * 60)
        print("MAXIMUM REQUEST LIMIT REACHED")
        print("Stopping safely.")
        print("=" * 60)

        break

    attempts += 1

    print(
        f"\n[Request {attempts}/{MAX_REQUESTS}] "
        f"Train {train_number} | {journey_date}"
    )


    # ========================================================
    # API REQUEST
    # ========================================================

    url = (
        f"{BASE_URL}/"
        f"{train_number}/live"
    )

    params = {
        "date": journey_date,
        "authoritative": "true"
    }


    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=20
        )


        print(
            f"HTTP Status: {response.status_code}"
        )


        # ====================================================
        # HTTP 200
        # ====================================================

        if response.status_code == 200:

            try:

                response_json = response.json()

            except ValueError:

                failures += 1

                print(
                    "FAILED: Response was not valid JSON."
                )

                print(
                    response.text[:700]
                )

                continue


            # ------------------------------------------------
            # IMPORTANT:
            #
            # RailRadar response:
            #
            # {
            #     "success": true,
            #     "data": {
            #         "trainNumber": "...",
            #         "route": [...]
            #     }
            # }
            #
            # Therefore we first get "data".
            # ------------------------------------------------

            data = response_json.get(
                "data",
                {}
            )


            if not isinstance(data, dict):

                failures += 1

                print(
                    "FAILED: 'data' field "
                    "is not an object."
                )

                print(
                    str(response_json)[:1000]
                )

                continue


            # ------------------------------------------------
            # Display basic information
            # ------------------------------------------------

            train_name = data.get(
                "trainName",
                "Unknown"
            )

            journey_status = data.get(
                "status",
                "Unknown"
            )

            start_date = data.get(
                "startDate",
                journey_date
            )


            print(
                f"Train name: {train_name}"
            )

            print(
                f"Journey status: {journey_status}"
            )

            print(
                f"Start date: {start_date}"
            )


            # ------------------------------------------------
            # Get route
            # ------------------------------------------------

            route = data.get(
                "route",
                []
            )


            if route:

                saved_rows = save_journey(
                    train_number,
                    journey_date,
                    data
                )


                if saved_rows > 0:

                    successes += 1

                    filename = output_file(
                        train_number,
                        journey_date
                    )

                    print(
                        f"SUCCESS: "
                        f"{saved_rows} stations saved"
                    )

                    print(
                        f"File: {filename}"
                    )


                else:

                    failures += 1

                    print(
                        "FAILED: Route was returned "
                        "but no rows were saved."
                    )


            else:

                no_route += 1

                print(
                    "NO ROUTE DATA returned."
                )

                print(
                    "Available data keys:"
                )

                print(
                    list(data.keys())
                )

                print(
                    "\nData preview:"
                )

                print(
                    str(data)[:1200]
                )


        # ====================================================
        # HTTP 429
        # ====================================================

        elif response.status_code == 429:

            failures += 1

            print(
                "RATE LIMIT (429) received."
            )

            print(
                "Stopping immediately "
                "to protect the API."
            )

            print(
                response.text[:700]
            )

            break


        # ====================================================
        # HTTP 503
        # ====================================================

        elif response.status_code == 503:

            failures += 1

            print(
                "SERVICE UNAVAILABLE (503)."
            )

            print(
                "RailRadar upstream service "
                "may be temporarily degraded."
            )

            print(
                response.text[:700]
            )


        # ====================================================
        # OTHER HTTP ERRORS
        # ====================================================

        else:

            failures += 1

            print(
                f"FAILED: HTTP "
                f"{response.status_code}"
            )

            print(
                response.text[:700]
            )


    except requests.RequestException as e:

        failures += 1

        print(
            f"REQUEST ERROR: {e}"
        )


    # ========================================================
    # WAIT
    # ========================================================

    if attempts < MAX_REQUESTS:

        print(
            f"\nWaiting "
            f"{REQUEST_DELAY_SECONDS - 0.5:.1f} "
            f"seconds before next request..."
        )

        time.sleep(
            REQUEST_DELAY_SECONDS - 0.5
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n")

print("=" * 60)
print("COLLECTION SUMMARY")
print("=" * 60)

print(
    f"API attempts:        {attempts}"
)

print(
    f"Successful journeys: {successes}"
)

print(
    f"No route data:       {no_route}"
)

print(
    f"Other failures:      {failures}"
)

print(
    "\nCollection finished safely."
)

print("=" * 60)