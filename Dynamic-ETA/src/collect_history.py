import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load API key
load_dotenv()

api_key = os.getenv("RAILRADAR_API_KEY")

# Train we want to collect
train_number = "12919"

# Date range for our first test
start_date = "2026-09-06"
end_date = "2026-09-10"

# API URL
url = f"https://api.railradar.in/v1/trains/{train_number}/live"

headers = {
    "Authorization": f"Bearer {api_key}"
}


def get_dates(start, end):

    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    dates = []

    current = start

    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


# Get all dates
dates = get_dates(start_date, end_date)

print("Dates to collect:")
print(dates)

print("\nTotal API requests:", len(dates))


for journey_date in dates:

    print("\n--------------------------------")
    print("Collecting:", journey_date)

    params = {
        "date": journey_date
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        print("Status code:", response.status_code)

        # Successful request
        if response.status_code == 200:

            result = response.json()
            data = result["data"]

            actual_date = data["startDate"]
            status = data.get("status")

            print("Returned date:", actual_date)
            print("Status:", status)

            rows = []

            for station in data["route"]:

                if station.get("isHalt") != True:
                    continue

                rows.append({
                    "train_number": train_number,
                    "journey_date": actual_date,

                    "station_sequence": station.get("sequence"),
                    "station_code": station.get("stationCode"),
                    "station_name": station.get("stationName"),

                    "scheduled_arrival": station.get("scheduledArrival"),
                    "actual_arrival": station.get("actualArrival"),

                    "scheduled_departure": station.get("scheduledDeparture"),
                    "actual_departure": station.get("actualDeparture"),

                    "arrival_delay": station.get("delayArrival"),
                    "departure_delay": station.get("delayDeparture")
                })

            df = pd.DataFrame(rows)

            # Create folder
            os.makedirs("data/processed", exist_ok=True)

            filename = (
                f"data/processed/"
                f"{train_number}_{actual_date}.csv"
            )

            df.to_csv(filename, index=False)

            print("Stations:", len(df))
            print("Saved:", filename)

        else:

            print("Request failed:")
            print(response.text)

    except requests.RequestException as e:

        print("Network/API error:")
        print(e)


print("\n================================")
print("Collection finished!")
print("================================")