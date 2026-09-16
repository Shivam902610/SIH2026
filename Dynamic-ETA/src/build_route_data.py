import json
import csv
import os

# File locations
INPUT_FILE = "data/archive (1)/schedules.json"
OUTPUT_FILE = "data/dataset/train_routes.csv"

# Make sure output directory exists
os.makedirs("data/dataset", exist_ok=True)

# Load schedules
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    schedules = json.load(file)

# We only need our pilot trains for now
pilot_trains = ["12919", "12920", "12951", "12952", "12002"]

# Store processed records
route_records = []

for train_number in pilot_trains:

    # Get all schedule records for this train
    train_schedule = [
        record
        for record in schedules
        if record.get("train_number") == train_number
    ]

    # Preserve the order in schedules.json
    for sequence, record in enumerate(train_schedule, start=1):

        route_records.append({
            "train_number": train_number,
            "sequence": sequence,
            "station_code": record.get("station_code"),
            "station_name": record.get("station_name"),
            "day": record.get("day"),
            "scheduled_arrival": record.get("arrival"),
            "scheduled_departure": record.get("departure")
        })

# Save CSV
fieldnames = [
    "train_number",
    "sequence",
    "station_code",
    "station_name",
    "day",
    "scheduled_arrival",
    "scheduled_departure"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(route_records)

print("Route data created successfully!")
print("Output:", OUTPUT_FILE)
print("Total records:", len(route_records))

for train_number in pilot_trains:
    count = sum(
        1 for record in route_records
        if record["train_number"] == train_number
    )
    print(f"{train_number}: {count} stations")