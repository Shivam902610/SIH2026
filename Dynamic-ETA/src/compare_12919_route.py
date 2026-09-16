import pandas as pd

ROUTE_FILE = "data/dataset/train_routes.csv"
HISTORY_FILE = "data/dataset/train_history_clean.csv"

# Load data
routes = pd.read_csv(ROUTE_FILE)
history = pd.read_csv(HISTORY_FILE)

# Select train 12919
route = routes[routes["train_number"].astype(str) == "12919"].copy()
history = history[history["train_number"].astype(str) == "12919"].copy()

# Remove duplicate station codes while keeping original order
route = route.drop_duplicates(subset=["station_code"])
history = history.drop_duplicates(subset=["station_code"])

print("\n========== STATIC ROUTE ==========")
print("Total stations:", len(route))

for _, row in route.iterrows():
    print(
        f'{int(row["sequence"]):3} | '
        f'{row["station_code"]:6} | '
        f'{row["station_name"]}'
    )


print("\n========== HISTORICAL ROUTE ==========")
print("Total stations:", len(history))

for _, row in history.sort_values("station_sequence").iterrows():
    print(
        f'{int(row["station_sequence"]):3} | '
        f'{row["station_code"]:6} | '
        f'{row["station_name"]}'
    )


print("\n========== START / END ==========")

first_route = route.iloc[0]
last_route = route.iloc[-1]

first_history = history.sort_values("station_sequence").iloc[0]
last_history = history.sort_values("station_sequence").iloc[-1]

print("\nStatic route:")
print("Start:", first_route["station_code"], "-", first_route["station_name"])
print("End:  ", last_route["station_code"], "-", last_route["station_name"])

print("\nHistorical route:")
print("Start:", first_history["station_code"], "-", first_history["station_name"])
print("End:  ", last_history["station_code"], "-", last_history["station_name"])


print("\n========== COMMON STATIONS IN ORDER ==========")

history_codes = set(history["station_code"])

common = route[route["station_code"].isin(history_codes)]

print("Common stations:", len(common))

for _, row in common.iterrows():
    print(
        f'{int(row["sequence"]):3} | '
        f'{row["station_code"]:6} | '
        f'{row["station_name"]}'
    )