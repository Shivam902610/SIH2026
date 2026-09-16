import pandas as pd

ROUTE_FILE = "data/dataset/train_routes.csv"
HISTORY_FILE = "data/dataset/train_history_clean.csv"


# Load datasets
routes = pd.read_csv(ROUTE_FILE)
history = pd.read_csv(HISTORY_FILE)


print("\n===== BASIC INFORMATION =====")

print("Route data:")
print("Rows:", len(routes))
print("Trains:", routes["train_number"].nunique())

print("\nHistorical data:")
print("Rows:", len(history))
print("Trains:", history["train_number"].nunique())


# Make train numbers strings
routes["train_number"] = routes["train_number"].astype(str)
history["train_number"] = history["train_number"].astype(str)


# Check each train
print("\n===== TRAIN-WISE COMPARISON =====")

for train in sorted(history["train_number"].unique()):

    route_train = routes[routes["train_number"] == train]
    history_train = history[history["train_number"] == train]

    route_stations = set(route_train["station_code"].dropna())
    history_stations = set(history_train["station_code"].dropna())

    matched = route_stations & history_stations
    route_only = route_stations - history_stations
    history_only = history_stations - route_stations

    print(f"\nTrain: {train}")

    print("Route stations:", len(route_stations))
    print("Historical stations:", len(history_stations))
    print("Matched stations:", len(matched))
    print("Only in route data:", len(route_only))
    print("Only in historical data:", len(history_only))

    if route_only:
        print("Route-only examples:", list(route_only)[:10])

    if history_only:
        print("History-only examples:", list(history_only)[:10])


print("\n===== DONE =====")