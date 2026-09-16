import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "train_history_clean.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "ml_dataset_enhanced.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING ENHANCED ML DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nHistorical dataset loaded.")
print("Rows:", len(df))


# ============================================================
# BASIC CLEANING
# ============================================================

df["train_number"] = (
    df["train_number"]
    .astype(str)
)

df["journey_date"] = (
    pd.to_datetime(
        df["journey_date"],
        errors="coerce"
    )
)

df["station_sequence"] = pd.to_numeric(
    df["station_sequence"],
    errors="coerce"
)

df["arrival_delay"] = pd.to_numeric(
    df["arrival_delay"],
    errors="coerce"
)

df["departure_delay"] = pd.to_numeric(
    df["departure_delay"],
    errors="coerce"
)


# ============================================================
# PARSE ISO TIMESTAMPS
# ============================================================
#
# Example:
#
# 2026-08-10T06:01:00+05:30
#
# We only need the time-of-day for calculating scheduled
# travel time and departure hour.
#
# ============================================================

df["scheduled_arrival_dt"] = pd.to_datetime(
    df["scheduled_arrival"],
    errors="coerce"
)

df["scheduled_departure_dt"] = pd.to_datetime(
    df["scheduled_departure"],
    errors="coerce"
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "journey_date",
        "train_number",
        "station_sequence"
    ]
).reset_index(drop=True)


# ============================================================
# UPCOMING RECORDS
# ============================================================

if "status" in df.columns:

    upcoming_mask = (
        df["status"]
        .astype(str)
        .str.lower()
        .eq("upcoming")
    )

else:

    upcoming_mask = pd.Series(
        False,
        index=df.index
    )


print(
    "\nUpcoming station records found:",
    upcoming_mask.sum()
)


# ============================================================
# JOURNEY KEY
# ============================================================

df["journey_key"] = (
    df["train_number"]
    + "_"
    + df["journey_date"]
    .dt.strftime("%Y-%m-%d")
)


# ============================================================
# TOTAL STATIONS PER JOURNEY
# ============================================================

journey_total_stations = (
    df.groupby(
        "journey_key"
    )["station_sequence"]
    .max()
    .to_dict()
)


# ============================================================
# HISTORICAL DATA
# ============================================================
#
# Only completed/historical observations are used.
#
# Upcoming records are excluded.
#
# Historical averages for a journey are calculated using
# journeys BEFORE the current journey date.
#
# ============================================================

history_df = df[
    df["arrival_delay"].notna()
    &
    (~upcoming_mask)
].copy()


# ============================================================
# JOURNEYS
# ============================================================

journeys = (
    df[
        [
            "journey_key",
            "train_number",
            "journey_date"
        ]
    ]
    .drop_duplicates()
    .copy()
)

journeys = journeys.sort_values(
    [
        "journey_date",
        "train_number"
    ]
).reset_index(drop=True)


print(
    "\nTotal historical journeys:",
    len(journeys)
)


# ============================================================
# CREATE ML ROWS
# ============================================================

ml_rows = []


for journey_index, journey_info in enumerate(
    journeys.itertuples(index=False),
    start=1
):

    journey_key = (
        journey_info.journey_key
    )

    train_number = (
        journey_info.train_number
    )

    journey_date = (
        journey_info.journey_date
    )


    print(
        f"\rProcessing journey "
        f"{journey_index}/{len(journeys)}: "
        f"{journey_key}",
        end=""
    )


    # --------------------------------------------------------
    # CURRENT JOURNEY
    # --------------------------------------------------------

    journey = df[
        df["journey_key"]
        == journey_key
    ].copy()


    journey = journey.sort_values(
        "station_sequence"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # ONLY PREVIOUS JOURNEYS
    # --------------------------------------------------------

    previous_history = history_df[
        history_df["journey_date"]
        < journey_date
    ]


    # --------------------------------------------------------
    # HISTORICAL TRAIN DELAY
    # --------------------------------------------------------

    train_history = (
        previous_history[
            previous_history["train_number"]
            == train_number
        ]["arrival_delay"]
    )


    if len(train_history) > 0:

        historical_train_delay = (
            train_history.mean()
        )

    else:

        historical_train_delay = 0.0


    # --------------------------------------------------------
    # STATION TRANSITIONS
    # --------------------------------------------------------

    for i in range(
        len(journey) - 1
    ):

        current = journey.iloc[i]

        next_station = journey.iloc[i + 1]


        # ====================================================
        # TARGET MUST NOT BE UPCOMING
        # ====================================================

        next_status = str(
            next_station.get(
                "status",
                ""
            )
        ).lower()


        if next_status == "upcoming":
            continue


        # ====================================================
        # CURRENT DEPARTURE DELAY
        # ====================================================

        current_departure_delay = (
            current["departure_delay"]
        )


        # ====================================================
        # TARGET ARRIVAL DELAY
        # ====================================================

        target_arrival_delay = (
            next_station["arrival_delay"]
        )


        if pd.isna(
            current_departure_delay
        ):
            continue


        if pd.isna(
            target_arrival_delay
        ):
            continue


        # ====================================================
        # PREVIOUS DEPARTURE DELAY
        # ====================================================

        if i > 0:

            previous = journey.iloc[i - 1]

            previous_departure_delay = (
                previous["departure_delay"]
            )


            if pd.isna(
                previous_departure_delay
            ):

                previous_departure_delay = (
                    current_departure_delay
                )

        else:

            previous_departure_delay = (
                current_departure_delay
            )


        # ====================================================
        # DELAY CHANGE
        # ====================================================

        delay_change = (
            current_departure_delay
            - previous_departure_delay
        )


        # ====================================================
        # SCHEDULED TRAVEL TIME
        # ====================================================

        current_time = (
            current[
                "scheduled_arrival_dt"
            ]
        )

        next_time = (
            next_station[
                "scheduled_arrival_dt"
            ]
        )


        if (
            pd.isna(current_time)
            or pd.isna(next_time)
        ):

            continue


        current_minutes = (
            current_time.hour * 60
            + current_time.minute
        )

        next_minutes = (
            next_time.hour * 60
            + next_time.minute
        )


        scheduled_travel_time = (
            next_minutes
            - current_minutes
        )


        # Handle midnight crossing.

        if scheduled_travel_time < 0:

            scheduled_travel_time += (
                24 * 60
            )


        if scheduled_travel_time <= 0:

            continue


        # ====================================================
        # ROUTE FEATURES
        # ====================================================

        current_sequence = (
            current["station_sequence"]
        )


        total_stations = (
            journey_total_stations[
                journey_key
            ]
        )


        if (
            pd.isna(current_sequence)
            or pd.isna(total_stations)
            or total_stations <= 0
        ):

            continue


        route_progress = (
            current_sequence
            / total_stations
        )


        stations_remaining = (
            total_stations
            - current_sequence
        )


        # ====================================================
        # HISTORICAL STATION DELAY
        # ====================================================

        station_history = (
            previous_history[
                (
                    previous_history[
                        "train_number"
                    ]
                    == train_number
                )
                &
                (
                    previous_history[
                        "station_code"
                    ]
                    == current[
                        "station_code"
                    ]
                )
            ]["arrival_delay"]
        )


        if len(station_history) > 0:

            historical_station_delay = (
                station_history.mean()
            )

        else:

            historical_station_delay = (
                historical_train_delay
            )


        # ====================================================
        # SCHEDULED DEPARTURE HOUR
        # ====================================================

        scheduled_departure_hour = (
            current[
                "scheduled_departure_dt"
            ].hour
            +
            current[
                "scheduled_departure_dt"
            ].minute / 60
            if pd.notna(
                current[
                    "scheduled_departure_dt"
                ]
            )
            else 0.0
        )


        # ====================================================
        # DAY OF WEEK
        # ====================================================

        day_of_week = (
            journey_date.dayofweek
        )


        # ====================================================
        # SAVE ROW
        # ====================================================

        ml_rows.append(
            {

                "train_number":
                    train_number,

                "journey_date":
                    journey_date.strftime(
                        "%Y-%m-%d"
                    ),

                "current_station":
                    current[
                        "station_code"
                    ],

                "next_station":
                    next_station[
                        "station_code"
                    ],

                "station_sequence":
                    current_sequence,

                "current_departure_delay":
                    current_departure_delay,

                "previous_departure_delay":
                    previous_departure_delay,

                "delay_change":
                    delay_change,

                "next_station_sequence":
                    next_station[
                        "station_sequence"
                    ],

                "scheduled_travel_time":
                    scheduled_travel_time,

                "route_progress":
                    route_progress,

                "stations_remaining":
                    stations_remaining,

                "historical_station_delay":
                    historical_station_delay,

                "historical_train_delay":
                    historical_train_delay,

                "scheduled_departure_hour":
                    scheduled_departure_hour,

                "target_arrival_delay":
                    target_arrival_delay,

                "day_of_week":
                    day_of_week
            }
        )


print("\n")


# ============================================================
# CREATE DATAFRAME
# ============================================================

ml_df = pd.DataFrame(
    ml_rows
)


# ============================================================
# SAFETY CHECK
# ============================================================

if ml_df.empty:

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(
        "\nNo ML rows were created."
    )

    print(
        "The existing ML dataset was not modified."
    )

    raise SystemExit(1)


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

ml_df = ml_df.replace(
    [
        float("inf"),
        float("-inf")
    ],
    pd.NA
)


ml_df = ml_df.dropna()

ml_df = ml_df.reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("ENHANCED ML DATASET CREATED")
print("=" * 60)

print(
    "\nRows:",
    len(ml_df)
)

print(
    "Columns:",
    len(ml_df.columns)
)


print(
    "\nColumns:"
)

print(
    list(
        ml_df.columns
    )
)


print(
    "\nRows per train:"
)

print(
    ml_df[
        "train_number"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nJourneys per train:"
)

print(
    ml_df
    .groupby(
        "train_number"
    )["journey_date"]
    .nunique()
    .sort_index()
)


print(
    "\nMissing values:"
)

print(
    ml_df.isnull().sum()
)


print(
    "\nTarget arrival delay statistics:"
)

print(
    ml_df[
        "target_arrival_delay"
    ].describe()
)


print(
    "\nDelay change statistics:"
)

print(
    ml_df[
        "delay_change"
    ].describe()
)


print(
    "\nHistorical station delay statistics:"
)

print(
    ml_df[
        "historical_station_delay"
    ].describe()
)


print(
    "\nHistorical train delay statistics:"
)

print(
    ml_df[
        "historical_train_delay"
    ].describe()
)


print(
    "\nRoute progress statistics:"
)

print(
    ml_df[
        "route_progress"
    ].describe()
)


print(
    "\nStations remaining statistics:"
)

print(
    ml_df[
        "stations_remaining"
    ].describe()
)


print(
    "\nScheduled departure hour statistics:"
)

print(
    ml_df[
        "scheduled_departure_hour"
    ].describe()
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\n" + "=" * 60
)

print(
    "ENHANCED ML DATASET COMPLETE"
)

print(
    "=" * 60
)