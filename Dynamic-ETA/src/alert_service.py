from datetime import datetime, timedelta


def create_station_alert(
    train_number: str,
    station_code: str,
    station_name: str,
    expected_arrival: datetime,
    current_time: datetime,
    alert_type: str = "boarding",
    alert_before_minutes: int = 15,
):
    """
    Decide whether a passenger should receive a station alert.

    Parameters:
        train_number: Train number
        station_code: Passenger's selected station code
        station_name: Passenger's selected station name
        expected_arrival: ML-predicted arrival time
        current_time: Current time
        alert_type: "boarding", "drop", or "both"
        alert_before_minutes: Send alert this many minutes before arrival

    Returns:
        Dictionary containing alert information.
    """

    # Calculate how many minutes remain until the train arrives.
    minutes_until_arrival = (
        expected_arrival - current_time
    ).total_seconds() / 60

    # Check whether the station is approaching.
    should_alert = (
        0 <= minutes_until_arrival <= alert_before_minutes
    )

    if not should_alert:
        return {
            "should_alert": False,
            "train_number": train_number,
            "station_code": station_code,
            "station_name": station_name,
            "minutes_until_arrival": round(minutes_until_arrival, 1),
        }

    # Create the passenger message.
    if alert_type == "boarding":
        message = (
            f"TrainETA Alert: Train {train_number} is approaching "
            f"{station_name} ({station_code}). "
            f"Expected arrival in approximately "
            f"{round(minutes_until_arrival)} minutes. "
            f"Please get ready to board."
        )

    elif alert_type == "drop":
        message = (
            f"TrainETA Alert: Train {train_number} is approaching "
            f"{station_name} ({station_code}). "
            f"Expected arrival in approximately "
            f"{round(minutes_until_arrival)} minutes. "
            f"Please get ready to get off."
        )

    else:
        message = (
            f"TrainETA Alert: Train {train_number} is approaching "
            f"{station_name} ({station_code}). "
            f"Expected arrival in approximately "
            f"{round(minutes_until_arrival)} minutes. "
            f"Please get ready."
        )

    return {
        "should_alert": True,
        "train_number": train_number,
        "station_code": station_code,
        "station_name": station_name,
        "minutes_until_arrival": round(minutes_until_arrival, 1),
        "alert_type": alert_type,
        "message": message,
    }