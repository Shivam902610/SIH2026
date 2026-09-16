from datetime import datetime, timedelta

from alert_service import create_station_alert


# Simulated current time
current_time = datetime.now()

# Simulate a train arriving 15 minutes from now
expected_arrival = current_time + timedelta(minutes=15)


result = create_station_alert(
    train_number="12919",
    station_code="NDLS",
    station_name="New Delhi",
    expected_arrival=expected_arrival,
    current_time=current_time,
    alert_type="boarding",
    alert_before_minutes=15,
)


print("\n===== TRAINETA ALERT TEST =====")
print("Should alert:", result["should_alert"])
print("Train:", result["train_number"])
print("Station:", result["station_name"])
print("Minutes remaining:", result["minutes_until_arrival"])
print("Message:", result.get("message"))