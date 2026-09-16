import time
import threading
import requests

from src.alert_manager import get_all_subscriptions
from src.alert_checker import check_station_alerts


# ============================================================
# SCHEDULER SETTINGS
# ============================================================

CHECK_INTERVAL_SECONDS = 300
# 300 seconds = 5 minutes

API_BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# GET CURRENT PREDICTION
# ============================================================

def get_prediction(train_number):
    """
    Get the latest ML prediction from the running FastAPI app.
    """

    url = (
        f"{API_BASE_URL}/predict/{train_number}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# CHECK ALL ACTIVE ALERTS
# ============================================================

def check_all_alerts():
    """
    Check all active passenger subscriptions.
    """

    try:

        subscriptions = get_all_subscriptions()

        if not subscriptions:

            print(
                "[ALERT SCHEDULER] "
                "No subscriptions found."
            )

            return

        # ----------------------------------------------------
        # Find trains that still have unsent alerts
        # ----------------------------------------------------

        trains_to_check = set()

        for subscription in subscriptions:

            if subscription["alert_sent"] == 0:

                trains_to_check.add(
                    (
                        subscription["train_number"],
                        subscription["journey_date"]
                    )
                )

        if not trains_to_check:

            print(
                "[ALERT SCHEDULER] "
                "No pending alerts."
            )

            return

        # ----------------------------------------------------
        # Check each train
        # ----------------------------------------------------

        for train_number, journey_date in trains_to_check:

            try:

                prediction = get_prediction(
                    train_number
                )

                state = {

                    "next_station":
                        prediction[
                            "next_station"
                        ],

                    "next_station_name":
                        prediction[
                            "next_station_name"
                        ]
                }

                result = check_station_alerts(

                    train_number=train_number,

                    journey_date=journey_date,

                    state=state,

                    expected_arrival=
                        prediction[
                            "expected_arrival"
                        ]
                )

                print(
                    "[ALERT SCHEDULER]",
                    train_number,
                    result
                )

            except Exception as e:

                print(
                    "[ALERT SCHEDULER] "
                    f"Failed for train "
                    f"{train_number}: {e}"
                )

    except Exception as e:

        print(
            "[ALERT SCHEDULER] "
            f"Scheduler check failed: {e}"
        )


# ============================================================
# BACKGROUND LOOP
# ============================================================

def scheduler_loop():
    """
    Run the alert checker periodically.
    """

    print(
        "[ALERT SCHEDULER] "
        "Automatic alert checker started."
    )

    while True:

        check_all_alerts()

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


# ============================================================
# START SCHEDULER
# ============================================================

def start_alert_scheduler():
    """
    Start the scheduler in a background thread.
    """

    scheduler_thread = threading.Thread(
        target=scheduler_loop,
        daemon=True
    )

    scheduler_thread.start()

    return scheduler_thread