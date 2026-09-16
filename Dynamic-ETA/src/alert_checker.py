from datetime import datetime

import pandas as pd

from src.alert_manager import (
    get_active_subscriptions,
    mark_alert_sent,
)

from src.alert_service import (
    create_station_alert,
)

from src.sms_service import (
    send_sms,
)


def check_station_alerts(
    train_number: str,
    journey_date: str,
    state: dict,
    expected_arrival: str | None,
):
    """
    Check active passenger subscriptions for a train.

    An alert is triggered only when:
    1. The passenger's selected station is the current next station.
    2. The predicted arrival is within the passenger's alert window.
    3. The alert has not already been sent.
    """

    # --------------------------------------------------------
    # CHECK EXPECTED ARRIVAL
    # --------------------------------------------------------

    if not expected_arrival:

        return {
            "success": True,
            "alerts_checked": 0,
            "alerts_sent": 0,
            "message": "Expected arrival is unavailable.",
            "results": []
        }

    try:

        arrival_time = pd.to_datetime(
            expected_arrival
        )

    except Exception:

        return {
            "success": False,
            "alerts_checked": 0,
            "alerts_sent": 0,
            "message": "Invalid expected arrival time.",
            "results": []
        }

    # --------------------------------------------------------
    # CURRENT TIME
    # --------------------------------------------------------

    if arrival_time.tzinfo is not None:

        current_time = pd.Timestamp.now(
            tz=arrival_time.tz
        )

    else:

        current_time = pd.Timestamp.now()

    # --------------------------------------------------------
    # GET ACTIVE SUBSCRIPTIONS
    # --------------------------------------------------------

    subscriptions = get_active_subscriptions(
        train_number=train_number,
        journey_date=journey_date
    )

    results = []

    alerts_sent = 0

    # --------------------------------------------------------
    # CHECK EACH SUBSCRIPTION
    # --------------------------------------------------------

    for subscription in subscriptions:

        station_code = subscription[
            "station_code"
        ]

        station_name = subscription[
            "station_name"
        ]

        # ----------------------------------------------------
        # ONLY CHECK THE CURRENT NEXT STATION
        # ----------------------------------------------------

        if station_code != state[
            "next_station"
        ]:

            results.append({

                "subscription_id":
                    subscription["id"],

                "station":
                    station_code,

                "should_alert":
                    False,

                "reason":
                    "Selected station is not the current next station."
            })

            continue

        # ----------------------------------------------------
        # CREATE ALERT DECISION
        # ----------------------------------------------------

        alert_result = create_station_alert(

            train_number=train_number,

            station_code=station_code,

            station_name=station_name,

            expected_arrival=
                arrival_time.to_pydatetime(),

            current_time=
                current_time.to_pydatetime(),

            alert_type=
                subscription["alert_type"],

            alert_before_minutes=
                subscription[
                    "alert_before_minutes"
                ]
        )

        # ----------------------------------------------------
        # ALERT NOT REQUIRED YET
        # ----------------------------------------------------

        if not alert_result[
            "should_alert"
        ]:

            results.append({

                "subscription_id":
                    subscription["id"],

                "station":
                    station_code,

                "should_alert":
                    False,

                "reason":
                    "Station is not within the alert window.",

                "minutes_until_arrival":
                    alert_result[
                        "minutes_until_arrival"
                    ]
            })

            continue

        # ----------------------------------------------------
        # SEND SMS
        # ----------------------------------------------------

        try:

            send_sms(
                subscription[
                    "phone_number"
                ]
            )

            # -----------------------------------------------
            # MARK ALERT AS SENT
            # -----------------------------------------------

            mark_alert_sent(
                subscription["id"]
            )

            alerts_sent += 1

            results.append({

                "subscription_id":
                    subscription["id"],

                "station":
                    station_code,

                "should_alert":
                    True,

                "sms_sent":
                    True,

                "message":
                    alert_result["message"],

                "minutes_until_arrival":
                    alert_result[
                        "minutes_until_arrival"
                    ]
            })

        except Exception as e:

            results.append({

                "subscription_id":
                    subscription["id"],

                "station":
                    station_code,

                "should_alert":
                    True,

                "sms_sent":
                    False,

                "error":
                    str(e),

                "message":
                    alert_result["message"]
            })

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "success": True,

        "alerts_checked":
            len(subscriptions),

        "alerts_sent":
            alerts_sent,

        "next_station":
            state[
                "next_station"
            ],

        "next_station_name":
            state[
                "next_station_name"
            ],

        "expected_arrival":
            expected_arrival,

        "results":
            results
    }