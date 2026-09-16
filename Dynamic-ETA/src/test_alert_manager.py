from alert_manager import (
    create_subscription,
    get_active_subscriptions,
    mark_alert_sent,
    get_all_subscriptions,
)


print("\n===== CREATING TEST SUBSCRIPTION =====")

result = create_subscription(
    train_number="12919",
    journey_date="2026-09-15",
    station_code="NDLS",
    station_name="New Delhi",
    phone_number="+919026105875",
    alert_type="boarding",
    alert_before_minutes=15,
)

print(result)


print("\n===== ACTIVE SUBSCRIPTIONS =====")

subscriptions = get_active_subscriptions(
    train_number="12919",
    journey_date="2026-09-15",
)

for subscription in subscriptions:
    print(subscription)


if subscriptions:

    subscription_id = subscriptions[0]["id"]

    print("\n===== MARKING ALERT AS SENT =====")

    mark_alert_sent(subscription_id)

    print("Alert marked as sent.")


print("\n===== ACTIVE SUBSCRIPTIONS AFTER SENDING =====")

subscriptions = get_active_subscriptions(
    train_number="12919",
    journey_date="2026-09-15",
)

for subscription in subscriptions:
    print(subscription)


print("\n===== ALL SUBSCRIPTIONS =====")

for subscription in get_all_subscriptions():
    print(subscription)