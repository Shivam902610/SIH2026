import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")


def send_sms(to_number: str):
    """
    Send a predefined Twilio trial SMS.
    """

    if not TWILIO_ACCOUNT_SID:
        raise RuntimeError("TWILIO_ACCOUNT_SID is missing from .env")

    if not TWILIO_AUTH_TOKEN:
        raise RuntimeError("TWILIO_AUTH_TOKEN is missing from .env")

    if not TWILIO_FROM_NUMBER:
        raise RuntimeError("TWILIO_FROM_NUMBER is missing from .env")

    client = Client(
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN
    )

    message = client.messages.create(
        body="sms_event_notifications",
        from_=TWILIO_FROM_NUMBER,
        to=to_number
    )

    print("SMS sent successfully!")
    print("Message SID:", message.sid)
    print("Status:", message.status)

    return message.sid