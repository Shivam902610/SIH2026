from sms_service import send_sms


# Use your own verified Twilio number here.
TO_NUMBER = "+919026105875"


try:
    send_sms(TO_NUMBER)

except Exception as e:
    print("\nSMS FAILED")
    print("Error:", e)