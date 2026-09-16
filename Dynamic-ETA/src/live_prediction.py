import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAILRADAR_API_KEY")
TRAIN_NUMBER = "12919"

url = f"https://api.railradar.in/v1/trains/{TRAIN_NUMBER}/live"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

print("=" * 60)
print("DYNAMIC ETA - LIVE TRAIN DATA")
print("=" * 60)

response = requests.get(
    url,
    headers=headers,
    timeout=15
)

print("Status Code:", response.status_code)

if response.ok:
    data = response.json()

    print("\nLive train data received successfully!")

    print("\nTop-level keys:")
    print(data.keys())

    # Save response locally so we can inspect it
    with open("data/live_response.json", "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2)

    print("\nLive response saved to:")
    print("data/live_response.json")

else:
    print("\nAPI ERROR:")
    print(response.text)