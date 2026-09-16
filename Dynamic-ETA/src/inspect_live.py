import os
import json
import requests

from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("RAILRADAR_API_KEY")

TRAIN_NUMBER = "12919"

url = (
    f"https://api.railradar.in/"
    f"v1/trains/{TRAIN_NUMBER}/live"
)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}


response = requests.get(
    url,
    headers=headers,
    timeout=20
)

print("Status:", response.status_code)

if response.status_code != 200:
    print(response.text)
    raise SystemExit


data = response.json()["data"]


print("\n================ CURRENT LOCATION ================")

print(
    json.dumps(
        data.get("currentLocation"),
        indent=2
    )
)


print("\n================ NEXT HALT ================")

print(
    json.dumps(
        data.get("nextHalt"),
        indent=2
    )
)