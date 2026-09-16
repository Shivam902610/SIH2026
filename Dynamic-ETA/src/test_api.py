import os
import json
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

response = requests.get(url, headers=headers, timeout=15)

print("Status Code:", response.status_code)

if response.ok:
    print("\n===== LIVE TRAIN DATA =====")

    data = response.json()
    print(json.dumps(data, indent=2))

else:
    print("\n===== ERROR =====")
    print(response.text)