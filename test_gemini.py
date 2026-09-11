import requests
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
print(f"API Key: {GEMINI_API_KEY[:5]}...")
payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "Hello"}]}
    ]
}
resp = requests.post(url, json=payload)
print(resp.status_code, resp.text)
