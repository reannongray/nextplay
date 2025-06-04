# keep_alive.py
import requests
import time

# Replace with your actual Streamlit app URL
URL = "https://nextplay.streamlit.app"

def ping():
    try:
        response = requests.get(URL)
        print(f"[{time.ctime()}] Pinged {URL} - Status: {response.status_code}")
    except Exception as e:
        print(f"[{time.ctime()}] Failed to ping {URL}: {e}")

if __name__ == "__main__":
    ping()
