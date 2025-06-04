# keep_alive.py
import requests
import time

URL = "https://nextplaygaming.streamlit.app/"

with open("ping_log.txt", "a") as log:
    try:
        response = requests.get(URL)
        msg = f"[{time.ctime()}] Pinged {URL} - Status: {response.status_code}"
        print(msg)
        log.write(msg + "\n")
    except Exception as e:
        msg = f"[{time.ctime()}] Failed to ping {URL}: {e}"
        print(msg)
        log.write(msg + "\n")
