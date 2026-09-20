import os
import time
import requests

SOURCE_API = os.getenv(
    "SOURCE_API",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
)
BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")
INGEST_SECRET = os.environ["INGEST_SECRET"]
POLL = float(os.getenv("COLLECTOR_POLL_SECONDS", "3"))
BACKFILL_SIZE = max(5, min(int(os.getenv("COLLECTOR_BACKFILL_SIZE", "20")), 100))

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": os.getenv("WINGO_ORIGIN", "https://www.92pak8.com"),
    "Referer": os.getenv("WINGO_REFERER", "https://www.92pak8.com/"),
}

session = requests.Session()
session.headers.update(HEADERS)
delivered = set()

def issue_key(x):
    s = str(x.get("issueNumber", ""))
    try:
        return (0, int(s))
    except Exception:
        return (1, s)

def send_round(item):
    issue = str(item["issueNumber"])
    payload = {
        "issue": issue,
        "number": int(item["number"]),
        "color": str(item.get("color", "")),
        "secret": INGEST_SECRET,
    }
    # Retry transient backend failures with increasing delay.
    last_exc = None
    for attempt in range(5):
        try:
            r = requests.post(BACKEND_URL + "/api/ingest", json=payload, timeout=30)
            r.raise_for_status()
            body = r.json()
            if not body.get("ok"):
                raise RuntimeError(body.get("error", "ingest rejected"))
            print(
                "[OK]", issue,
                "stored=", body.get("stored_rounds"),
                "restored=", body.get("restored_from_firestore", 0),
            )
            delivered.add(issue)
            return True
        except Exception as exc:
            last_exc = exc
            time.sleep(min(5 * (attempt + 1), 20))
    print("[INGEST ERROR]", issue, type(last_exc).__name__, last_exc)
    return False

print("WinGo V8.3.1 stable collector:", BACKEND_URL)
print("Backfill window:", BACKFILL_SIZE)

while True:
    try:
        r = session.get(
            SOURCE_API,
            params={
                "pageNo": 1,
                "pageSize": BACKFILL_SIZE,
                "ts": int(time.time() * 1000),
            },
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("data", {}).get("list", [])

        # API normally returns newest first. Oldest->newest means an outage can
        # recover every still-visible completed round, not just the newest one.
        for item in sorted(items, key=issue_key):
            issue = str(item.get("issueNumber", ""))
            if not issue or issue in delivered:
                continue
            if not send_round(item):
                # Preserve order: retry this missing round on the next poll.
                break

        # Bound memory; Firestore/backend deduplication remains authoritative.
        if len(delivered) > 500:
            delivered = set(sorted(delivered)[-250:])

    except Exception as exc:
        print("[SOURCE ERROR]", type(exc).__name__, exc)

    time.sleep(POLL)
