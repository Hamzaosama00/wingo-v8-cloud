import os
import time
import requests

API = os.getenv(
    "SOURCE_API",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
)
BACKEND = os.environ["BACKEND_URL"].rstrip("/")
SECRET = os.environ["INGEST_SECRET"]
POLL = float(os.getenv("COLLECTOR_POLL_SECONDS", "3"))
N = max(10, min(int(os.getenv("COLLECTOR_BACKFILL_SIZE", "20")), 100))

H = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Origin": "https://www.92pak8.com",
    "Referer": "https://www.92pak8.com/",
}

source = requests.Session()
source.headers.update(H)
backend = requests.Session()

seen = set()
last_backend_count = None

def key(x):
    try:
        return int(x.get("issueNumber", "0"))
    except Exception:
        return 0

def post_round(x):
    global last_backend_count, seen

    issue = str(x.get("issueNumber", ""))
    body = {
        "issue": issue,
        "number": int(x["number"]),
        "color": str(x.get("color", "")),
        "secret": SECRET,
    }

    for attempt in range(1, 6):
        try:
            q = backend.post(BACKEND + "/api/ingest", json=body, timeout=30)
            q.raise_for_status()
            data = q.json()

            if not data.get("ok"):
                raise RuntimeError(data.get("error", "ingest rejected"))

            count = int(data.get("stored_rounds") or 0)
            restored = int(data.get("restored_from_firestore") or 0)

            # Detect Render restart / lost ephemeral SQLite cache.
            # Do not mistake a backend counter reset for normal progress.
            reset = (
                last_backend_count is not None
                and count > 0
                and count < max(1, last_backend_count - 2)
            )

            if reset:
                print(
                    f"[BACKEND RESTART] stored {last_backend_count} -> {count}; "
                    f"replaying recent {N} completed rounds"
                )
                # Force the current source page to be replayed on the next loop.
                # Backend deduplication makes this safe.
                seen.clear()

            last_backend_count = max(count, last_backend_count or 0) if not reset else count
            seen.add(issue)

            extra = f" restored={restored}" if restored else ""
            print(f"[OK] {issue} backend={count}{extra}")
            return True

        except Exception as e:
            wait = min(3 * attempt, 15)
            print(f"[RETRY] {issue} {attempt}/5 {type(e).__name__} wait={wait}s")
            time.sleep(wait)

    print(f"[FAILED] {issue} - will retry from backfill window")
    return False

print(f"WinGo V9.4.1 collector -> {BACKEND}")
print(f"Recovery/backfill window: {N} rounds")

while True:
    try:
        r = source.get(
            API,
            params={
                "pageNo": 1,
                "pageSize": N,
                "ts": int(time.time() * 1000),
            },
            timeout=15,
        )
        r.raise_for_status()

        items = r.json().get("data", {}).get("list", [])
        items = sorted(items, key=key)  # oldest -> newest

        for x in items:
            issue = str(x.get("issueNumber", ""))
            if not issue or issue in seen:
                continue

            if not post_round(x):
                # Keep ordering. Next poll will retry this round first.
                break

        # Keep RAM bounded.
        if len(seen) > 500:
            seen = set(sorted(seen)[-250:])

    except Exception as e:
        print("[SOURCE ERROR]", type(e).__name__, e)

    time.sleep(POLL)
