import os
import time
import requests

API = os.getenv(
    "SOURCE_API",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
)
BACKEND = os.getenv("BACKEND_URL", "").strip().rstrip("/")
SECRET = os.getenv("INGEST_SECRET", "").strip()

try:
    POLL = max(0.5, float(os.getenv("COLLECTOR_POLL_SECONDS", "3")))
    N = max(10, min(int(os.getenv("COLLECTOR_BACKFILL_SIZE", "20")), 100))
except ValueError as exc:
    raise SystemExit(f"Invalid collector numeric setting: {exc}") from exc

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
    }

    for attempt in range(1, 6):
        try:
            q = backend.post(
                BACKEND + "/api/ingest",
                json=body,
                headers={"X-Ingest-Secret": SECRET},
                timeout=30,
            )
            if q.status_code in (401, 403):
                raise SystemExit(
                    "Collector authentication failed. Check that INGEST_SECRET matches the backend."
                )
            if 400 <= q.status_code < 500 and q.status_code not in (408, 429):
                print(f"[REJECTED] {issue} HTTP {q.status_code}: {q.text[:200]}")
                seen.add(issue)
                return True
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
                    f"backend cache reset detected; replaying recent {N} rounds as fallback"
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
            print(f"[RETRY] {issue} {attempt}/5 {type(e).__name__}: {e} wait={wait}s")
            time.sleep(wait)

    print(f"[FAILED] {issue} - will retry from backfill window")
    return False

def validate_config():
    if not BACKEND:
        raise SystemExit("BACKEND_URL is required")
    if not BACKEND.startswith(("http://", "https://")):
        raise SystemExit("BACKEND_URL must start with http:// or https://")
    if not SECRET:
        raise SystemExit("INGEST_SECRET is required")


def poll_once():
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
    payload = r.json()
    if payload.get("code") not in (None, 0):
        raise RuntimeError(f"source returned code={payload.get('code')}")

    items = payload.get("data", {}).get("list", [])
    if not isinstance(items, list):
        raise RuntimeError("source response data.list is not a list")
    items = sorted(items, key=key)  # oldest -> newest

    for x in items:
        issue = str(x.get("issueNumber", ""))
        if not issue or issue in seen:
            continue

        try:
            number = int(x.get("number"))
        except (TypeError, ValueError):
            print(f"[SKIP] {issue} invalid number")
            seen.add(issue)
            continue
        if number < 0 or number > 9:
            print(f"[SKIP] {issue} number out of range")
            seen.add(issue)
            continue

        if not post_round(x):
            # Keep ordering. Next poll will retry this round first.
            break

    # Keep RAM bounded and retain the numerically newest issues.
    if len(seen) > 500:
        newest = sorted(
            seen,
            key=lambda value: (value.isdigit(), int(value) if value.isdigit() else value),
        )[-250:]
        seen.clear()
        seen.update(newest)


def main():
    validate_config()
    print(f"WinGo V9.5.1 collector -> {BACKEND}")
    print(f"Recovery/backfill window: {N} rounds")

    while True:
        try:
            poll_once()
        except SystemExit:
            raise
        except Exception as e:
            print("[SOURCE ERROR]", type(e).__name__, e)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
