import asyncio
import logging
import os
import time

import httpx

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("wingo.collector")
logging.getLogger("httpx").setLevel(os.getenv("HTTPX_LOG_LEVEL", "WARNING").upper())

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

seen = set()
last_backend_count = None

def key(x):
    try:
        return int(x.get("issueNumber", "0"))
    except Exception:
        return 0

async def post_round(x, backend: httpx.AsyncClient):
    global last_backend_count, seen

    issue = str(x.get("issueNumber", ""))
    body = {
        "issue": issue,
        "number": int(x["number"]),
        "color": str(x.get("color", "")),
    }

    for attempt in range(1, 6):
        try:
            q = await backend.post(
                BACKEND + "/api/ingest",
                json=body,
                headers={"X-Ingest-Secret": SECRET},
            )
            if q.status_code in (401, 403):
                raise SystemExit(
                    "Collector authentication failed. Check that INGEST_SECRET matches the backend."
                )
            if 400 <= q.status_code < 500 and q.status_code not in (408, 429):
                logger.error("rejected issue=%s HTTP %s: %s", issue, q.status_code, q.text[:200])
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
                logger.warning(
                    "backend restart stored=%s -> %s; replaying recent %s rounds",
                    last_backend_count,
                    count,
                    N,
                )
                # Force the current source page to be replayed on the next loop.
                # Backend deduplication makes this safe.
                seen.clear()

            last_backend_count = max(count, last_backend_count or 0) if not reset else count
            seen.add(issue)

            logger.info("ingested issue=%s backend=%s restored=%s", issue, count, restored)
            return True

        except Exception as e:
            wait = min(3 * attempt, 15)
            logger.warning(
                "retry issue=%s attempt=%s/5 error=%s: %s wait=%ss",
                issue,
                attempt,
                type(e).__name__,
                e,
                wait,
            )
            await asyncio.sleep(wait)

    logger.error("failed issue=%s; will retry from backfill window", issue)
    return False

def validate_config():
    if not BACKEND:
        raise SystemExit("BACKEND_URL is required")
    if not BACKEND.startswith(("http://", "https://")):
        raise SystemExit("BACKEND_URL must start with http:// or https://")
    if not SECRET:
        raise SystemExit("INGEST_SECRET is required")


async def poll_once(source: httpx.AsyncClient, backend: httpx.AsyncClient):
    r = await source.get(
        API,
        params={
            "pageNo": 1,
            "pageSize": N,
            "ts": int(time.time() * 1000),
        },
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
            logger.warning("skip issue=%s invalid number", issue)
            seen.add(issue)
            continue
        if number < 0 or number > 9:
            logger.warning("skip issue=%s number out of range", issue)
            seen.add(issue)
            continue

        if not await post_round(x, backend):
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


async def run_collector():
    validate_config()
    logger.info("WinGo V9.5.1 collector -> %s", BACKEND)
    logger.info("recovery/backfill window=%s rounds", N)

    async with (
        httpx.AsyncClient(headers=H, timeout=15.0, follow_redirects=True) as source,
        httpx.AsyncClient(timeout=30.0, follow_redirects=True) as backend,
    ):
        while True:
            try:
                await poll_once(source, backend)
            except SystemExit:
                raise
            except Exception as exc:
                logger.error("source error %s: %s", type(exc).__name__, exc)
            await asyncio.sleep(POLL)


def main():
    asyncio.run(run_collector())


if __name__ == "__main__":
    main()
