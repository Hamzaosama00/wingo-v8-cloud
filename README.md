# WinGo V9.5.3 Flash Live

Big/Small research API, live paper simulator, collector, and static dashboard.
Predictions are experimental and are not guaranteed outcomes.

## Backend

```text
pip install -r backend/requirements.txt
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

Required environment variable for external ingest: `INGEST_SECRET`.

Set `ENABLE_INTERNAL_COLLECTOR=true` (the default) to let the Render backend
poll completed results itself every `POLL_SECONDS` (default: 3 seconds). The
worker starts only after startup Firestore restoration. If the source returns
HTTP 403 from Render, no bypass is attempted: set
`ENABLE_INTERNAL_COLLECTOR=false` and run the existing external/Termux
collector instead. `/health` reports the selected collector mode, liveness,
last source status, and last successful source request.

Recommended production variables: `FIREBASE_PROJECT_ID`,
`GOOGLE_APPLICATION_CREDENTIALS`, `CORS_ORIGINS`, `HISTORY_LIMIT`, and
`DB_PATH`. The backend restores Firestore history into its SQLite cache at
startup when the cache is empty.

For laptop-only SQLite mode, set `FIRESTORE_ENABLED=false`. Without an explicit
flag, Firestore is enabled only when a Firebase project ID or credentials path
is configured, preventing credential timeouts during local development.

Network access uses `httpx`; the collector uses `AsyncClient`. SQLite-backed
model work stays in the dedicated worker/FastAPI threadpool boundary, while the
health probe uses `aiosqlite` so its database check does not block the event
loop. Structured Python logging replaces ad-hoc prints; set `LOG_LEVEL` when a
different verbosity is needed.

`/health` now performs a real database query and reports DB latency, row count,
collector mode, worker liveness, and time since the last poll. When internal
collection is disabled, the external-ingest mode does not require a worker.

Walk-forward scoring freezes the sequence and trains only on the strict prefix
before each target. Isotonic calibration requires 60 out-of-sample points by
default; tune with `V93_ISOTONIC_MIN_SAMPLES` (minimum accepted value: 40).
Probability clipping and divisions use shared numerical-safety helpers.

## Collector

Run `python collector/collector.py` with:

- `BACKEND_URL` set to the deployed backend URL.
- `INGEST_SECRET` set to exactly the same value as the backend.
- Optional `SOURCE_API`, `COLLECTOR_POLL_SECONDS`, and
  `COLLECTOR_BACKFILL_SIZE` settings.

The collector sends credentials in `X-Ingest-Secret`, retries transient
failures, replays a bounded recovery window, and stops with a clear message on
authentication/configuration errors.

## Dashboard

Deploy `frontend/` as a static site. If the backend URL changes, enter the new
URL in the dashboard's **Backend URL** field and select **Connect**. The value
is saved in the browser; `?api=https://example.com` is also supported.

The dashboard includes a browser-local fake wallet with a round-by-round paper
ledger. It never connects to a real wallet or places real bets. A protected
**Reset Data** button can clear SQLite and configured Firestore collections.
Set a strong `ADMIN_RESET_SECRET` on the backend; the reset endpoint remains
disabled when this variable is empty and requires the exact confirmation phrase
plus the secret on every request.

## Checks

```text
python -m unittest backend/test_smoke.py collector/test_collector.py -v
```

The checks cover ingest authentication and validation, duplicate/conflicting
results, live prediction settlement, the backend-poller/collector race, source
ordering, and collector header authentication.
