# WinGo V9.5.1 Flash Live

Big/Small research API, live paper simulator, collector, and static dashboard.
Predictions are experimental and are not guaranteed outcomes.

## Backend

```text
pip install -r backend/requirements.txt
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

Required environment variable: `INGEST_SECRET`.

Recommended production variables: `FIREBASE_PROJECT_ID`,
`GOOGLE_APPLICATION_CREDENTIALS`, `CORS_ORIGINS`, `HISTORY_LIMIT`, and
`DB_PATH`. The backend restores Firestore history into its SQLite cache at
startup when the cache is empty.

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

## Checks

```text
python -m unittest backend/test_smoke.py collector/test_collector.py -v
```

The checks cover ingest authentication and validation, duplicate/conflicting
results, live prediction settlement, the backend-poller/collector race, source
ordering, and collector header authentication.
