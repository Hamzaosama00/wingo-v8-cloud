# V8 Firebase deployment

1. Firebase Console: create a project, then Build > Firestore Database > Create database.
2. Deploy `firebase/firestore.rules`. Client reads are public for the dashboard; client writes are denied.
3. Backend: deploy `backend/` to a Python host. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Backend env: `INGEST_SECRET`, `FIREBASE_PROJECT_ID`, `CORS_ORIGINS`. On Google Cloud use Application Default Credentials. On another host, mount a private Firebase service-account JSON and set `GOOGLE_APPLICATION_CREDENTIALS` to its path. Never commit that JSON.
5. Collector: deploy `collector/` as an always-on worker. Set `BACKEND_URL`, the same `INGEST_SECRET`, and optionally `COLLECTOR_POLL_SECONDS=3`. It reads only already-published/completed history. If the source returns 403 from a provider, do not bypass it; use an authorized source/network.
6. Frontend: deploy `frontend/` to Netlify/static hosting. Open it and enter the backend URL.
7. Verify backend `/` shows version 8.0.0 and `/health` is online.
8. Existing SQLite history is not automatically uploaded to Firestore; new ingested rounds are mirrored there.
