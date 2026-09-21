# V9.5.3 Render + Firebase deployment

1. Firebase Console: create a project, then Build > Firestore Database > Create database.
2. Deploy `firebase/firestore.rules`. Client reads are public for the dashboard; client writes are denied.
3. Backend: deploy `backend/` to a Python host. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Backend env: `INGEST_SECRET`, a strong `ADMIN_RESET_SECRET`, `FIREBASE_PROJECT_ID`, `CORS_ORIGINS`, `ENABLE_INTERNAL_COLLECTOR=true`, and optionally `POLL_SECONDS=3`. On Google Cloud use Application Default Credentials. On another host, mount a private Firebase service-account JSON and set `GOOGLE_APPLICATION_CREDENTIALS` to its path. Never commit either secret or that JSON.
5. The built-in collector reads only published/completed history. Verify `/health` reports `source_status: 200` and worker mode `internal`. If Render receives source HTTP 403, do not bypass it: set `ENABLE_INTERNAL_COLLECTOR=false` and run `collector/` on the existing authorized Termux/network with `BACKEND_URL` and matching `INGEST_SECRET`.
6. Frontend: deploy `frontend/` to Netlify/static hosting. Open it and enter the backend URL.
7. Verify backend `/` shows version 9.5.3 and `/health` is online.
8. Startup restores Firestore before enabling the internal collector. The protected dashboard reset clears SQLite and Firestore only after the exact admin secret is supplied.
