# V8.3.1 Render stability fix

Main fixes:
- Firestore is authoritative; Render SQLite is disposable cache.
- Rounds are streamed from Firestore instead of `list(stream())`, reducing RAM spikes.
- Prediction ledger is restored from Firestore too. A restart therefore does not replace an older forecast with a newly computed one after the actual result exists.
- Verified prediction results are synced back to Firestore.
- Render's built-in source polling worker is OFF by default. Your Termux collector is the single ingestion path, avoiding duplicate polling/races.
- Default model history reduced from 500 to 300 rounds to lower peak memory.
- `/api/memory` reports current process RAM.

Render environment:
- ENABLE_SOURCE_WORKER=0
- HISTORY_LIMIT=300
- BACKFILL_TARGET=300
- Keep existing FIREBASE_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS and INGEST_SECRET.
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`

Important:
Do not delete Firestore `rounds` or `predictions`; they are now the persistent ledger.
The Termux collector continues to send only already-published completed rounds.
