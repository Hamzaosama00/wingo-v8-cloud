# Persistent history setup

Render's local SQLite filesystem is ephemeral. This build uses Firestore as the
permanent source of truth for collected rounds.

Required Render environment:
- FIREBASE_PROJECT_ID=wingo-9aa59
- GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/firebase-service-account.json
- existing INGEST_SECRET

Keep the Firebase service-account JSON as a Render Secret File named:
firebase-service-account.json

On a fresh deployment, the first `/api/ingest` call detects an empty SQLite DB,
loads recent rounds from Firestore, then processes the new round. The API response
includes `restored_from_firestore`.

IMPORTANT: Data collected by an older deployment can only be recovered if it was
already written to Firestore. Old data that existed only on Render's ephemeral
SQLite disk cannot be reconstructed after that disk has already been destroyed.
