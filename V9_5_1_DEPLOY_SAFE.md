# V9.5.1 Deploy-Safe

Fixes Render ephemeral SQLite resets during GitHub redeploys.

- Firestore is the permanent history source.
- On process startup, if SQLite is empty, the backend restores up to `HISTORY_LIMIT` rounds from Firestore **before** becoming ready.
- Firestore hydration streams documents instead of first building a full Python list, reducing peak RAM.
- The collector keeps retry/backfill behavior as a fallback.
- Existing V9.5 live simulator behavior is retained.

Expected deploy log:

```text
[STARTUP] Firestore -> SQLite restored=500 rounds
[STARTUP] backend ready with 500 cached rounds
```

Then collector should continue near 500 rather than `555 -> 1`. If `HISTORY_LIMIT=500`, restoring 500 instead of all 555 is expected. Set a larger `HISTORY_LIMIT` only if Render memory allows it.
