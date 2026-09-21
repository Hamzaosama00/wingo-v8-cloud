# V9.5 Collector Recovery Fix

The collector now:
- retries transient 502/backend errors with bounded backoff;
- keeps a 20-round recovery window by default;
- processes completed rounds oldest -> newest;
- detects when Render's local `stored_rounds` counter suddenly drops after a restart;
- clears its local `seen` cache after that reset so the recent page is replayed;
- relies on backend issue deduplication, so replaying recent rounds is safe;
- prints `backend=<count>` and `restored=<count>` separately instead of making a restart look like normal counting.

Example:
    [OK] ...52067 backend=17
    [RETRY] ...52068 1/5 HTTPError wait=3s
    [OK] ...52068 backend=18

If Render actually restarted:
    [BACKEND RESTART] stored 17 -> 1; replaying recent 20 completed rounds

The next poll re-sends the recent completed window to rebuild the cache even if Firestore hydration was unavailable.
