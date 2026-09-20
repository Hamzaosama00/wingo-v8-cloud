# WinGo V9 Research Engine

V9 is a Big/Small research/backtesting engine. It does not claim guaranteed prediction of a fair RNG.

## V9 changes
- Leak-free features: round N uses only earlier rounds.
- Streak/run features, transition probabilities, 20/50/100/300 multi-scale stats.
- Number/parity/modulo features and regime/entropy features.
- Issue modulo features are explicitly experimental.
- Ensemble: L2 Logistic Regression + HistGradientBoosting + order 1–3 Markov.
- Purged block walk-forward evaluation with embargo.
- Isotonic probability calibration from out-of-sample predictions.
- Confidence/coverage curve and fixed signal threshold.
- Feature ablation and logistic contribution display.
- Firestore persistence over REST + google-auth; no firebase-admin/gRPC.
- No internal source worker. ESP8266/Python collector remains the only ingest path.

## Firestore
Existing `rounds` is reused. V9 writes predictions to `v9_predictions`, so old V8 predictions stay untouched.

## Render
Root Directory: `backend`

Build:
```bash
pip install -r requirements.txt
```

Start:
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
```

Recommended env:
```text
INGEST_SECRET=<fresh secret>
FIREBASE_PROJECT_ID=wingo-9aa59
GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/firebase-service-account.json
CORS_ORIGINS=*
HISTORY_LIMIT=1200
V9_MIN_TRAIN=100
V9_BACKTEST_BLOCK=50
V9_EMBARGO=20
V9_SIGNAL_THRESHOLD=0.60
```
Keep the existing Render secret file `firebase-service-account.json`.

Rotate `INGEST_SECRET` if it has ever been pasted into chat or committed anywhere.

## ESP8266 compatibility
Same endpoint and JSON remain supported:
```text
POST /api/ingest
{"issue":"...","number":2,"color":"red","secret":"..."}
```
Treat the response as accepted only when the JSON body contains `"ok": true`; HTTP 200 by itself is not enough.

## Dashboard
Deploy `frontend/` on Netlify/Vercel/static hosting. It defaults to `https://wingo-v8-cloud.onrender.com`.
You can override it with:
```text
index.html?api=https://your-backend.example.com
```

## Optional Python collector
```bash
cd collector
pip install -r requirements.txt
export BACKEND_URL="https://wingo-v8-cloud.onrender.com"
export INGEST_SECRET="<same Render secret>"
export COLLECTOR_POLL_SECONDS="3"
export COLLECTOR_BACKFILL_WINDOW="20"
python collector.py
```

Interpret `PREDICT` as a research threshold being met and `SKIP` as insufficient evidence. Always compare OOS accuracy with coverage and Brier score.
