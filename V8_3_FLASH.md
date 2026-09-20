# WinGo V8.3 Flash

Selective-prediction experimental build.

Changes:
- Strict confidence gate: confidence < 0.85 => SKIP.
- CHOPPY regime => SKIP.
- Rising entropy trend => SKIP.
- Requires 3/3 feature agreement.
- Requires >=70% model-family Big/Small agreement.
- Explicit regime priors:
  - stable: Markov + N-gram dominant
  - choppy: EMA + Z-score dominant
  - normal: frequency + momentum dominant
- Adds triadic Color|Size|Parity co-occurrence diagnostics.
- Keeps Firestore restart persistence and V8.3.1 low-memory fixes.

Important:
`confidence` is an evidence score, not a calibrated 85% probability and not a
guarantee of future accuracy. Color/Size/Parity are derived from the same digit,
so their co-occurrence is structural and should not be treated as independent
evidence.

Render:
ENABLE_SOURCE_WORKER=0
HISTORY_LIMIT=300
BACKFILL_TARGET=300

Start:
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
