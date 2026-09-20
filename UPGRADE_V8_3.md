# V8.3 improvements

- Strict confidence/noise gate: below evidence threshold => SKIP / no actionable signal.
- Dynamic lookback: stable up to 200, normal 100, choppy 40.
- Higher-order Markov up to order 3 with support fallback.
- Z-score frequency deviation diagnostic with weak mean-reversion prior.
- Lag-5 / lag-10 empirical evidence model.
- Momentum indicator for recent size persistence.
- Walk-forward scoring by regime.
- Weighted log-loss penalizes confident mistakes more heavily.
- Existing Firestore persistence, outage backfill, and prediction-vs-actual compare panel retained.

Important: the displayed confidence is a calibrated evidence score, not a promise of accuracy or a guaranteed probability. Random/independent outcomes may remain unpredictable.
