# V9.4 Flash

Changes from V9.3:
- Minimum history reduced from 80 to 10 completed rounds.
- V9.4 starts producing a directional Big/Small model output after 10 rounds.
- Weak evidence is explicitly reported as `decision: NO EDGE` / `signal: SKIP`.
- Raw probability is exposed as `raw_p_big`.
- Isotonic calibration is support-aware: with only a small walk-forward sample it cannot dominate the raw model.
- Calibration weight gradually increases from 0 after 30 OOS samples to full weight around 100 samples.
- Simulator can start with 10 training rounds and requires `max_rounds >= 10`.
- `/api/v9.4/flash` and `/api/v9.4/simulate` added; old V9.3 routes remain as compatibility aliases.

Important: 10 rounds is enough to generate an experimental output, not enough to establish predictive accuracy. SKIP/NO EDGE is intentional when evidence is weak.
