# WinGo V9.5 Flash Live

Changes from V9.4.1:

- Fixed chronology bug: `load_history()` already returns oldest -> newest, so V9.4.1 was reversing the sequence a second time inside Flash and the simulator.
- Prediction direction now generates after 10 completed rounds on every round. `SKIP` no longer blocks the paper simulator.
- Confidence is not artificially inflated. A low edge can still honestly sit around 50-55%.
- Added evidence `quality` (`LOW`, `MEDIUM`, `STRONG`) and warnings instead of suppressing predictions.
- Walk-forward metrics now evaluate all generated directions.
- Historical simulator chronology fixed and no longer skips valid predictions.
- Added backend live paper simulator: each new completed ingest settles the pending prediction and opens the next one automatically.
- New endpoint: `/api/v9.5/flash`
- New endpoint: `/api/v9.5/live-sim`
- New alias: `/api/v9.5/simulate`
- Frontend polls every 3 seconds and shows pending prediction plus live win/loss history.

The live simulator uses paper results only; it has no wallet or real-bet integration.
