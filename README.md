# WinGo V9.2 Pro Experimental

Adds an experimental Big/Small research engine on top of the V7 number-first backend.

## V9.2 features
- Pattern transition matrix
- lag-1 / lag-2 / lag-3
- gap since Big / Small
- run length and momentum
- rolling entropy / volatility proxy
- order-aware sequence stability
- STABLE / CHOPPY regime-specific ensemble
- training-only ablation pruning
- walk-forward Isotonic calibration
- strict PREDICT/SKIP gating
- OOS accuracy + coverage reporting
- bounded calibration window to reduce cloud RAM/CPU usage

Endpoint: `/api/v9.2/pro`

Default gates:
- STABLE confidence: 0.72
- CHOPPY confidence: 0.80
- entropy stop: 0.75

These are experiment thresholds, not a promised accuracy. Use the OOS accuracy and coverage fields to judge whether the model has demonstrated an edge.

## Run
pip install -r backend/requirements.txt
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000

For Render, scikit-learn/numpy use materially more memory than the old statistical engine. A small-memory instance may still restart. If that happens, run V9.2 on a VM with more RAM or separate model computation from the API.
