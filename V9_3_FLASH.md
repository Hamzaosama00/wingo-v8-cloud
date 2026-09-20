# V9.3 Flash Persistent

## Main changes
- Strict signal gating
  - Stable threshold: 0.80
  - Choppy threshold: 0.85
  - 3/3 model agreement by default
  - entropy + switch-rate volatility gates
- Recency / pattern decay
  - EMA 3, 8, 20
- Gap analysis
  - gap since Big / Small
  - gap-to-mean feature
- Sequence entropy
  - conditional transition entropy
- Diversified ensemble
  1. Trend Follower: EMA + momentum
  2. Reversionist: recent imbalance + gap pressure
  3. Pattern Matcher: Markov + N-gram
- Isotonic walk-forward calibration
- OOS accuracy and prediction coverage
- Firestore persistence retained, so new Render deploys restore historical rounds.

## Endpoint
GET /api/v9.3/flash

## Important
Higher confidence thresholds reduce coverage; they do not automatically guarantee higher future accuracy.
The reversion feature is treated as a hypothesis and is validated through walk-forward results.

## No money manager
This build intentionally does not include real-money bet sizing, Kelly staking,
Martingale, target-profit automation, or bankroll controls. It reports model
quality using prediction accuracy, support, agreement and coverage only.


## Virtual PKR simulator
Endpoint:
`GET /api/v9.3/simulate`

Parameters:
- `starting_balance` — virtual PKR starting balance
- `target_balance` — stop once the historical simulation reaches this amount
- `stake_percent` — virtual fixed-fraction stake, default 2%, capped at 5%
- `max_rounds` — maximum historical rounds to process

The response includes ending balance, maximum balance reached, minimum balance,
P/L, hit rate, prediction/skip counts and historical simulation rows.

A simple UI is included at `frontend/simulator.html`.

This feature is historical/virtual simulation only. It does not connect to a real
wallet, place wagers, or automate live betting.
