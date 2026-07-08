# Task 1 Report — TA Indicator Expansion (stocks.py)

## Status
COMPLETE — all 43 tests pass (prior 37 + new 6 from TestIndicators).

## Files Modified
- `marketlib/stocks.py` — appended new functions + extended existing ones
- `tests/test_stocks.py` — appended TestIndicators class with 6 new tests

## New Functions Added

### `bollinger(closes, n=20, k=2.0) -> Optional[dict]`
Returns `{"mid", "upper", "lower", "pctb"}`. mid = SMA(n) using `statistics.mean`; std via `statistics.pstdev` (population stddev). pctb=None when band width is 0. Returns None when `len(closes) < n`.

### `vwap(bars, n) -> Optional[float]`
Σ(close·volume)/Σ(volume) over last n bars. Returns None when `len(bars) < n` or Σvolume == 0. Labeled as 大戶成本代理 (ESTIMATE, not real broker data).

### `resample_monthly(bars) -> list[MarketBar]`
Groups by `(year, month)` using `OrderedDict` (mirrors `resample_weekly`). close=last, high=max, low=min, volume=sum, date=last-in-month.

## Extended Functions

### `technical_metrics(daily_bars)` — new keys added
- `ma120`, `ma240` — SMA 120/240 day (None if insufficient)
- `boll` — bollinger dict or None
- `macd_monthly` — resample_monthly → macd (or None)
- `cost_vwap60`, `cost_vwap120` — 大戶成本代理
- `price_vs_cost60` — close/cost_vwap60 − 1 (or None)
- `kd` — full KD dict (was only kd_k/kd_d before)
- All existing keys preserved

### `tech_raw_score(daily_bars)` — expanded from 4 to 7 signals
Fraction-of-computable over:
1. close ≥ ma20
2. ma20 ≥ ma60
3. daily MACD bullish (golden cross or hist > 0)
4. weekly MACD hist > 0
5. boll pctb > 0.5  ← NEW
6. close ≥ cost_vwap60  ← NEW
7. KD k ≥ d  ← NEW

Still returns None if < 2 computable.

## Math Verification
- `vwap([10,20], [1,3], n=2)` → (10×1 + 20×3)/4 = 17.5 ✓
- `bollinger([10]*20)` → mid=10.0, std=0, pctb=None ✓
- `resample_monthly` groups Jan/Feb correctly, close=last-in-month ✓

## Test Results
```
Ran 43 tests in 0.003s — OK
```
All prior 37 tests remain intact. 6 new TestIndicators tests pass.

## Concerns / Notes
- `ma240` requires 240+ bars; typical 10-month lookback (~200 days) will return None. Expected behaviour per brief.
- `macd_monthly` also requires 26+ monthly bars (~2+ years of data) before producing a value.
- `price_vs_cost60` is None whenever cost_vwap60 is None (< 60 bars) — consistent with brief.
- No network calls; all new functions are pure/offline. stdlib only (statistics.pstdev used for Bollinger).
