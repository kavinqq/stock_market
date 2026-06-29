# Task 1 Report: Extract indicator primitives into marketlib.core

## What Was Implemented

Extracted the indicator primitive code from `daily_market_report.py` into a new `marketlib/core.py` module, following strict TDD (RED → GREEN) order.

### Files Changed

- **Created** `/Users/kuanlin/stock_market/marketlib/core.py`
  - Contains: `MarketBar` dataclass, `IndicatorSet` dataclass, `clean_num`, `pct_change`, `simple_moving_average`, `compute_indicators`
  - Imports only stdlib: `__future__`, `html`, `math`, `re`, `dataclasses`, `statistics`, `typing`
  - `compute_indicators` preserves the prior-20-excluding-today volume behavior (`prior_vols = [... for bar in bars[:-1]]`)

- **Created** `/Users/kuanlin/stock_market/tests/test_core.py`
  - 4 test cases as specified verbatim in the brief

- **Modified** `/Users/kuanlin/stock_market/daily_market_report.py`
  - Replaced `from dataclasses import asdict, dataclass` with `from dataclasses import asdict` + `from marketlib.core import (MarketBar, IndicatorSet, clean_num, pct_change, simple_moving_average, compute_indicators,)`
  - Deleted the `MarketBar` dataclass definition
  - Deleted the `IndicatorSet` dataclass definition
  - Deleted `clean_num` function
  - Deleted `simple_moving_average` function
  - Deleted `pct_change` function
  - Deleted `compute_indicators` function
  - Retained all other imports (`html`, `re`, `math`, `mean`, `pstdev`) that are still used elsewhere in the file

## TDD RED/GREEN Evidence

### RED (Step 2)
```
$ python3 -m unittest -v tests.test_core
test_core (unittest.loader._FailedTest.test_core) ... ERROR
ImportError: Failed to import test module: test_core
ModuleNotFoundError: No module named 'marketlib.core'
FAILED (errors=1)
```

### GREEN (Step 5)
```
$ python3 -m unittest -v tests.test_core
test_clean_num_strips_symbols (tests.test_core.TestCore.test_clean_num_strips_symbols) ... ok
test_pct_change (tests.test_core.TestCore.test_pct_change) ... ok
test_sma (tests.test_core.TestCore.test_sma) ... ok
test_volume_ratio_excludes_today (tests.test_core.TestCore.test_volume_ratio_excludes_today) ... ok

Ran 4 tests in 0.000s
OK
```

### Smoke Test
```
$ python3 -c "import daily_market_report"
(no output — success)
```

## Self-Review Findings

1. `compute_indicators` in core.py correctly uses `bars[:-1]` for prior volumes, matching the already-fixed behavior that excludes today from the 20-day average volume denominator. The `test_volume_ratio_excludes_today` test confirms this with ratio=2.0.

2. The `dataclass` decorator import was removed from `daily_market_report.py` since no dataclass definitions remain there. `asdict` is kept because it is used on lines referencing `MarketBar` and `IndicatorSet` instances (which still work via the imported names).

3. The `mean` import is preserved in `daily_market_report.py` because it is still used on line ~1251 (`mean(f1_scores)`). `pstdev` is retained too.

4. No stdlib-only constraint violated: `marketlib/core.py` uses only `html`, `math`, `re`, `dataclasses`, `statistics`, `typing`, `__future__`.

## Concerns

None. All acceptance criteria satisfied cleanly.
