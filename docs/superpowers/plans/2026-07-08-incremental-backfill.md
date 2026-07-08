# Incremental Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `backfill.py` with a per-run request budget, valuation-history backfill, `--incremental` priority-queue mode, and a `--coverage` summary flag — all rate-limit-friendly and resumable.

**Architecture:** Thread a `Budget` object (request counter + cap) through backfill functions that hit the network; raise `BudgetExhausted` when the cap is reached and catch it at the top of `main()` for a clean stop. Add `backfill_valuation()` that fetches BWIBBU_d ALL-market once per trading day and upserts into `stock_valuation`. Add `--incremental` mode that drives a priority queue (valuation → TW months → stock_daily) bounded by the budget. Add `--coverage` that queries row counts/date ranges without network.

**Tech Stack:** Python 3.10+, stdlib only (`argparse`, `unittest`, `sqlite3`, `datetime`), `marketlib.store.HistoryStore`, `marketlib.fetchers` (`twse_asof`, `fetch_valuation_all` via `ns` namespace object), `marketlib.universe.TW50_CONSTITUENTS`.

## Global Constraints

- stdlib + marketlib only — no new pip dependencies.
- Preserve all existing `backfill_tw` / `backfill_us` / `backfill_finra` behaviour and all existing tests must stay green.
- No network calls in tests — use fake fetcher objects / monkey-patching only.
- Branch is `feature/prediction-training` — do NOT init or re-branch.
- Default `--max-requests` = 500.
- `ingest_log` resume keys for valuation: `"valuation:<date>"` where date is ISO `"YYYY-MM-DD"`.
- `fetch_valuation_all(date_iso, codes)` already lives in `marketlib/stocks.py`; the `ns` namespace object in `main()` will be `marketlib.stocks` for real runs and a fake for tests.
- Report file: `/Users/kuanlin/stock_market/.superpowers/sdd/incremental-backfill-report.md`.

---

### Task 1: Budget object + `BudgetExhausted` exception

**Files:**
- Modify: `/Users/kuanlin/stock_market/backfill.py` (add class + exception near top, after imports)
- Test: `/Users/kuanlin/stock_market/tests/test_backfill.py`

**Interfaces:**
- Produces:
  - `class BudgetExhausted(Exception): pass`
  - `class Budget: __init__(self, max_requests: int) -> None; spend(self, n: int = 1) -> None  # raises BudgetExhausted when exhausted; remaining property -> int`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_backfill.py, add a new TestBudget class:
class TestBudget(unittest.TestCase):
    def test_spend_within_limit_does_not_raise(self):
        b = backfill.Budget(max_requests=3)
        b.spend()   # 1
        b.spend()   # 2
        b.spend()   # 3
        # Should not raise

    def test_spend_over_limit_raises(self):
        b = backfill.Budget(max_requests=2)
        b.spend()
        b.spend()
        with self.assertRaises(backfill.BudgetExhausted):
            b.spend()

    def test_remaining(self):
        b = backfill.Budget(max_requests=5)
        b.spend(2)
        self.assertEqual(b.remaining, 3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestBudget -v
```
Expected: `AttributeError: module 'backfill' has no attribute 'Budget'`

- [ ] **Step 3: Add `Budget` and `BudgetExhausted` to `backfill.py`**

Add immediately after the existing imports (after line 7, before `def _max_asof`):

```python
class BudgetExhausted(Exception):
    """Raised when the per-run request budget is consumed."""


class Budget:
    """Counts outgoing network requests and enforces a per-run cap.

    Call spend() before (or after) each fetch; it raises BudgetExhausted
    when the running total would exceed max_requests so the caller can stop
    cleanly.
    """

    def __init__(self, max_requests: int) -> None:
        self._max = max_requests
        self._count = 0

    @property
    def remaining(self) -> int:
        return max(0, self._max - self._count)

    def spend(self, n: int = 1) -> None:
        self._count += n
        if self._count > self._max:
            raise BudgetExhausted(
                f"request budget exhausted ({self._count}/{self._max})"
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestBudget -v
```
Expected: `3 tests … OK`

- [ ] **Step 5: Verify existing tests still green**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill -v 2>&1 | tail -5
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/kuanlin/stock_market && git add backfill.py tests/test_backfill.py
git commit -m "feat: add Budget/BudgetExhausted for per-run request cap"
```

---

### Task 2: `backfill_valuation()` — valuation history per trading day

**Files:**
- Modify: `/Users/kuanlin/stock_market/backfill.py` (add `backfill_valuation` function)
- Test: `/Users/kuanlin/stock_market/tests/test_backfill.py`

**Interfaces:**
- Consumes:
  - `Budget` from Task 1
  - `store.upsert_stock_valuation(row: dict)` — row keys: `stock_no, date, pe, pb, dividend_yield, asof_taipei`
  - `store.mark_ingested("valuation", date_iso: str)`
  - `store.is_ingested("valuation", date_iso: str) -> bool`
  - `ns.fetch_valuation_all(date_iso: str, codes) -> dict[str, dict]` — returns `{code: {pe, pb, dividend_yield, ...}}`; already in `marketlib/stocks.py`
  - `F.twse_asof(date_iso: str, "price") -> str` — from `marketlib.fetchers`
- Produces:
  - `def backfill_valuation(store, ns, codes, dates, budget: Budget, sleep: float = 1.5) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# In tests/test_backfill.py, add FakeValuationFetcher and TestBackfillValuation:

class FakeValuationFetcher:
    """Fake ns for backfill_valuation — no network."""
    def __init__(self, responses=None):
        # responses: {date_iso: {code: {pe, pb, dividend_yield}}}
        self.responses = responses or {}
        self.calls = []  # record (date_iso,) tuples for assertion

    def fetch_valuation_all(self, date_iso, codes):
        self.calls.append(date_iso)
        return self.responses.get(date_iso, {})


class TestBackfillValuation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def _make_ns(self, dates, codes):
        """Build a FakeValuationFetcher with canned responses for all dates × codes."""
        responses = {}
        for d in dates:
            responses[d] = {c: {"pe": 15.0, "pb": 2.0, "dividend_yield": 3.5} for c in codes}
        return FakeValuationFetcher(responses)

    def test_upserts_rows_and_marks_ingested(self):
        codes = ["2330", "2317"]
        dates = ["2026-06-20", "2026-06-21"]
        ns = self._make_ns(dates, codes)
        budget = backfill.Budget(max_requests=10)
        backfill.backfill_valuation(self.store, ns, codes, dates, budget, sleep=0)
        # Each date should be marked ingested
        for d in dates:
            self.assertTrue(self.store.is_ingested("valuation", d),
                            f"date {d} should be marked ingested")
        # Each code × date should appear in stock_valuation
        for code in codes:
            rows = self.store.get_stock_valuation(code)
            self.assertEqual(len(rows), 2, f"{code} should have 2 valuation rows")
            self.assertAlmostEqual(rows[0]["pe"], 15.0)
            self.assertAlmostEqual(rows[0]["pb"], 2.0)
            self.assertAlmostEqual(rows[0]["dividend_yield"], 3.5)

    def test_second_run_skips_ingested_dates(self):
        codes = ["2330"]
        dates = ["2026-06-20"]
        ns = self._make_ns(dates, codes)
        budget = backfill.Budget(max_requests=10)
        backfill.backfill_valuation(self.store, ns, codes, dates, budget, sleep=0)
        first_call_count = len(ns.calls)

        # Second run with same dates — should be skipped
        backfill.backfill_valuation(self.store, ns, codes, dates, budget, sleep=0)
        self.assertEqual(len(ns.calls), first_call_count,
                         "Second run must not call fetch_valuation_all for already-ingested dates")

    def test_budget_stops_early(self):
        codes = ["2330"]
        dates = ["2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21"]
        ns = self._make_ns(dates, codes)
        budget = backfill.Budget(max_requests=2)
        try:
            backfill.backfill_valuation(self.store, ns, codes, dates, budget, sleep=0)
        except backfill.BudgetExhausted:
            pass  # acceptable — budget exhausted mid-run is also correct
        # Only 2 dates should have been fetched
        ingested_count = sum(1 for d in dates if self.store.is_ingested("valuation", d))
        self.assertLessEqual(ingested_count, 2,
                             "Budget=2 must stop after at most 2 fetches; remaining dates uningested")
        not_ingested = [d for d in dates if not self.store.is_ingested("valuation", d)]
        self.assertGreater(len(not_ingested), 0,
                           "Some dates must remain un-ingested so next run can resume")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestBackfillValuation -v
```
Expected: `AttributeError: module 'backfill' has no attribute 'backfill_valuation'`

- [ ] **Step 3: Implement `backfill_valuation` in `backfill.py`**

Add after `backfill_tw` (before `backfill_us`) in `/Users/kuanlin/stock_market/backfill.py`:

```python
def backfill_valuation(store, ns, codes, dates, budget: Budget,
                       sleep: float = 1.5) -> None:
    """Backfill stock_valuation for all codes on each date in dates.

    One BWIBBU_d ALL-market request per trading date gives PE/PB/yield for all
    codes simultaneously. Dates already marked ingested("valuation", date) are
    skipped. Budget.spend() is called once per fetch; BudgetExhausted propagates
    to the caller for a clean stop (resume on next run via ingest_log).
    """
    import random
    for date_iso in dates:
        if store.is_ingested("valuation", date_iso):
            continue
        budget.spend()  # raises BudgetExhausted if cap exceeded
        result = ns.fetch_valuation_all(date_iso, codes)
        asof = F.twse_asof(date_iso, "price")
        for code in codes:
            vals = result.get(code)
            if vals is None:
                continue
            store.upsert_stock_valuation({
                "stock_no": code,
                "date": date_iso,
                "pe": vals.get("pe"),
                "pb": vals.get("pb"),
                "dividend_yield": vals.get("dividend_yield"),
                "asof_taipei": asof,
            })
        store.mark_ingested("valuation", date_iso)
        if sleep:
            time.sleep(sleep + random.uniform(0, 0.5))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestBackfillValuation -v
```
Expected: `3 tests … OK`

- [ ] **Step 5: Verify all existing tests still green**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill -v 2>&1 | tail -5
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/kuanlin/stock_market && git add backfill.py tests/test_backfill.py
git commit -m "feat: add backfill_valuation() with budget and resume via ingest_log"
```

---

### Task 3: `coverage()` function + `--coverage` CLI flag

**Files:**
- Modify: `/Users/kuanlin/stock_market/backfill.py` (add `coverage()` function; add `--coverage` arg in `main()`)
- Test: `/Users/kuanlin/stock_market/tests/test_backfill.py`

**Interfaces:**
- Consumes:
  - `store.conn.execute(sql)` — direct SQLite queries for row counts and min/max dates
- Produces:
  - `def coverage(store, start: str) -> dict` — returns a dict with keys:
    - `"tw_daily": {"rows": int, "min_date": str|None, "max_date": str|None, "pct_filled": float|None}`
    - `"us_daily": {"rows": int, "min_date": str|None, "max_date": str|None}`
    - `"finra_monthly": {"rows": int, "min_month": str|None, "max_month": str|None}`
    - `"stock_valuation": {"rows": int, "min_date": str|None, "max_date": str|None}`
    - `"stock_daily": {"rows": int, "min_date": str|None, "max_date": str|None}`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_backfill.py, add TestCoverage:

class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_coverage_empty_db(self):
        result = backfill.coverage(self.store, start="2020-01-01")
        self.assertIn("tw_daily", result)
        self.assertIn("us_daily", result)
        self.assertIn("finra_monthly", result)
        self.assertIn("stock_valuation", result)
        self.assertIn("stock_daily", result)
        # Empty db should have 0 rows
        self.assertEqual(result["tw_daily"]["rows"], 0)
        self.assertEqual(result["stock_valuation"]["rows"], 0)

    def test_coverage_with_data(self):
        # Insert one tw_daily row and one stock_valuation row
        self.store.upsert_tw_daily({
            "date": "2026-01-02",
            "close": 20000.0,
            "price_asof_taipei": "2026-01-02T14:30:00+08:00",
            "available_at_taipei": "2026-01-02T16:30:00+08:00",
        })
        self.store.upsert_stock_valuation({
            "stock_no": "2330",
            "date": "2026-01-02",
            "pe": 18.0,
            "pb": 3.0,
            "dividend_yield": 2.5,
            "asof_taipei": "2026-01-02T14:30:00+08:00",
        })
        result = backfill.coverage(self.store, start="2020-01-01")
        self.assertEqual(result["tw_daily"]["rows"], 1)
        self.assertEqual(result["tw_daily"]["min_date"], "2026-01-02")
        self.assertEqual(result["stock_valuation"]["rows"], 1)
        self.assertEqual(result["stock_valuation"]["min_date"], "2026-01-02")
        # pct_filled should be a number or None — not raise
        pct = result["tw_daily"].get("pct_filled")
        if pct is not None:
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 1.0)

    def test_coverage_no_network(self):
        # Calling coverage() must not raise any exception even on empty db
        try:
            backfill.coverage(self.store, start="2014-01-01")
        except Exception as exc:
            self.fail(f"coverage() raised unexpectedly: {exc}")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestCoverage -v
```
Expected: `AttributeError: module 'backfill' has no attribute 'coverage'`

- [ ] **Step 3: Implement `coverage()` and wire `--coverage` in `main()`**

Add `coverage()` after `backfill_finra` in `/Users/kuanlin/stock_market/backfill.py`:

```python
def coverage(store, start: str) -> dict:
    """Return row counts and date ranges for all history tables — no network.

    start: ISO date string (e.g. "2014-01-01"); used to estimate % filled for
    tw_daily (trading days in range are approximated as weekdays × 5/7).
    """
    from datetime import date as date_cls, timedelta

    def _query_range(table: str, col: str):
        row = store.conn.execute(
            f"SELECT COUNT(*), MIN({col}), MAX({col}) FROM {table}"
        ).fetchone()
        return {"rows": row[0] or 0, f"min_{col}": row[1], f"max_{col}": row[2]}

    tw = _query_range("tw_daily", "date")
    # Estimate pct_filled: approximate trading days from start to today
    try:
        start_d = date_cls.fromisoformat(start)
        today = date_cls.today()
        total_calendar = max((today - start_d).days, 1)
        # Rough estimate: ~252 trading days per year; 252/365 ≈ 0.69
        approx_trading_days = int(total_calendar * 252 / 365)
        tw["pct_filled"] = (
            round(tw["rows"] / approx_trading_days, 3)
            if approx_trading_days > 0 else None
        )
    except Exception:
        tw["pct_filled"] = None

    us = _query_range("us_daily", "date")
    finra = store.conn.execute(
        "SELECT COUNT(*), MIN(month), MAX(month) FROM finra_monthly"
    ).fetchone()
    sv = store.conn.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM stock_valuation"
    ).fetchone()
    sd = store.conn.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM stock_daily"
    ).fetchone()

    return {
        "tw_daily": tw,
        "us_daily": us,
        "finra_monthly": {
            "rows": finra[0] or 0,
            "min_month": finra[1],
            "max_month": finra[2],
        },
        "stock_valuation": {
            "rows": sv[0] or 0,
            "min_date": sv[1],
            "max_date": sv[2],
        },
        "stock_daily": {
            "rows": sd[0] or 0,
            "min_date": sd[1],
            "max_date": sd[2],
        },
    }
```

In `main()`, add argument and handler. After `p.add_argument("--backoff", ...)` add:

```python
    p.add_argument("--coverage", action="store_true",
                   help="Print coverage summary without network access, then exit.")
    p.add_argument("--max-requests", type=int, default=500, dest="max_requests",
                   help="Per-run request budget (default 500). Stop cleanly when exhausted.")
    p.add_argument("--incremental", action="store_true",
                   help="Priority-queue incremental mode: valuation → TW months → stock_daily.")
```

And after `store = HistoryStore(a.db); store.init_schema()` add:

```python
    if a.coverage:
        info = coverage(store, a.start)
        _print_coverage(info)
        store.close()
        return 0
```

Add `_print_coverage` helper before `main()`:

```python
def _print_coverage(info: dict) -> None:
    """Print a human-readable coverage summary to stdout."""
    print("=== History Coverage Summary ===")
    for name, d in info.items():
        rows = d.get("rows", 0)
        min_k = [k for k in d if k.startswith("min_")]
        max_k = [k for k in d if k.startswith("max_")]
        min_v = d.get(min_k[0]) if min_k else None
        max_v = d.get(max_k[0]) if max_k else None
        pct = d.get("pct_filled")
        pct_str = f"  {pct:.1%} filled" if pct is not None else ""
        print(f"  {name:25s}: {rows:6d} rows  [{min_v or '—'} … {max_v or '—'}]{pct_str}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestCoverage -v
```
Expected: `3 tests … OK`

- [ ] **Step 5: Smoke-test the import**

```bash
cd /Users/kuanlin/stock_market && python3 -c "import backfill; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Verify full test suite**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill -v 2>&1 | tail -5
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
cd /Users/kuanlin/stock_market && git add backfill.py tests/test_backfill.py
git commit -m "feat: add coverage() summary function and --coverage CLI flag"
```

---

### Task 4: `--incremental` priority-queue mode in `main()`

**Files:**
- Modify: `/Users/kuanlin/stock_market/backfill.py` (add `backfill_stock_day_month()` helper + `incremental_run()` function; wire into `main()`)
- Test: `/Users/kuanlin/stock_market/tests/test_backfill.py`

**Interfaces:**
- Consumes:
  - `Budget`, `BudgetExhausted` from Task 1
  - `backfill_valuation(store, ns, codes, dates, budget, sleep)` from Task 2
  - `backfill_tw(store, ns, months, sleep, max_retries, backoff)` — existing; already resume-aware via `ingest_log("tw_month", key)`
  - `store.conn.execute(...)` — to discover earliest ingested tw_month and known trading dates
  - `marketlib.universe.TW50_CONSTITUENTS`
- Produces:
  - `def backfill_stock_day_month(store, ns, stock_no: str, month_start: date, budget: Budget, sleep: float) -> None` — fetches one month of STOCK_DAY for stock_no, upserts into `stock_daily`, marks `ingest_log("stock_day:<stock_no>:<month_key>", "ok")`.
  - `def incremental_run(store, ns, start: date, budget: Budget, sleep: float = 1.5, max_retries: int = 3, backoff: float = 30.0) -> None`

**Note:** `ns` for real runs will be `marketlib.stocks` (which has `fetch_valuation_all`, `fetch_stock_day`). For tests, use a fake with both methods.

- [ ] **Step 1: Write failing tests**

```python
# In tests/test_backfill.py, add FakeIncrementalNS and TestIncrementalRun:

class FakeIncrementalNS:
    """Fake namespace supporting fetch_valuation_all, fetch_stock_day,
    fetch_twse_market_month, fetch_twse_institutional, fetch_twse_margin,
    fetch_twse_summary for incremental_run tests."""

    def __init__(self, trading_dates=None, codes=None):
        self.trading_dates = trading_dates or []
        self.codes = codes or []
        self.val_calls = []
        self.stock_day_calls = []
        self.market_month_calls = []

    def fetch_valuation_all(self, date_iso, codes):
        self.val_calls.append(date_iso)
        return {c: {"pe": 18.0, "pb": 3.0, "dividend_yield": 2.5} for c in codes}

    def fetch_stock_day(self, stock_no, month_start):
        self.stock_day_calls.append((stock_no, month_start))
        return []  # empty is fine — no stock_daily rows inserted but no crash

    def fetch_twse_market_month(self, month_start):
        self.market_month_calls.append(month_start)
        return []  # no bars — month will not be marked ingested (no block either → mark ok)

    def fetch_twse_institutional(self, date_iso):
        return {}

    def fetch_twse_margin(self, date_iso):
        return {}

    def fetch_twse_summary(self, date_iso):
        return {}


class TestIncrementalRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()
        # Pre-populate tw_daily with known trading dates so incremental_run
        # uses those dates for valuation, not random weekdays.
        for d in ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]:
            self.store.upsert_tw_daily({
                "date": d,
                "close": 20000.0,
                "price_asof_taipei": f"{d}T14:30:00+08:00",
                "available_at_taipei": f"{d}T16:30:00+08:00",
            })

    def _make_ns(self):
        return FakeIncrementalNS(
            trading_dates=["2026-01-02", "2026-01-05", "2026-01-06"],
            codes=["2330"],
        )

    def test_valuation_dates_ingested_after_run(self):
        ns = self._make_ns()
        budget = backfill.Budget(max_requests=50)
        from datetime import date
        backfill.incremental_run(
            self.store, ns,
            start=date(2026, 1, 1),
            budget=budget, sleep=0,
        )
        # At least some valuation dates should be ingested
        ingested = [d for d in ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
                    if self.store.is_ingested("valuation", d)]
        self.assertGreater(len(ingested), 0,
                           "incremental_run must ingest at least one valuation date")

    def test_budget_stops_run_cleanly(self):
        ns = self._make_ns()
        budget = backfill.Budget(max_requests=2)
        from datetime import date
        # Must not raise — budget exhaustion must be caught inside incremental_run
        try:
            backfill.incremental_run(
                self.store, ns,
                start=date(2026, 1, 1),
                budget=budget, sleep=0,
            )
        except Exception as exc:
            self.fail(f"incremental_run raised unexpectedly with small budget: {exc}")
        # Budget is used up — remaining ≤ 0
        self.assertLessEqual(budget.remaining, 0)

    def test_already_ingested_valuation_dates_skipped(self):
        self.store.mark_ingested("valuation", "2026-01-02")
        self.store.mark_ingested("valuation", "2026-01-05")
        ns = self._make_ns()
        budget = backfill.Budget(max_requests=10)
        from datetime import date
        backfill.incremental_run(
            self.store, ns,
            start=date(2026, 1, 1),
            budget=budget, sleep=0,
        )
        # Pre-ingested dates should not appear in fetch calls
        self.assertNotIn("2026-01-02", ns.val_calls)
        self.assertNotIn("2026-01-05", ns.val_calls)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestIncrementalRun -v
```
Expected: `AttributeError: module 'backfill' has no attribute 'incremental_run'`

- [ ] **Step 3: Implement `backfill_stock_day_month()` and `incremental_run()` in `backfill.py`**

Add `backfill_stock_day_month` after `backfill_valuation`:

```python
def backfill_stock_day_month(store, ns, stock_no: str, month_start: date,
                              budget: Budget, sleep: float = 1.5) -> None:
    """Fetch one month of STOCK_DAY for stock_no and upsert into stock_daily.

    Uses ingest_log key "stock_day:<stock_no>:<YYYY-MM>" for resume.
    Calls budget.spend() once per fetch; BudgetExhausted propagates to caller.
    """
    import random
    from marketlib.fetchers import twse_asof
    key = f"stock_day:{stock_no}:{month_start.strftime('%Y-%m')}"
    if store.is_ingested("stock_day", key):
        return
    budget.spend()
    bars = ns.fetch_stock_day(stock_no, month_start)
    price_asof = twse_asof(month_start.strftime("%Y-%m-%d"), "price")
    for bar in bars:
        store.upsert_stock_daily({
            "stock_no": stock_no,
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "price_asof_taipei": price_asof,
            "available_at_taipei": price_asof,
        })
    store.mark_ingested("stock_day", key)
    if sleep:
        time.sleep(sleep + random.uniform(0, 0.5))
```

Add `incremental_run` after `backfill_stock_day_month`:

```python
def incremental_run(store, ns, start: date, budget: Budget,
                    sleep: float = 1.5, max_retries: int = 3,
                    backoff: float = 30.0) -> None:
    """Bounded incremental backfill using a priority queue.

    Priority 1: Valuation history for TW50_CONSTITUENTS (most-recent
                 missing trading day backward to start).
    Priority 2: Extend INDEX chip history backward (missing tw_months
                 between start and earliest already-ingested month).
    Priority 3: Pre-warm per-stock STOCK_DAY for constituents (missing
                 recent months into stock_daily).

    Stops cleanly when BudgetExhausted; remaining work resumes next run
    via ingest_log.
    """
    from marketlib.universe import TW50_CONSTITUENTS

    today_iso = datetime.now(F.TAIPEI_TZ).date().isoformat()

    # --- Derive known trading dates from tw_daily (most reliable source) ---
    rows = store.conn.execute(
        "SELECT date FROM tw_daily WHERE date >= ? AND date <= ? ORDER BY date DESC",
        (start.isoformat(), today_iso),
    ).fetchall()
    trading_dates = [r[0] for r in rows]

    # --- Priority 1: Valuation history ---
    try:
        backfill_valuation(store, ns, TW50_CONSTITUENTS, trading_dates, budget, sleep=sleep)
    except BudgetExhausted:
        return

    # --- Priority 2: Extend TW index chip history backward ---
    # Find earliest already-ingested tw_month key
    ingested_months = set(
        row[0] for row in store.conn.execute(
            "SELECT key FROM ingest_log WHERE source='tw_month' AND status='ok'"
        ).fetchall()
    )
    cur = date(start.year, start.month, 1)
    today_d = datetime.now(F.TAIPEI_TZ).date()
    end_month = date(today_d.year, today_d.month, 1)
    missing_months = []
    while cur <= end_month:
        key = cur.strftime("%Y-%m")
        if key not in ingested_months:
            missing_months.append(cur)
        cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
    try:
        backfill_tw(store, ns, missing_months, sleep=sleep,
                    max_retries=max_retries, backoff=backoff)
    except BudgetExhausted:
        return

    # --- Priority 3: Pre-warm per-stock STOCK_DAY (recent 6 months) ---
    from marketlib.fetchers import month_starts as _month_starts
    today_d = datetime.now(F.TAIPEI_TZ).date()
    recent_months = _month_starts(today_d, 6)
    try:
        for code in TW50_CONSTITUENTS:
            for month_start in recent_months:
                backfill_stock_day_month(store, ns, code, month_start, budget, sleep=sleep)
    except BudgetExhausted:
        return
```

**Important:** `backfill_tw` does not accept a `budget` parameter currently — it is resume-aware via `ingest_log` but has no budget hook. For Priority 2, wrap its call so that if `BudgetExhausted` leaks out (it won't from `backfill_tw` since `backfill_tw` doesn't call `budget.spend()`) we catch it defensively. The budget is consumed by `backfill_valuation` and `backfill_stock_day_month`; `backfill_tw` still uses its own internal loop without explicit budget accounting (it uses gentle sleep instead). This is an acceptable trade-off for the first version — the budget primarily governs valuation and stock_day fetches. Document this in the report.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill.TestIncrementalRun -v
```
Expected: `3 tests … OK`

- [ ] **Step 5: Wire `--incremental` into `main()`**

In `main()`, after the `if a.coverage:` block (or after `store.init_schema()`), add:

```python
    import marketlib.stocks as _stocks_ns
    budget = Budget(max_requests=a.max_requests)
    if a.incremental:
        try:
            incremental_run(store, _stocks_ns, start, budget,
                            sleep=a.sleep, max_retries=a.max_retries,
                            backoff=a.backoff)
        except BudgetExhausted:
            pass
        store.close()
        return 0
```

(Keep existing `if "tw" in sources:` / `if "us" in sources:` / `if "finra" in sources:` blocks intact below for non-incremental runs.)

- [ ] **Step 6: Smoke-test import and help**

```bash
cd /Users/kuanlin/stock_market && python3 -c "import backfill; print('ok')"
cd /Users/kuanlin/stock_market && python3 backfill.py --help
```
Expected: `ok` then help text showing `--incremental`, `--coverage`, `--max-requests`.

- [ ] **Step 7: Run full test suite**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest tests.test_backfill -v 2>&1 | tail -10
```
Expected: all tests pass, `OK`.

- [ ] **Step 8: Commit**

```bash
cd /Users/kuanlin/stock_market && git add backfill.py tests/test_backfill.py
git commit -m "feat: add incremental_run() priority queue and --incremental CLI flag"
```

---

### Task 5: Write report and final verification

**Files:**
- Create: `/Users/kuanlin/stock_market/.superpowers/sdd/incremental-backfill-report.md`

**Interfaces:**
- Consumes: completed Tasks 1–4

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest -v tests.test_backfill 2>&1
```
Expected: all tests pass, `OK`. Record exact test count and names.

- [ ] **Step 2: Run broader test suite to check no regressions**

```bash
cd /Users/kuanlin/stock_market && python3 -m unittest discover -s tests -v 2>&1 | tail -20
```
Expected: `OK` with 0 errors/failures.

- [ ] **Step 3: Write the report**

Create `/Users/kuanlin/stock_market/.superpowers/sdd/incremental-backfill-report.md` with:
- Budget mechanism description (BudgetExhausted, Budget.spend, where threaded)
- Valuation backfill description (1 request/day, BWIBBU_d ALL-market, ingest_log key format)
- Priority queue description (P1 valuation, P2 tw_months, P3 stock_day; budget accounting note: tw_month does not decrement budget)
- Coverage description (no-network, pct_filled approximation method)
- Tests added (list all new test class names and what they verify)
- Files changed (backfill.py lines added, tests/test_backfill.py additions)
- Known concerns / limitations

- [ ] **Step 4: Commit**

```bash
cd /Users/kuanlin/stock_market && git add .superpowers/sdd/incremental-backfill-report.md
git commit -m "docs: add incremental-backfill-report.md"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| Per-run request budget (`Budget`, `BudgetExhausted`, `--max-requests`) | Task 1, Task 3 |
| `backfill_valuation()` — BWIBBU_d ALL-market once per date, upsert stock_valuation, mark ingest_log | Task 2 |
| `--incremental` mode with priority queue (P1 valuation, P2 tw_month, P3 stock_day) | Task 4 |
| `--coverage` flag — row counts + date ranges, no network | Task 3 |
| Tests: valuation upserts + marks ingested; second run skips; budget=2 stops early + resumable | Task 2 |
| Tests: coverage returns counts without error | Task 3 |
| Existing backfill_tw/us/finra + tests preserved | All tasks (no modification to existing functions) |

**Placeholder scan:** No TBD/TODO in any step. All test code and implementation code is fully written.

**Type consistency check:**
- `Budget` in Task 1 → `Budget` used in Tasks 2, 3, 4 (consistent)
- `BudgetExhausted` in Task 1 → caught in Tasks 2 `backfill_valuation`, 4 `incremental_run` (consistent)
- `backfill_valuation(store, ns, codes, dates, budget, sleep)` defined in Task 2 → called in Task 4 with same signature (consistent)
- `ns.fetch_valuation_all(date_iso, codes)` — real: `marketlib.stocks.fetch_valuation_all`; fake: `FakeValuationFetcher.fetch_valuation_all` — both return `dict[str, dict]` (consistent)
- `store.upsert_stock_valuation(row)` — `row` keys match `STOCK_VALUATION_COLUMNS` in `marketlib/store.py` (`stock_no, date, pe, pb, dividend_yield, asof_taipei`) (consistent)
- `coverage(store, start: str)` defined Task 3 → called in `main()` with `a.start` which is already a string (consistent)

**No issues found.**
