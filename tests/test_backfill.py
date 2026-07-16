import os, tempfile, unittest
from datetime import date
from unittest.mock import patch
from marketlib.store import HistoryStore
from marketlib.core import MarketBar
import backfill

class FakeFetchers:
    def fetch_twse_market_month(self, month_start):
        from marketlib.core import MarketBar
        return [MarketBar(date="2026-06-26", close=100.0, turnover=5e11, volume=1e9)]
    def fetch_twse_institutional(self, date_iso):
        return {"rows": {"合計": {"net_ntd": 1.2e9}}}
    def fetch_twse_margin(self, date_iso):
        return {"margin_financing": {"today_balance_ntd": 2e11, "balance_change_ntd": 3e9}}
    def fetch_twse_summary(self, date_iso):
        return {"breadth": {"上漲": {"stocks": 600}, "下跌": {"stocks": 300}}}

class TestBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_tw_backfill_populates_asof(self):
        from datetime import date
        backfill.backfill_tw(self.store, FakeFetchers(), [date(2026, 6, 1)], sleep=0)
        rows = self.store.get_tw_rows()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertIsNotNone(r["price_asof_taipei"])
        self.assertIsNotNone(r["inst_asof_taipei"])
        self.assertIsNotNone(r["available_at_taipei"])
        self.assertAlmostEqual(r["inst_total_net_ntd"], 1.2e9)

    def test_resume_skips_done(self):
        from datetime import date
        backfill.backfill_tw(self.store, FakeFetchers(), [date(2026, 6, 1)], sleep=0)
        # 第二次:已 ingested 的月份不再覆蓋(用 is_ingested 判斷)
        self.assertTrue(self.store.is_ingested("tw_month", "2026-06"))

class TestBackfillUsVixOnly(unittest.TestCase):
    """Regression: VIX-only day must have price_asof_taipei=None but available_at_taipei set."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_vix_only_day_available_at_not_from_price(self):
        # 2026-01-02: VIX data exists but NO SPY/QQQ/IWM/COMP data (e.g. holiday or data gap)
        # 2026-01-05: both SPY and VIX data exist (control day)
        vix_only_date = date(2026, 1, 2)
        both_date = date(2026, 1, 5)

        vix_bar_only = MarketBar(date="2026-01-02", close=18.5)
        vix_bar_both = MarketBar(date="2026-01-05", close=17.0)
        spy_bar = MarketBar(date="2026-01-05", close=590.0, volume=1e8)

        def fake_fetch_nasdaq(symbol, asset_type, start, end):
            if symbol == "SPY":
                return [spy_bar]
            return []

        def fake_fetch_vix(start, end):
            return [vix_bar_only, vix_bar_both]

        import marketlib.fetchers as F
        with patch.object(F, "fetch_nasdaq_history", side_effect=fake_fetch_nasdaq), \
             patch.object(F, "fetch_vix_history", side_effect=fake_fetch_vix):
            backfill.backfill_us(self.store, date(2026, 1, 1), date(2026, 1, 31), sleep=0)

        rows = {r["date"]: r for r in self.store.get_us_rows()}

        # VIX-only day: price_asof must be None, but available_at must be set (from vix_asof)
        r_vix = rows[str(vix_only_date)]
        self.assertIsNone(r_vix["price_asof_taipei"],
                          "price_asof_taipei should be None on VIX-only day")
        self.assertIsNotNone(r_vix["available_at_taipei"],
                             "available_at_taipei must be set on VIX-only day (from vix_asof)")
        expected_vix_asof = F.vix_asof(str(vix_only_date))
        self.assertEqual(r_vix["available_at_taipei"], expected_vix_asof,
                         "available_at_taipei should equal vix_asof, NOT a price-derived timestamp")

        # Control day (both price + VIX): price_asof and available_at both set
        r_both = rows[str(both_date)]
        self.assertIsNotNone(r_both["price_asof_taipei"])
        self.assertIsNotNone(r_both["available_at_taipei"])


class FakeBlockingFetchers:
    """Fake fetcher where fetch_twse_margin always raises a TWSE block error."""
    def fetch_twse_market_month(self, month_start):
        return [MarketBar(date="2020-02-13", close=9500.0, turnover=1e11, volume=5e8)]

    def fetch_twse_institutional(self, date_iso):
        return {"rows": {"合計": {"net_ntd": 3.5e9}}}

    def fetch_twse_margin(self, date_iso):
        raise RuntimeError("... FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED ...")

    def fetch_twse_summary(self, date_iso):
        return {"breadth": {"上漲": {"stocks": 500}, "下跌": {"stocks": 250}}}


class TestBackfillTwBlockResilient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()
        self.fetcher = FakeBlockingFetchers()

    def _run_backfill(self):
        backfill.backfill_tw(
            self.store,
            self.fetcher,
            [date(2020, 2, 1)],
            sleep=0,
            backoff=0.0,
            max_retries=3,
        )

    def test_block_does_not_propagate(self):
        """backfill_tw must not raise even when margin fetch is always blocked."""
        try:
            self._run_backfill()
        except Exception as exc:
            self.fail(f"backfill_tw raised an exception: {exc}")

    def test_block_month_not_marked_ingested(self):
        """Month with an unrecovered block must NOT be marked ingested (resume integrity)."""
        self._run_backfill()
        self.assertFalse(
            self.store.is_ingested("tw_month", "2020-02"),
            "Month should NOT be marked ingested when a block exhausted all retries",
        )

    def test_block_price_data_still_stored(self):
        """Price data from fetch_twse_market_month must be stored despite margin block."""
        self._run_backfill()
        rows = self.store.get_tw_rows()
        self.assertEqual(len(rows), 1, "Expected exactly 1 row from the one MarketBar")
        self.assertAlmostEqual(rows[0]["close"], 9500.0)

    def test_is_block_error_detection(self):
        """Unit test the _is_block_error helper function."""
        self.assertTrue(
            backfill._is_block_error(
                RuntimeError("FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED")
            )
        )
        self.assertTrue(backfill._is_block_error(RuntimeError("無法呈現")))
        self.assertTrue(
            backfill._is_block_error(RuntimeError("SECURITY REASONS something else"))
        )
        self.assertFalse(backfill._is_block_error(RuntimeError("connection timeout")))
        self.assertFalse(backfill._is_block_error(ValueError("some other error")))


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


class FakeValuationFetcher:
    """Fake ns for backfill_valuation — no network."""
    def __init__(self, responses=None):
        # responses: {date_iso: {code: {pe, pb, dividend_yield}}}
        self.responses = responses or {}
        self.calls = []  # record date_iso for assertion

    def fetch_valuation_all(self, date_iso, codes):
        self.calls.append(date_iso)
        return self.responses.get(date_iso, {})


class TestBackfillValuation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def _make_ns(self, dates, codes):
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
        for d in dates:
            self.assertTrue(self.store.is_ingested("valuation", d),
                            f"date {d} should be marked ingested")
        for code in codes:
            rows = self.store.get_stock_valuation(code)
            self.assertEqual(len(rows), 2, f"{code} should have 2 valuation rows")
            self.assertAlmostEqual(rows[0]["pe"], 15.0)
            self.assertAlmostEqual(rows[0]["pb"], 2.0)
            self.assertAlmostEqual(rows[0]["dividend_yield"], 3.5)
            self.assertIsNotNone(rows[0]["asof_taipei"])

    def test_second_run_skips_ingested_dates(self):
        codes = ["2330"]
        dates = ["2026-06-20"]
        ns = self._make_ns(dates, codes)
        budget = backfill.Budget(max_requests=10)
        backfill.backfill_valuation(self.store, ns, codes, dates, budget, sleep=0)
        first_call_count = len(ns.calls)
        backfill.backfill_valuation(self.store, ns, codes, dates, budget, sleep=0)
        self.assertEqual(len(ns.calls), first_call_count,
                         "Second run must not call fetch_valuation_all for ingested dates")

    def test_budget_stops_early(self):
        codes = ["2330"]
        dates = ["2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21"]
        ns = self._make_ns(dates, codes)
        budget = backfill.Budget(max_requests=2)
        try:
            backfill.backfill_valuation(self.store, ns, codes, dates, budget, sleep=0)
        except backfill.BudgetExhausted:
            pass  # acceptable — mid-run exhaustion is also correct
        ingested_count = sum(1 for d in dates if self.store.is_ingested("valuation", d))
        self.assertLessEqual(ingested_count, 2,
                             "Budget=2 must stop after at most 2 fetches")
        not_ingested = [d for d in dates if not self.store.is_ingested("valuation", d)]
        self.assertGreater(len(not_ingested), 0,
                           "Some dates must remain un-ingested so next run can resume")


class FakeTaifexNS:
    """Fake ns for backfill_taifex — returns a fixed per-date map, records calls."""
    def __init__(self, by_date):
        self.by_date = by_date
        self.calls = []

    def fetch_taifex_inst_futures(self, start_iso, end_iso=None):
        self.calls.append((start_iso, end_iso))
        return self.by_date


class TestBackfillTaifex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_partial_upsert_preserves_price_and_adds_oi(self):
        # 既有 tw_daily 列(已回填價量),TAIFEX 只補期貨欄位,不得覆蓋 close。
        self.store.upsert_tw_daily({
            "date": "2026-07-14", "close": 45631.59,
            "price_asof_taipei": "2026-07-14T14:30:00+08:00"})
        ns = FakeTaifexNS({"2026-07-14": {
            "tx_foreign_oi_net_lots": -83390.0, "tx_inst_oi_net_lots": -7165.0}})
        backfill.backfill_taifex(self.store, ns, [date(2026, 7, 1)], sleep=0)
        r = self.store.get_tw_rows()[0]
        self.assertEqual(r["close"], 45631.59)                # preserved
        self.assertEqual(r["tx_foreign_oi_net_lots"], -83390.0)
        self.assertEqual(r["tx_inst_oi_net_lots"], -7165.0)
        self.assertIsNotNone(r["taifex_asof_taipei"])
        self.assertTrue(self.store.is_ingested("taifex_month", "2026-07"))

    def test_second_run_skips_ingested_month(self):
        ns = FakeTaifexNS({})
        backfill.backfill_taifex(self.store, ns, [date(2026, 7, 1)], sleep=0)
        n_first = len(ns.calls)
        backfill.backfill_taifex(self.store, ns, [date(2026, 7, 1)], sleep=0)
        self.assertEqual(len(ns.calls), n_first,
                         "Ingested TAIFEX month must not be re-fetched")


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_coverage_empty_db(self):
        result = backfill.coverage(self.store, start="2020-01-01")
        for key in ("tw_daily", "us_daily", "finra_monthly",
                    "stock_valuation", "stock_daily"):
            self.assertIn(key, result)
        self.assertEqual(result["tw_daily"]["rows"], 0)
        self.assertEqual(result["stock_valuation"]["rows"], 0)

    def test_coverage_with_data(self):
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
        pct = result["tw_daily"].get("pct_filled")
        if pct is not None:
            self.assertGreaterEqual(pct, 0.0)

    def test_coverage_no_network(self):
        try:
            backfill.coverage(self.store, start="2014-01-01")
        except Exception as exc:
            self.fail(f"coverage() raised unexpectedly: {exc}")

    def test_print_coverage_no_error(self):
        info = backfill.coverage(self.store, start="2020-01-01")
        try:
            backfill._print_coverage(info)
        except Exception as exc:
            self.fail(f"_print_coverage raised unexpectedly: {exc}")


class FakeIncrementalNS:
    """Fake namespace for incremental_run tests (no network)."""
    def __init__(self, codes=None):
        self.codes = codes or []
        self.val_calls = []
        self.stock_day_calls = []
        self.market_month_calls = []

    def fetch_valuation_all(self, date_iso, codes):
        self.val_calls.append(date_iso)
        return {c: {"pe": 18.0, "pb": 3.0, "dividend_yield": 2.5} for c in codes}

    def fetch_stock_day(self, stock_no, month_start):
        self.stock_day_calls.append((stock_no, month_start))
        return []

    def fetch_twse_market_month(self, month_start):
        self.market_month_calls.append(month_start)
        return []

    def fetch_twse_institutional(self, date_iso):
        return {}

    def fetch_twse_margin(self, date_iso):
        return {}

    def fetch_twse_summary(self, date_iso):
        return {}

    def fetch_taifex_inst_futures(self, start_iso, end_iso=None):
        return {}


class TestBackfillStockDayMonth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_upserts_and_marks_ingested(self):
        ns = FakeIncrementalNS(codes=["2330"])
        ns.fetch_stock_day = lambda s, m: [MarketBar(date="2026-05-05", close=800.0,
                                                     open=790.0, high=805.0, low=788.0,
                                                     volume=1e7)]
        budget = backfill.Budget(max_requests=10)
        backfill.backfill_stock_day_month(self.store, ns, "2330", date(2026, 5, 1),
                                          budget, sleep=0)
        rows = self.store.get_stock_daily("2330")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["close"], 800.0)
        self.assertTrue(self.store.is_ingested("stock_day", "stock_day:2330:2026-05"))

    def test_second_run_skips(self):
        ns = FakeIncrementalNS(codes=["2330"])
        budget = backfill.Budget(max_requests=10)
        backfill.backfill_stock_day_month(self.store, ns, "2330", date(2026, 5, 1),
                                          budget, sleep=0)
        n_first = len(ns.stock_day_calls)
        backfill.backfill_stock_day_month(self.store, ns, "2330", date(2026, 5, 1),
                                          budget, sleep=0)
        self.assertEqual(len(ns.stock_day_calls), n_first,
                         "Second run must skip already-ingested stock_day month")


class TestIncrementalRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()
        for d in ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]:
            self.store.upsert_tw_daily({
                "date": d,
                "close": 20000.0,
                "price_asof_taipei": f"{d}T14:30:00+08:00",
                "available_at_taipei": f"{d}T16:30:00+08:00",
            })

    def test_valuation_dates_ingested_after_run(self):
        ns = FakeIncrementalNS(codes=["2330"])
        budget = backfill.Budget(max_requests=50)
        backfill.incremental_run(self.store, ns, start=date(2026, 1, 1),
                                 budget=budget, sleep=0)
        ingested = [d for d in ["2026-01-02", "2026-01-05", "2026-01-06",
                                 "2026-01-07", "2026-01-08"]
                    if self.store.is_ingested("valuation", d)]
        self.assertGreater(len(ingested), 0,
                           "incremental_run must ingest at least one valuation date")

    def test_budget_stops_run_cleanly(self):
        ns = FakeIncrementalNS(codes=["2330"])
        budget = backfill.Budget(max_requests=2)
        try:
            backfill.incremental_run(self.store, ns, start=date(2026, 1, 1),
                                     budget=budget, sleep=0)
        except Exception as exc:
            self.fail(f"incremental_run raised unexpectedly with small budget: {exc}")
        self.assertLessEqual(budget.remaining, 0)

    def test_already_ingested_valuation_dates_skipped(self):
        self.store.mark_ingested("valuation", "2026-01-02")
        self.store.mark_ingested("valuation", "2026-01-05")
        ns = FakeIncrementalNS(codes=["2330"])
        budget = backfill.Budget(max_requests=50)
        backfill.incremental_run(self.store, ns, start=date(2026, 1, 1),
                                 budget=budget, sleep=0)
        self.assertNotIn("2026-01-02", ns.val_calls)
        self.assertNotIn("2026-01-05", ns.val_calls)


class TestBackfillTwBudget(unittest.TestCase):
    """Optional budget threading into backfill_tw is backward compatible."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = HistoryStore(os.path.join(self.tmp, "h.db"))
        self.store.init_schema()

    def test_budget_none_unlimited(self):
        # Default budget=None → behaves as before (no BudgetExhausted).
        backfill.backfill_tw(self.store, FakeFetchers(), [date(2026, 6, 1)], sleep=0)
        self.assertTrue(self.store.is_ingested("tw_month", "2026-06"))

    def test_budget_counts_fetches(self):
        # With a real budget, fetches are counted; tiny budget stops early.
        budget = backfill.Budget(max_requests=1)
        try:
            backfill.backfill_tw(self.store, FakeFetchers(), [date(2026, 6, 1)],
                                 sleep=0, budget=budget)
        except backfill.BudgetExhausted:
            pass
        # Budget consumed at least one request
        self.assertLessEqual(budget.remaining, 0)


if __name__ == "__main__":
    unittest.main()
