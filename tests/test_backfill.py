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


if __name__ == "__main__":
    unittest.main()
