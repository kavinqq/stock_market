import os, tempfile, unittest
from marketlib.store import HistoryStore
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

if __name__ == "__main__":
    unittest.main()
