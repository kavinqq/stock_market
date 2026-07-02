import os, tempfile, unittest
from marketlib.store import HistoryStore

class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "h.db")
        self.s = HistoryStore(self.db)
        self.s.init_schema()

    def tearDown(self):
        self.s.close()

    def test_upsert_and_query_tw(self):
        self.s.upsert_tw_daily({
            "date": "2026-06-26", "close": 100.0, "turnover": 5e11,
            "inst_total_net_ntd": 1e9, "inst_asof_taipei": "2026-06-26T16:00:00+08:00",
            "price_asof_taipei": "2026-06-26T14:30:00+08:00",
            "available_at_taipei": "2026-06-26T16:00:00+08:00",
        })
        # upsert 同日覆蓋
        self.s.upsert_tw_daily({"date": "2026-06-26", "close": 101.0})
        rows = self.s.get_tw_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["close"], 101.0)
        self.assertEqual(rows[0]["inst_asof_taipei"], "2026-06-26T16:00:00+08:00")

    def test_ingest_cursor(self):
        self.assertFalse(self.s.is_ingested("tw_margin", "2026-06-26"))
        self.s.mark_ingested("tw_margin", "2026-06-26")
        self.assertTrue(self.s.is_ingested("tw_margin", "2026-06-26"))

    def test_date_range_query(self):
        for d in ("2026-06-01", "2026-06-10", "2026-06-20"):
            self.s.upsert_us_daily({"date": d, "spy_close": 1.0})
        rows = self.s.get_us_rows(start="2026-06-05", end="2026-06-15")
        self.assertEqual([r["date"] for r in rows], ["2026-06-10"])

class TestStockTables(unittest.TestCase):
    def setUp(self):
        import tempfile, os
        self.s = HistoryStore(os.path.join(tempfile.mkdtemp(), "h.db")); self.s.init_schema()
    def tearDown(self): self.s.close()
    def test_stock_daily_roundtrip(self):
        self.s.upsert_stock_daily({"stock_no":"2330","date":"2026-07-01","close":1000.0,
            "volume":30000000.0,"inst_total_net_lots":5000.0,
            "price_asof_taipei":"2026-07-01T14:30:00+08:00"})
        self.s.upsert_stock_daily({"stock_no":"2330","date":"2026-07-01","close":1005.0})  # partial
        rows=self.s.get_stock_daily("2330")
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]["close"],1005.0)
        self.assertEqual(rows[0]["inst_total_net_lots"],5000.0)  # preserved
    def test_valuation_and_revenue(self):
        self.s.upsert_stock_valuation({"stock_no":"2330","date":"2026-07-01","pe":20.0,"pb":5.0,"dividend_yield":1.8})
        self.s.upsert_stock_monthly_revenue({"stock_no":"2330","month":"2026-06","revenue_thousand":2.5e8,"yoy_pct":30.0})
        self.assertEqual(self.s.get_stock_valuation("2330")[0]["pe"],20.0)
        self.assertEqual(self.s.get_stock_monthly_revenue("2330")[0]["yoy_pct"],30.0)
    def test_stock_scoping(self):
        self.s.upsert_stock_daily({"stock_no":"2330","date":"2026-07-01","close":1.0})
        self.s.upsert_stock_daily({"stock_no":"2317","date":"2026-07-01","close":2.0})
        self.assertEqual(len(self.s.get_stock_daily("2330")),1)

if __name__ == "__main__":
    unittest.main()
