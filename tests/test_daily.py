import json, os, tempfile, unittest
from daily_market_report import model_prediction

MODEL = {
    "market": "tw", "feature_names": ["ret_1d"],
    "standardization": {"mean": {"ret_1d": 0.0}, "std": {"ret_1d": 1.0}},
    "imputation_median": {"ret_1d": 0.0}, "coefficients": {"ret_1d": 5.0},
    "intercept": 0.0, "thresholds": {"bull": 0.6, "bear": 0.4},
    "confidence_bands": {"mid": 0.05, "high": 0.2},
}

def _rows():
    return [{"date": f"2026-06-{d:02d}", "close": 100 + d, "volume": 1e9,
             "turnover": 5e11, "price_asof_taipei": f"2026-06-{d:02d}T14:30:00+08:00"}
            for d in range(1, 28)]

class TestDaily(unittest.TestCase):
    def test_model_prediction_returns_proba(self):
        d = tempfile.mkdtemp(); path = os.path.join(d, "model_tw.json")
        json.dump(MODEL, open(path, "w"))
        out = model_prediction("tw", path, _rows(), [])
        self.assertIn(out["bias"], ("偏多", "震盪", "偏空"))
        self.assertTrue(0.0 <= out["probability"] <= 1.0)

    def test_missing_model_returns_none(self):
        self.assertIsNone(model_prediction("tw", "/no/such.json", _rows(), []))

    def test_upsert_report_into_store(self):
        """upsert_report_into_store maps report data to store rows correctly."""
        from daily_market_report import upsert_report_into_store
        from marketlib.store import HistoryStore

        # Build a minimal report dict that mirrors what build_taiwan_report /
        # build_us_report produce.
        tw_date = "2026-06-15"
        us_date = "2026-06-14"

        report = {
            "report_date": tw_date,
            "taiwan": {
                "latest_date": tw_date,
                "bars": [
                    {"date": tw_date, "close": 21000.0, "turnover": 3e11,
                     "volume": 5000.0},
                ],
                "margin": {
                    "margin_financing": {
                        "today_balance_ntd": 2e11,
                        "balance_change_ntd": 1e9,
                    }
                },
                "institutional": {
                    "rows": {
                        "合計": {"net_ntd": 5e9},
                    }
                },
                "summary": {
                    "breadth": {
                        "advance_decline_ratio_stocks": 1.5,
                        "上漲": {"stocks": 800},
                        "下跌": {"stocks": 600},
                    }
                },
            },
            "us": {
                "latest_date": us_date,
                "series": {
                    "SPY": {
                        "bars": [{"date": us_date, "close": 550.0, "volume": 8e7}],
                    },
                    "QQQ": {
                        "bars": [{"date": us_date, "close": 470.0, "volume": 4e7}],
                    },
                    "IWM": {
                        "bars": [{"date": us_date, "close": 210.0, "volume": 2e7}],
                    },
                    "COMP": {
                        "bars": [{"date": us_date, "close": 18000.0, "volume": 0}],
                    },
                },
                "vix": {
                    "bars": [{"date": us_date, "close": 16.0}],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "history.db")
            store = HistoryStore(db_path)
            store.init_schema()
            upsert_report_into_store(report, store)

            tw_rows = store.get_tw_rows()
            self.assertEqual(len(tw_rows), 1)
            row = tw_rows[0]
            self.assertEqual(row["date"], tw_date)
            self.assertIsNotNone(row["price_asof_taipei"],
                                 "price_asof_taipei must be populated")
            self.assertEqual(row["inst_total_net_ntd"], 5e9)
            self.assertEqual(row["margin_balance_ntd"], 2e11)
            self.assertIsNotNone(row["available_at_taipei"])

            us_rows = store.get_us_rows()
            self.assertGreaterEqual(len(us_rows), 1)
            us_row = us_rows[0]
            self.assertEqual(us_row["date"], us_date)
            self.assertIsNotNone(us_row["price_asof_taipei"])
            self.assertEqual(us_row["spy_close"], 550.0)
            store.close()


if __name__ == "__main__":
    unittest.main()
