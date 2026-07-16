import json, os, tempfile, unittest
from daily_market_report import model_prediction
from marketlib.features import FEATURE_NAMES_TW

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


def _indicators(close=100.0):
    # All-neutral IndicatorSet kwargs; enough for build_taiwan_prediction's scoring.
    return {
        "close": close, "day_change_pct": 0.0, "five_day_pct": 0.0,
        "twenty_day_pct": 0.0, "sixty_day_pct": 0.0, "ma5": close, "ma20": close,
        "ma60": close, "volume_ratio_20d": 1.0, "drawdown_from_20d_high": 0.0,
        "drawdown_from_60d_high": 0.0, "trend_label": "中性",
    }


def _prediction_report():
    bars = [{"date": f"2026-07-{d:02d}", "close": 100.0 + d, "turnover": 5e11,
             "volume": 1e9} for d in range(1, 11)]
    return {
        "report_date": "2026-07-10",
        "taiwan": {"latest_date": "2026-07-10", "bars": bars,
                   "indicators": _indicators(), "margin": {}, "institutional": {},
                   "summary": {}},
        "us": {"latest_date": "2026-07-09",
               "series": {"QQQ": {"indicators": _indicators(470.0)}},
               "vix": {"indicators": {"close": 16.0}}},
    }

class TestDaily(unittest.TestCase):
    def test_model_prediction_returns_proba(self):
        d = tempfile.mkdtemp(); path = os.path.join(d, "model_tw.json")
        with open(path, "w") as fh:
            json.dump(MODEL, fh)
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


    def test_archived_eval_skips_us_market(self):
        """US removed from output: evaluate_archived_predictions must only produce
        台股 records even when an archived JSON still carries a 美股 prediction."""
        from daily_market_report import evaluate_archived_predictions
        report = {
            "report_date": "2026-07-03",
            "taiwan": {"bars": [
                {"date": "2026-07-01", "close": 100.0},
                {"date": "2026-07-02", "close": 102.0},
            ]},
            # us block still fetched, but must not be evaluated for scoring
            "us": {"series": {"SPY": {"bars": [
                {"date": "2026-07-01", "close": 500.0},
                {"date": "2026-07-02", "close": 510.0},
            ]}}},
        }
        with tempfile.TemporaryDirectory() as d:
            old = {
                "report_date": "2026-07-01",
                "taiwan": {"latest_date": "2026-07-01"},
                "us": {"latest_date": "2026-07-01"},
                "predictions": {
                    "taiwan": {"market": "台股", "bias": "偏多",
                               "target_date_estimate": "2026-07-02",
                               "neutral_threshold_pct": 0.005,
                               "formula_version": "rule-v1.1", "reasons": []},
                    "us": {"market": "美股", "bias": "偏空",
                           "target_date_estimate": "2026-07-02",
                           "neutral_threshold_pct": 0.005,
                           "formula_version": "rule-v1.1", "reasons": []},
                },
            }
            with open(os.path.join(d, "2026-07-01.json"), "w") as fh:
                json.dump(old, fh)
            out = evaluate_archived_predictions(report, __import__("pathlib").Path(d))
        markets = {r["market"] for r in out["records"]}
        self.assertEqual(markets, {"台股"})
        self.assertNotIn("美股", markets)

    def test_build_predictions_has_no_us_key(self):
        """build_predictions must only return a taiwan prediction (US removed)."""
        from daily_market_report import build_predictions
        preds = build_predictions(_prediction_report())
        self.assertIn("taiwan", preds)
        self.assertNotIn("us", preds)

    def test_sparsity_fallback_returns_none(self):
        """Model with full TW canonical feature list falls back when data is price-only (chip+US features absent)."""
        sparse_model = {
            "market": "tw",
            "feature_names": FEATURE_NAMES_TW,
            "standardization": {
                "mean": {n: 0.0 for n in FEATURE_NAMES_TW},
                "std": {n: 1.0 for n in FEATURE_NAMES_TW},
            },
            "imputation_median": {n: 0.0 for n in FEATURE_NAMES_TW},
            "coefficients": {n: 0.0 for n in FEATURE_NAMES_TW},
            "intercept": 0.0,
            "thresholds": {"bull": 0.6, "bear": 0.4},
        }
        d = tempfile.mkdtemp()
        path = os.path.join(d, "model_tw.json")
        with open(path, "w") as fh:
            json.dump(sparse_model, fh)
        # _rows() is price-only: no chip or US columns -> fewer than half of the
        # 20 canonical TW features are present -> sparsity guard returns None.
        self.assertIsNone(model_prediction("tw", path, _rows(), []))


def _bars():
    from marketlib.core import MarketBar
    from datetime import date, timedelta
    base = date(2026, 1, 2)
    # Generate 80 consecutive calendar days (including weekends, which resample_weekly handles)
    return [
        MarketBar(
            date=(base + timedelta(days=m)).isoformat(),
            close=1000 + (m % 10),
            high=1010.0,
            low=990.0,
            volume=3e7,
        )
        for m in range(80)
    ]


class TestPickQueryDate(unittest.TestCase):
    """Unit tests for _pick_query_date date-selection helper (Bug B fix)."""

    def _make_bars(self, dates):
        from marketlib.core import MarketBar
        return [MarketBar(date=d, close=1000.0) for d in dates]

    def test_returns_latest_when_data_available(self):
        from daily_market_report import _pick_query_date
        bars = self._make_bars(["2026-07-01", "2026-07-02"])
        calls = []

        def fetch(stock_no, date_iso):
            calls.append(date_iso)
            return {"pe": 20.0} if date_iso == "2026-07-02" else None

        date_used, result = _pick_query_date(bars, fetch, "2330")
        self.assertEqual(date_used, "2026-07-02")
        self.assertIsNotNone(result)
        self.assertEqual(result["pe"], 20.0)

    def test_falls_back_to_prior_bar_when_latest_is_none(self):
        from daily_market_report import _pick_query_date
        bars = self._make_bars(["2026-06-30", "2026-07-01", "2026-07-02"])

        def fetch(stock_no, date_iso):
            # Latest returns None (not yet posted), prior returns data
            if date_iso == "2026-07-01":
                return {"pe": 18.5}
            return None

        date_used, result = _pick_query_date(bars, fetch, "2330")
        self.assertEqual(date_used, "2026-07-01")
        self.assertIsNotNone(result)
        self.assertEqual(result["pe"], 18.5)

    def test_returns_latest_date_when_all_none(self):
        from daily_market_report import _pick_query_date
        bars = self._make_bars(["2026-06-30", "2026-07-01", "2026-07-02"])

        def fetch(stock_no, date_iso):
            return None

        date_used, result = _pick_query_date(bars, fetch, "2330")
        self.assertEqual(date_used, "2026-07-02")
        self.assertIsNone(result)


class TestStock2330(unittest.TestCase):
    def test_assemble_complete_with_missing_revenue(self):
        from daily_market_report import assemble_2330
        r = assemble_2330(
            _bars(),
            {"pe": 20.0, "pb": 5.0, "dividend_yield": 1.8, "pe_pct": 0.3, "pb_pct": 0.3},
            None,
            {"inst_total_net_lots": 3000.0},
            {"smh_20d": 0.05, "adr_day": 0.01},
            {"index_trend": "多頭", "vix": 16.0},
        )
        self.assertIn("short_lean", r)
        self.assertIn("medium_lean", r)
        self.assertIsNone(r["facets"]["revenue"]["score"])   # missing source
        self.assertIn(r["short_lean"]["bias"], ("偏多", "中性", "偏空"))


class TestBuildImprovementLog(unittest.TestCase):
    """Tests for build_improvement_log: prediction/actual_direction keys and de-dup."""

    def _make_bt(self, misses):
        """Wrap a list of miss dicts into a backtest-like dict."""
        return {"recent_misses": misses}

    def _make_archive(self, records):
        """Wrap a list of records into an archive_eval-like dict."""
        return {"records": records}

    def test_rows_contain_prediction_and_actual_direction(self):
        """Each row in improvement_log must have prediction and actual_direction."""
        from daily_market_report import build_improvement_log
        tw_bt = self._make_bt([
            {
                "target_date": "2026-06-10",
                "miss_reason": "公式門檻過保守",
                "prediction": "震盪",
                "actual_direction": "偏多",
            }
        ])
        us_bt = self._make_bt([])
        archive = self._make_archive([])
        rows = build_improvement_log(tw_bt, us_bt, archive)
        self.assertEqual(len(rows), 1)
        self.assertIn("prediction", rows[0])
        self.assertIn("actual_direction", rows[0])
        self.assertEqual(rows[0]["prediction"], "震盪")
        self.assertEqual(rows[0]["actual_direction"], "偏多")

    def test_archive_rows_contain_prediction_and_actual_direction(self):
        """Archive miss records must also carry prediction/actual_direction."""
        from daily_market_report import build_improvement_log
        tw_bt = self._make_bt([])
        us_bt = self._make_bt([])
        archive = self._make_archive([
            {
                "hit": False,
                "market": "美股",
                "target_date": "2026-06-05",
                "miss_reason": "缺少資料源",
                "prediction": "偏多",
                "actual_direction": "偏空",
            }
        ])
        rows = build_improvement_log(tw_bt, us_bt, archive)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "偏多")
        self.assertEqual(rows[0]["actual_direction"], "偏空")

    def test_duplicate_misses_collapse_to_one_row(self):
        """Identical misses on (market, date, prediction, actual_direction, reason) dedup to one row."""
        from daily_market_report import build_improvement_log
        miss = {
            "target_date": "2026-06-10",
            "miss_reason": "公式門檻過保守",
            "prediction": "震盪",
            "actual_direction": "偏多",
        }
        # Same miss appears in both recent_misses (last 5) → should produce one row
        tw_bt = self._make_bt([miss, miss])
        us_bt = self._make_bt([])
        # Also duplicate it in archive
        archive_miss = {
            "hit": False,
            "market": "台股",
            "target_date": "2026-06-10",
            "miss_reason": "公式門檻過保守",
            "prediction": "震盪",
            "actual_direction": "偏多",
        }
        archive = self._make_archive([archive_miss])
        rows = build_improvement_log(tw_bt, us_bt, archive)
        # All three map to the same (market=台股, date=2026-06-10, ...) key → 1 row
        self.assertEqual(len(rows), 1, f"Expected 1 after dedup, got {len(rows)}: {rows}")

    def test_cap_at_12_after_dedup(self):
        """Result is capped at 12 rows after deduplication."""
        from daily_market_report import build_improvement_log
        misses = [
            {
                "target_date": f"2026-06-{i:02d}",
                "miss_reason": "缺少資料源",
                "prediction": "偏多",
                "actual_direction": "偏空",
            }
            for i in range(1, 20)
        ]
        tw_bt = self._make_bt(misses)
        us_bt = self._make_bt(misses)
        archive = self._make_archive([])
        rows = build_improvement_log(tw_bt, us_bt, archive)
        self.assertLessEqual(len(rows), 12)


class TestScreener0050(unittest.TestCase):
    def test_assemble_ranks(self):
        codes = ["A", "B"]
        val = {"A": {"pe": 10, "pb": 1, "dividend_yield": 3}, "B": {"pe": 40, "pb": 8, "dividend_yield": 0.5}}
        rev = {"A": {"yoy_pct": 30}, "B": {"yoy_pct": -5}}
        chip = {"A": {"inst_total_net_lots": 5000}, "B": {"inst_total_net_lots": -2000}}
        tech = {"A": {"tech_raw": 1.0, "ret20": 0.05, "ret60": 0.1}, "B": {"tech_raw": 0.0, "ret20": -0.03, "ret60": -0.05}}
        from daily_market_report import assemble_screener
        out = assemble_screener(codes, val, rev, chip, tech)
        self.assertEqual(out[0]["code"], "A")
        self.assertEqual(out[0]["rank"], 1)

    def test_missing_revenue_still_ranked(self):
        codes = ["A", "B"]
        val = {"A": {"pe": 10, "pb": 1, "dividend_yield": 3}, "B": {"pe": 20, "pb": 2, "dividend_yield": 2}}
        from daily_market_report import assemble_screener
        out = assemble_screener(codes, val, {}, {}, {"A": {"tech_raw": 1.0, "ret20": 0.05}, "B": {"tech_raw": 0.0, "ret20": 0.0}})
        self.assertEqual(len(out), 2)
        self.assertIsNotNone(out[0]["composite"])


class TestScreenerNames(unittest.TestCase):
    """assemble_screener must propagate 'name' from valuation/chip sources."""

    def _make_val(self, name: str) -> dict:
        return {"pe": 15.0, "pb": 2.0, "dividend_yield": 2.0, "name": name}

    def _make_chip(self, name: str) -> dict:
        return {"inst_total_net_lots": 1000.0, "name": name}

    def test_name_from_valuation_source(self):
        from daily_market_report import assemble_screener
        codes = ["2330", "2303"]
        val = {"2330": self._make_val("台積電"), "2303": self._make_val("聯電")}
        out = assemble_screener(codes, val, {}, {}, {})
        by_code = {r["code"]: r for r in out}
        self.assertEqual(by_code["2330"]["name"], "台積電")
        self.assertEqual(by_code["2303"]["name"], "聯電")

    def test_name_from_chip_source_when_val_missing(self):
        from daily_market_report import assemble_screener
        codes = ["2454"]
        chip = {"2454": self._make_chip("聯發科")}
        out = assemble_screener(codes, {}, {}, chip, {})
        self.assertEqual(out[0]["name"], "聯發科")

    def test_name_falls_back_to_code_when_unknown(self):
        from daily_market_report import assemble_screener
        codes = ["XXXX"]
        out = assemble_screener(codes, {}, {}, {}, {})
        self.assertEqual(out[0]["name"], "XXXX")

    def test_name_present_in_existing_rank_test(self):
        """Ensure name key exists even in existing test scenario (no regression)."""
        from daily_market_report import assemble_screener
        codes = ["A", "B"]
        val = {"A": {"pe": 10, "pb": 1, "dividend_yield": 3}, "B": {"pe": 40, "pb": 8, "dividend_yield": 0.5}}
        out = assemble_screener(codes, val, {}, {}, {})
        for row in out:
            self.assertIn("name", row)


class TestStock2330TechnicalBlock(unittest.TestCase):
    """html_stock_2330_section must render the technical indicators block."""

    def _make_synthetic_stock(self) -> dict:
        """Build a minimal synthetic stock_2330 dict with technical_metrics populated."""
        return {
            "facets": {
                "technical": {"score": 0.5, "reason": "OK"},
                "volume": {"score": 0.3, "reason": "OK"},
                "chip": {"score": 0.1, "reason": "OK"},
                "valuation": {"score": -0.2, "reason": "OK"},
                "revenue": {"score": None, "reason": "缺資料"},
                "peer": {"score": 0.0, "reason": "OK"},
                "market": {"score": 0.2, "reason": "OK"},
            },
            "short_lean": {"bias": "偏多", "confidence": "中", "score": 0.4},
            "medium_lean": {"bias": "偏多", "confidence": "低", "score": 0.3},
            "missing": ["revenue"],
            "observe_levels": {
                "close": 980.0,
                "support_20d": 950.0,
                "resistance_20d": 1000.0,
                "ma5": 975.0,
                "ma20": 960.0,
                "ma60": 930.0,
            },
            "technical_metrics": {
                "close": 980.0,
                "ma5": 975.0,
                "ma20": 960.0,
                "ma60": 930.0,
                "ma120": 900.0,
                "ma240": 850.0,
                "boll": {"mid": 960.0, "upper": 1010.0, "lower": 910.0, "pctb": 0.7},
                "macd_daily": {"dif": 5.0, "dea": 3.0, "hist": 2.0, "cross": "golden"},
                "macd_weekly": {"dif": 8.0, "dea": 6.0, "hist": 2.0, "cross": None},
                "macd_monthly": {"dif": 12.0, "dea": 10.0, "hist": 2.0, "cross": None},
                "kd_k": 65.0,
                "kd_d": 60.0,
                "kd": {"k": 65.0, "d": 60.0},
                "cost_vwap60": 940.0,
                "cost_vwap120": 920.0,
                "price_vs_cost60": 0.043,
                "rsi": 58.0,
                "ret5": 0.01,
                "ret20": 0.02,
                "ret60": 0.05,
                "drawdown": -0.02,
                "volume_ratio_20d": 1.1,
            },
        }

    def test_renders_ma_labels(self):
        from daily_market_report import html_stock_2330_section
        html = html_stock_2330_section(self._make_synthetic_stock())
        self.assertIn("年線", html)
        self.assertIn("半年線", html)
        self.assertIn("季線", html)
        self.assertIn("月線", html)
        self.assertIn("週線", html)

    def test_renders_boll_label(self):
        from daily_market_report import html_stock_2330_section
        html = html_stock_2330_section(self._make_synthetic_stock())
        self.assertIn("布林", html)

    def test_renders_cost_label(self):
        from daily_market_report import html_stock_2330_section
        html = html_stock_2330_section(self._make_synthetic_stock())
        self.assertIn("大戶成本", html)
        self.assertIn("估計", html)

    def test_renders_macd_labels(self):
        from daily_market_report import html_stock_2330_section
        html = html_stock_2330_section(self._make_synthetic_stock())
        self.assertIn("MACD", html)

    def test_renders_kd_label(self):
        from daily_market_report import html_stock_2330_section
        html = html_stock_2330_section(self._make_synthetic_stock())
        self.assertIn("KD", html)

    def test_missing_technical_metrics_degrades_gracefully(self):
        """stock_2330 without technical_metrics key should still render without crash."""
        from daily_market_report import html_stock_2330_section
        stock = self._make_synthetic_stock()
        del stock["technical_metrics"]
        html = html_stock_2330_section(stock)
        self.assertIn("台積電", html)
        self.assertIn("N/A", html)

    def test_none_indicators_show_na(self):
        """None values in technical_metrics must render as N/A."""
        from daily_market_report import html_stock_2330_section
        stock = self._make_synthetic_stock()
        stock["technical_metrics"]["ma240"] = None
        stock["technical_metrics"]["boll"] = None
        stock["technical_metrics"]["cost_vwap60"] = None
        html = html_stock_2330_section(stock)
        self.assertIn("N/A", html)


if __name__ == "__main__":
    unittest.main()
