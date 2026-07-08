import unittest
from marketlib.stocks import _parse_stock_day, _parse_t86, _parse_bwibbu, _parse_monthly_revenue
from marketlib.stocks import ema, macd, rsi, resample_weekly, technical_metrics
from marketlib.stocks import percentile_rank, valuation_metrics, revenue_metrics
from marketlib.stocks import _parse_bwibbu_all, _parse_t86_all, tech_raw_score
from marketlib.stocks import bollinger, vwap, resample_monthly
from marketlib.core import MarketBar
from datetime import date, timedelta


def _series(closes, vols=None):
    b = date(2026, 1, 1)
    vols = vols or [1e6] * len(closes)
    return [
        MarketBar(
            date=(b + timedelta(days=i)).isoformat(),
            close=float(c),
            high=float(c),
            low=float(c),
            volume=float(v),
        )
        for i, (c, v) in enumerate(zip(closes, vols))
    ]


class TestIndicators(unittest.TestCase):
    def test_bollinger_mid_is_sma(self):
        out = bollinger([10] * 19 + [10], 20)  # all 10 -> stdev 0 -> pctb None, mid 10
        self.assertAlmostEqual(out["mid"], 10.0)
        self.assertIsNone(out["pctb"])

    def test_bollinger_pctb(self):
        out = bollinger([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 40], 20)
        self.assertGreater(out["upper"], out["mid"])
        self.assertTrue(0 <= out["pctb"] <= 1.5)

    def test_vwap(self):
        # closes [10,20], vols [1,3] -> (10*1+20*3)/4 = 17.5
        self.assertAlmostEqual(vwap(_series([10, 20], [1, 3]), 2), 17.5)

    def test_resample_monthly(self):
        bars = [
            MarketBar(date=d, close=c, high=c, low=c, volume=1.0)
            for d, c in [("2026-01-05", 1), ("2026-01-20", 2), ("2026-02-10", 3)]
        ]
        m = resample_monthly(bars)
        self.assertEqual(len(m), 2)
        self.assertEqual(m[0].close, 2)

    def test_technical_metrics_new_keys(self):
        tm = technical_metrics(_series(list(range(1, 130))))
        for k in ("ma120", "boll", "cost_vwap60", "price_vs_cost60", "macd_monthly"):
            self.assertIn(k, tm)

    def test_tech_raw_uses_new_signals(self):
        self.assertGreater(tech_raw_score(_series(list(range(1, 130)))), 0.5)


class TestStockParsers(unittest.TestCase):
    def test_stock_day(self):
        payload = {
            "stat": "OK",
            "fields": ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"],
            "data": [["115/07/01", "30,000,000", "3.0E10", "1000", "1010", "995", "1005", "+5", "50000"]],
        }
        bars = _parse_stock_day(payload)
        self.assertEqual(bars[0].date, "2026-07-01")
        self.assertEqual(bars[0].close, 1005.0)
        self.assertEqual(bars[0].volume, 30000000.0)

    def test_stock_day_bad_stat_returns_empty(self):
        payload = {"stat": "FAIL", "fields": [], "data": []}
        bars = _parse_stock_day(payload)
        self.assertEqual(bars, [])

    def test_t86_maps_by_header(self):
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外陸資買賣超股數(不含外資自營商)",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [["2330", "台積電", "5000000", "1000000", "-2000000", "4000000"]],
        }
        out = _parse_t86(payload, "2330")
        self.assertEqual(out["foreign_net_lots"], 5000.0)   # 5,000,000 / 1000
        self.assertEqual(out["trust_net_lots"], 1000.0)
        self.assertEqual(out["dealer_net_lots"], -2000.0)
        self.assertEqual(out["inst_total_net_lots"], 4000.0)
        self.assertIsNone(_parse_t86(payload, "9999"))

    def test_t86_fallback_foreign_field(self):
        # Test with alternative foreign field name
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外資買賣超股數",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [["0050", "元大台灣50", "2000000", "500000", "300000", "2800000"]],
        }
        out = _parse_t86(payload, "0050")
        self.assertEqual(out["foreign_net_lots"], 2000.0)
        self.assertEqual(out["inst_total_net_lots"], 2800.0)

    def test_t86_bad_stat_returns_none(self):
        payload = {"stat": "FAIL", "fields": [], "data": []}
        self.assertIsNone(_parse_t86(payload, "2330"))

    def test_bwibbu(self):
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [["2330", "台積電", "1.80", "2025", "20.5", "5.2", "2025/1"]],
        }
        out = _parse_bwibbu(payload, "2330")
        self.assertEqual(out["pe"], 20.5)
        self.assertEqual(out["pb"], 5.2)
        self.assertEqual(out["dividend_yield"], 1.80)

    def test_bwibbu_stock_not_found_returns_none(self):
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [["2330", "台積電", "1.80", "2025", "20.5", "5.2", "2025/1"]],
        }
        self.assertIsNone(_parse_bwibbu(payload, "9999"))

    def test_bwibbu_bad_stat_returns_none(self):
        payload = {"stat": "FAIL", "fields": [], "data": []}
        self.assertIsNone(_parse_bwibbu(payload, "2330"))

    def test_monthly_revenue(self):
        # Field names must use REAL TWSE prefixed keys from t187ap05_L
        items = [
            {
                "公司代號": "2330",
                "資料年月": "11506",
                "營業收入-當月營收": "250000000",
                "營業收入-去年同月增減(%)": "30.09",
                "營業收入-上月比較增減(%)": "1.52",
                "累計營業收入-前期比較增減(%)": "29.98",
            }
        ]
        out = _parse_monthly_revenue(items, "2330")
        self.assertEqual(out["month"], "2026-06")
        self.assertEqual(out["revenue_thousand"], 250000000.0)
        self.assertAlmostEqual(out["yoy_pct"], 30.09)
        self.assertAlmostEqual(out["mom_pct"], 1.52)
        self.assertAlmostEqual(out["cum_yoy_pct"], 29.98)

    def test_monthly_revenue_stock_not_found_returns_none(self):
        items = [{"公司代號": "2330", "資料年月": "11506", "營業收入-當月營收": "250000000",
                  "營業收入-去年同月增減(%)": "30.09", "營業收入-上月比較增減(%)": "1.52",
                  "累計營業收入-前期比較增減(%)": "29.98"}]
        self.assertIsNone(_parse_monthly_revenue(items, "9999"))

    def test_monthly_revenue_roc_date_conversion(self):
        # ROC 11412 -> year=114+1911=2025, month=12 -> "2025-12"
        items = [{"公司代號": "0050", "資料年月": "11412", "營業收入-當月營收": "1000",
                  "營業收入-去年同月增減(%)": "5.0", "營業收入-上月比較增減(%)": "1.0",
                  "累計營業收入-前期比較增減(%)": "4.0"}]
        out = _parse_monthly_revenue(items, "0050")
        self.assertEqual(out["month"], "2025-12")


class TestStockTech(unittest.TestCase):
    def test_ema_matches_manual(self):
        e = ema([1, 2, 3, 4, 5], 3)  # span3 alpha=0.5, seed 1
        self.assertAlmostEqual(e[0], 1.0)
        self.assertAlmostEqual(e[1], 1.5)
        self.assertAlmostEqual(e[2], 2.25)

    def test_macd_cross_golden(self):
        # rising series -> dif crosses above dea -> golden at the crossing tail
        closes = [10] * 30 + [10 + i for i in range(1, 20)]
        m = macd(closes)
        self.assertIsNotNone(m)
        self.assertIn(m["cross"], ("golden", "dead", None))
        self.assertGreater(m["dif"], m["dea"])  # uptrend: dif above dea

    def test_rsi_bounds(self):
        up = list(range(1, 40))
        self.assertGreater(rsi(up), 70)       # all-up -> high RSI
        down = list(range(40, 1, -1))
        self.assertLess(rsi(down), 30)        # all-down -> low RSI

    def test_resample_weekly_counts(self):
        bars = [
            MarketBar(date=f"2026-06-{d:02d}", close=float(d),
                      high=float(d), low=float(d), volume=1.0)
            for d in range(1, 15)
        ]
        wk = resample_weekly(bars)
        self.assertTrue(len(wk) < len(bars) and len(wk) >= 2)
        self.assertEqual(wk[-1].close, bars[-1].close)  # last week's close = last daily close

    def test_macd_insufficient_data_returns_none(self):
        self.assertIsNone(macd([10] * 25))  # need at least 26

    def test_rsi_insufficient_data_returns_none(self):
        self.assertIsNone(rsi([1, 2, 3]))  # need at least 14+1

    def test_kd_basic(self):
        from marketlib.stocks import kd
        highs = list(range(100, 160))
        lows = list(range(90, 150))
        closes = list(range(95, 155))
        result = kd(highs, lows, closes)
        self.assertIsNotNone(result)
        self.assertIn("k", result)
        self.assertIn("d", result)
        self.assertGreaterEqual(result["k"], 0)
        self.assertLessEqual(result["k"], 100)

    def test_technical_metrics_returns_dict(self):
        bars = [
            MarketBar(date=f"2026-0{1 + i // 28:01d}-{1 + i % 28:02d}",
                      close=100.0 + i * 0.5,
                      high=101.0 + i * 0.5,
                      low=99.0 + i * 0.5,
                      volume=1000.0 + i)
            for i in range(80)
        ]
        m = technical_metrics(bars)
        self.assertIsInstance(m, dict)
        self.assertIn("close", m)
        self.assertIn("rsi", m)
        self.assertIn("macd_daily", m)
        self.assertIn("macd_weekly", m)
        self.assertIn("volume_ratio_20d", m)


class TestStockFundamental(unittest.TestCase):
    def test_percentile_rank(self):
        self.assertAlmostEqual(percentile_rank([10, 20, 30, 40], 25), 0.5)
        self.assertAlmostEqual(percentile_rank([10, 20, 30, 40], 10), 0.25)
        self.assertIsNone(percentile_rank([], 5))

    def test_valuation_pct(self):
        # Build a history of exactly MIN_VALUATION_HISTORY points so the guard passes.
        # 60 evenly-spaced values from 10 to 40; value=15 → 2nd out of 5 buckets → ~pct 0.4.
        # With 60 points: values 10,10.5,...40 → pe=15 sits at roughly the lower 30%.
        from marketlib.stocks import MIN_VALUATION_HISTORY
        import statistics
        # Use 60 values: 12 copies each of [10, 15, 20, 25, 30] to hit exactly 0.4 percentile.
        repeats = MIN_VALUATION_HISTORY // 5  # 12
        pe_hist = [10, 15, 20, 25, 30] * repeats  # 60 items
        pb_hist = [2, 3, 4, 5, 6] * repeats
        out = valuation_metrics(
            {"pe": 15.0, "pb": 3.0, "dividend_yield": 2.0},
            pe_hist,
            pb_hist,
        )
        self.assertEqual(out["pe"], 15.0)
        self.assertAlmostEqual(out["pe_pct"], 0.4)  # 2/5 <=15 (24 of 60)


class TestScorecard(unittest.TestCase):
    def test_combine_lean_bias(self):
        from marketlib.stocks import combine_lean
        out = combine_lean({"a": 0.8, "b": 0.6, "c": None}, {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(out["bias"], "偏多")
        self.assertEqual(out["confidence"], "高")
        self.assertNotIn("c", str(out["reasons"]))  # None facet ignored

    def test_combine_lean_neutral_and_missing_renorm(self):
        from marketlib.stocks import combine_lean
        # Neutral case (no None)
        out = combine_lean({"a": 0.1, "b": -0.1}, {"a": 1.0, "b": 1.0})
        self.assertEqual(out["bias"], "中性")
        # Renormalisation: b=None → only facet "a" contributes; weight renorms to 1.0
        # so score = 0.6 * 0.5 / 0.5 = 0.6, NOT 0.6 * 0.5 = 0.3
        out2 = combine_lean({"a": 0.6, "b": None}, {"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(out2["score"], 0.6)
        self.assertEqual(out2["bias"], "偏多")

    def test_chip_facet_sign_correctness(self):
        """Bearish chip (net selling) must yield score < 0; bullish must yield score > 0."""
        from marketlib.stocks import chip_facet
        # Net-selling input: 5 days each selling 2000 lots = total −10000 lots
        selling = [{"inst_total_net_lots": -2000.0} for _ in range(5)]
        f_sell = chip_facet(selling)
        self.assertIsNotNone(f_sell["score"])
        self.assertLess(f_sell["score"], 0, "Net-selling chip should produce a negative score")
        # Net-buying input: 5 days each buying 2000 lots = total +10000 lots
        buying = [{"inst_total_net_lots": 2000.0} for _ in range(5)]
        f_buy = chip_facet(buying)
        self.assertIsNotNone(f_buy["score"])
        self.assertGreater(f_buy["score"], 0, "Net-buying chip should produce a positive score")

    def test_valuation_facet_cheap_is_bullish(self):
        from marketlib.stocks import valuation_facet
        f = valuation_facet({"pe": 15.0, "pb": 3.0, "dividend_yield": 2.0, "pe_pct": 0.1, "pb_pct": 0.1})
        self.assertGreater(f["score"], 0)

    def test_facet_missing_returns_none(self):
        from marketlib.stocks import valuation_facet
        self.assertIsNone(valuation_facet(None)["score"])

    def test_valuation_metrics_short_history_yields_none_pct(self):
        """valuation_metrics with < MIN_VALUATION_HISTORY points → pe_pct/pb_pct None."""
        from marketlib.stocks import valuation_metrics, MIN_VALUATION_HISTORY
        short_history = list(range(1, MIN_VALUATION_HISTORY))  # 59 points, one short
        out = valuation_metrics(
            {"pe": 20.0, "pb": 5.0, "dividend_yield": 1.5},
            short_history,
            short_history,
        )
        self.assertIsNone(out["pe_pct"], "pe_pct must be None for thin history")
        self.assertIsNone(out["pb_pct"], "pb_pct must be None for thin history")
        # Raw values are still passed through
        self.assertEqual(out["pe"], 20.0)
        self.assertEqual(out["pb"], 5.0)

    def test_valuation_metrics_long_history_yields_pct(self):
        """valuation_metrics with >= MIN_VALUATION_HISTORY points → pe_pct computed."""
        from marketlib.stocks import valuation_metrics, MIN_VALUATION_HISTORY
        # Use exactly MIN_VALUATION_HISTORY points
        history = list(range(1, MIN_VALUATION_HISTORY + 1))  # 60 points
        out = valuation_metrics(
            {"pe": 30.0, "pb": 3.0, "dividend_yield": 2.0},
            history,
            history,
        )
        self.assertIsNotNone(out["pe_pct"], "pe_pct must be computed with sufficient history")
        self.assertIsNotNone(out["pb_pct"])

    def test_valuation_facet_thin_history_score_none_metrics_present(self):
        """valuation_facet with pe_pct=None (thin history) → score None, metrics still present."""
        from marketlib.stocks import valuation_facet
        # Simulate output of valuation_metrics with thin history: pe_pct/pb_pct are None
        # but raw pe/pb are present.
        f = valuation_facet({"pe": 20.0, "pb": 5.0, "dividend_yield": 1.5, "pe_pct": None, "pb_pct": None})
        self.assertIsNone(f["score"], "score must be None when history is insufficient")
        self.assertIn("估值歷史不足", f["reason"])
        self.assertIn("pe", f["metrics"])
        self.assertIn("pb", f["metrics"])
        self.assertEqual(f["metrics"]["pe"], 20.0)


class TestBatch(unittest.TestCase):
    def test_bwibbu_all(self):
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [
                ["2330", "台積電", "1.8", "2025", "20.5", "5.2", "2025/1"],
                ["2317", "鴻海", "4.0", "2025", "12.0", "1.5", "2025/1"],
            ],
        }
        out = _parse_bwibbu_all(payload, {"2330", "2317"})
        self.assertEqual(out["2330"]["pe"], 20.5)
        self.assertEqual(out["2317"]["pb"], 1.5)

    def test_bwibbu_all_skips_missing_codes(self):
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [["2330", "台積電", "1.8", "2025", "20.5", "5.2", "2025/1"]],
        }
        out = _parse_bwibbu_all(payload, {"2330", "9999"})
        self.assertIn("2330", out)
        self.assertNotIn("9999", out)

    def test_t86_all(self):
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外陸資買賣超股數(不含外資自營商)",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [
                ["2330", "台積電", "5000000", "1000000", "-2000000", "4000000"],
                ["2317", "鴻海", "2000000", "500000", "300000", "2800000"],
            ],
        }
        out = _parse_t86_all(payload, {"2330", "2317"})
        self.assertEqual(out["2330"]["foreign_net_lots"], 5000.0)
        self.assertEqual(out["2317"]["inst_total_net_lots"], 2800.0)

    def test_t86_all_skips_missing_codes(self):
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外陸資買賣超股數(不含外資自營商)",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [["2330", "台積電", "5000000", "1000000", "-2000000", "4000000"]],
        }
        out = _parse_t86_all(payload, {"2330", "9999"})
        self.assertIn("2330", out)
        self.assertNotIn("9999", out)

    def test_monthly_revenue_all(self):
        from marketlib.stocks import _parse_monthly_revenue_all
        items = [
            {
                "公司代號": "2330",
                "資料年月": "11506",
                "營業收入-當月營收": "250000000",
                "營業收入-去年同月增減(%)": "30.09",
                "營業收入-上月比較增減(%)": "1.52",
                "累計營業收入-前期比較增減(%)": "29.98",
            },
            {
                "公司代號": "2317",
                "資料年月": "11506",
                "營業收入-當月營收": "50000000",
                "營業收入-去年同月增減(%)": "5.0",
                "營業收入-上月比較增減(%)": "2.0",
                "累計營業收入-前期比較增減(%)": "4.5",
            },
        ]
        out = _parse_monthly_revenue_all(items, {"2330", "2317"})
        self.assertEqual(out["2330"]["month"], "2026-06")
        self.assertAlmostEqual(out["2317"]["yoy_pct"], 5.0)

    def test_tech_raw_direction(self):
        from datetime import date, timedelta
        base = date(2025, 1, 1)
        up = [MarketBar(date=(base + timedelta(days=i)).isoformat(), close=100+i, high=100+i, low=100+i, volume=1e6) for i in range(80)]
        self.assertGreater(tech_raw_score(up), 0.5)

    def test_tech_raw_falling(self):
        from datetime import date, timedelta
        base = date(2025, 1, 1)
        down = [MarketBar(date=(base + timedelta(days=i)).isoformat(), close=200-i, high=200-i, low=200-i, volume=1e6) for i in range(80)]
        self.assertLess(tech_raw_score(down), 0.5)

    def test_tech_raw_insufficient_bars_returns_none(self):
        tiny = [MarketBar(date=f"2026-01-{i+1:02d}", close=100+i, high=100+i, low=100+i, volume=1e6) for i in range(3)]
        self.assertIsNone(tech_raw_score(tiny))


class TestNamesAndResolve(unittest.TestCase):
    """Task 2: name field in batch parsers + resolve_name helper."""

    def _bwibbu_payload(self):
        return {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [
                ["2330", "台積電", "1.8", "2025", "20.5", "5.2", "2025/1"],
                ["2317", "鴻海", "4.0", "2025", "12.0", "1.5", "2025/1"],
            ],
        }

    def _t86_payload(self):
        return {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外陸資買賣超股數(不含外資自營商)",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [
                ["2330", "台積電", "5000000", "1000000", "-2000000", "4000000"],
                ["2317", "鴻海", "2000000", "500000", "300000", "2800000"],
            ],
        }

    def test_bwibbu_all_includes_name(self):
        out = _parse_bwibbu_all(self._bwibbu_payload(), {"2330", "2317"})
        self.assertEqual(out["2330"]["name"], "台積電")
        self.assertEqual(out["2317"]["name"], "鴻海")

    def test_bwibbu_all_name_does_not_break_existing_fields(self):
        out = _parse_bwibbu_all(self._bwibbu_payload(), {"2330"})
        self.assertEqual(out["2330"]["pe"], 20.5)
        self.assertEqual(out["2330"]["pb"], 5.2)
        self.assertEqual(out["2330"]["dividend_yield"], 1.8)
        self.assertIn("name", out["2330"])

    def test_t86_all_includes_name(self):
        out = _parse_t86_all(self._t86_payload(), {"2330", "2317"})
        self.assertEqual(out["2330"]["name"], "台積電")
        self.assertEqual(out["2317"]["name"], "鴻海")

    def test_t86_all_name_does_not_break_existing_fields(self):
        out = _parse_t86_all(self._t86_payload(), {"2330"})
        self.assertEqual(out["2330"]["foreign_net_lots"], 5000.0)
        self.assertEqual(out["2330"]["inst_total_net_lots"], 4000.0)
        self.assertIn("name", out["2330"])

    def test_resolve_name_uses_first_live_source(self):
        from marketlib.stocks import resolve_name
        live1 = {"2330": {"name": "台積電-live", "pe": 20.0}}
        live2 = {"2330": {"name": "台積電-live2"}}
        # First source wins
        self.assertEqual(resolve_name("2330", live1, live2), "台積電-live")

    def test_resolve_name_falls_back_to_tw50_names(self):
        from marketlib.stocks import resolve_name
        empty_source = {}
        self.assertEqual(resolve_name("2330", empty_source), "台積電")

    def test_resolve_name_falls_back_to_code_when_unknown(self):
        from marketlib.stocks import resolve_name
        self.assertEqual(resolve_name("9999"), "9999")

    def test_resolve_name_skips_empty_name_in_source(self):
        from marketlib.stocks import resolve_name
        # Source has the code but name is empty string — should skip to TW50_NAMES
        source_with_empty = {"2330": {"name": ""}}
        self.assertEqual(resolve_name("2330", source_with_empty), "台積電")

    def test_resolve_name_no_sources(self):
        from marketlib.stocks import resolve_name
        # No sources at all — falls back to TW50_NAMES
        self.assertEqual(resolve_name("2330"), "台積電")
        # Unknown code with no sources → code itself
        self.assertEqual(resolve_name("8888"), "8888")


if __name__ == "__main__":
    unittest.main()
