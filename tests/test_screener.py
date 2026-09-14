import unittest

import pytest

from marketlib import screener
from marketlib.screener import cross_sectional_rank, build_screener, SCREENER_WEIGHTS


class TestCrossSectionalRank(unittest.TestCase):

    def test_rank_higher_better(self):
        r = cross_sectional_rank([10, 20, 30], higher_is_better=True)
        self.assertLess(r[0], r[2])
        self.assertAlmostEqual(r[2], 1.0)

    def test_rank_lower_better_and_none(self):
        r = cross_sectional_rank([10, 20, None, 30], higher_is_better=False)
        self.assertAlmostEqual(r[0], 1.0)  # lowest is best
        self.assertIsNone(r[2])

    def test_rank_single_value(self):
        r = cross_sectional_rank([42.0], higher_is_better=True)
        self.assertAlmostEqual(r[0], 0.5)

    def test_rank_single_non_none(self):
        r = cross_sectional_rank([None, 5.0, None], higher_is_better=True)
        self.assertAlmostEqual(r[1], 0.5)
        self.assertIsNone(r[0])
        self.assertIsNone(r[2])

    def test_rank_ties_share_average_percentile(self):
        # [10, 10, 20]: two values tied at worst; best=1.0; tied positions 0&1 share avg of (0.0+0.5)/2 = 0.25
        r = cross_sectional_rank([10, 10, 20], higher_is_better=True)
        self.assertAlmostEqual(r[2], 1.0)
        self.assertAlmostEqual(r[0], r[1])
        self.assertLess(r[0], r[2])

    def test_rank_all_none(self):
        r = cross_sectional_rank([None, None])
        self.assertIsNone(r[0])
        self.assertIsNone(r[1])

    def test_rank_lower_better_ordering(self):
        # lower_is_better: lowest gets 1.0, highest gets 0.0
        r = cross_sectional_rank([10, 20, 30], higher_is_better=False)
        self.assertAlmostEqual(r[0], 1.0)
        self.assertAlmostEqual(r[2], 0.0)
        self.assertLess(r[2], r[0])


class TestScreenerWeights(unittest.TestCase):

    def test_weights_keys(self):
        self.assertEqual(
            set(SCREENER_WEIGHTS.keys()),
            {"valuation", "revenue", "technical", "momentum", "chip"},
        )

    def test_weights_values(self):
        self.assertAlmostEqual(SCREENER_WEIGHTS["valuation"], 0.30)
        self.assertAlmostEqual(SCREENER_WEIGHTS["revenue"], 0.25)
        self.assertAlmostEqual(SCREENER_WEIGHTS["technical"], 0.20)
        self.assertAlmostEqual(SCREENER_WEIGHTS["momentum"], 0.15)
        self.assertAlmostEqual(SCREENER_WEIGHTS["chip"], 0.10)


class TestBuildScreener(unittest.TestCase):

    def test_build_screener_sorts_and_scores(self):
        recs = [
            {
                "code": "A",
                "pe": 10,
                "pb": 1,
                "dividend_yield": 3,
                "rev_yoy": 30,
                "ret20": 0.05,
                "ret60": 0.1,
                "chip_net_lots": 5000,
                "tech_raw": 1.0,
            },
            {
                "code": "B",
                "pe": 40,
                "pb": 8,
                "dividend_yield": 0.5,
                "rev_yoy": -5,
                "ret20": -0.03,
                "ret60": -0.05,
                "chip_net_lots": -2000,
                "tech_raw": 0.0,
            },
        ]
        out = build_screener(recs)
        self.assertEqual(out[0]["code"], "A")  # cheap+growing+strong ranks first
        self.assertEqual(out[0]["rank"], 1)
        self.assertIn("valuation", out[0]["scores"])
        self.assertEqual(out[0]["raw"]["pe"], 10)

    def test_build_screener_rank_field(self):
        recs = [
            {"code": "X", "pe": 5, "pb": None, "dividend_yield": None, "rev_yoy": 10,
             "ret20": 0.1, "ret60": None, "chip_net_lots": 100, "tech_raw": 0.5},
            {"code": "Y", "pe": 50, "pb": None, "dividend_yield": None, "rev_yoy": -10,
             "ret20": -0.1, "ret60": None, "chip_net_lots": -100, "tech_raw": 0.0},
        ]
        out = build_screener(recs)
        self.assertEqual(out[0]["rank"], 1)
        self.assertEqual(out[1]["rank"], 2)

    def test_missing_facet_renormalizes(self):
        recs = [
            {
                "code": "A",
                "pe": 10,
                "pb": None,
                "dividend_yield": None,
                "rev_yoy": None,
                "ret20": 0.05,
                "ret60": None,
                "chip_net_lots": None,
                "tech_raw": 1.0,
            },
            {
                "code": "B",
                "pe": 20,
                "pb": None,
                "dividend_yield": None,
                "rev_yoy": None,
                "ret20": 0.01,
                "ret60": None,
                "chip_net_lots": None,
                "tech_raw": 0.0,
            },
        ]
        out = build_screener(recs)  # only valuation/momentum/technical present -> renormalize
        self.assertIsNotNone(out[0]["composite"])

    def test_no_facets_gives_none_composite(self):
        recs = [
            {"code": "Z", "pe": None, "pb": None, "dividend_yield": None,
             "rev_yoy": None, "ret20": None, "ret60": None,
             "chip_net_lots": None, "tech_raw": None},
        ]
        out = build_screener(recs)
        self.assertIsNone(out[0]["composite"])

    def test_raw_fields_preserved(self):
        recs = [
            {"code": "A", "pe": 10, "pb": 1.5, "dividend_yield": 2.0, "rev_yoy": 20,
             "ret20": 0.03, "ret60": 0.06, "chip_net_lots": 300, "tech_raw": 0.8},
        ]
        out = build_screener(recs)
        raw = out[0]["raw"]
        self.assertEqual(raw["pe"], 10)
        self.assertEqual(raw["pb"], 1.5)
        self.assertEqual(raw["dividend_yield"], 2.0)
        self.assertEqual(raw["rev_yoy"], 20)
        self.assertEqual(raw["ret20"], 0.03)
        self.assertEqual(raw["chip_net_lots"], 300)
        self.assertEqual(raw["tech_raw"], 0.8)

    def test_none_composite_sorts_last(self):
        recs = [
            {"code": "A", "pe": 10, "pb": None, "dividend_yield": None, "rev_yoy": None,
             "ret20": None, "ret60": None, "chip_net_lots": None, "tech_raw": None},
            {"code": "B", "pe": None, "pb": None, "dividend_yield": None, "rev_yoy": None,
             "ret20": None, "ret60": None, "chip_net_lots": None, "tech_raw": None},
            {"code": "C", "pe": 5, "pb": None, "dividend_yield": None, "rev_yoy": 10,
             "ret20": 0.05, "ret60": None, "chip_net_lots": None, "tech_raw": None},
        ]
        out = build_screener(recs)
        # C has facets => has composite => should rank before A (only pe) and B (no facets)
        codes = [r["code"] for r in out]
        self.assertEqual(codes[-1], "B")  # B has no facets at all → composite=None → last


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Technical intersection — 0050 潛力股 (added 2026-09-14)
# ---------------------------------------------------------------------------

def _bull_metrics(**overrides):
    """A stock passing every condition; override single fields to break one."""
    m = {
        "close": 110.0, "ma5": 108.0, "ma20": 105.0, "ma60": 100.0,
        "rsi": 62.0,
        "kd": {"k": 65.0, "d": 58.0},
        "macd_daily": {"dif": 1.2, "dea": 0.9, "hist": 0.3, "cross": None},
        "macd_weekly": {"dif": 2.0, "dea": 1.5, "hist": 0.5, "cross": None},
        "boll": {"mid": 105.0, "upper": 115.0, "lower": 95.0, "pctb": 0.75},
        "cost_vwap60": 102.0,
        "volume_ratio_20d": 1.4,
    }
    m.update(overrides)
    return m


def test_intersection_passes_when_every_core_condition_holds():
    result = screener.technical_intersection(_bull_metrics())
    assert result["is_candidate"] is True
    assert result["core_failed"] == []
    assert result["core_passed"] == result["core_total"]
    # Reasons must be readable sentences, not field names.
    assert any("月線" in r for r in result["reasons"])
    assert any("MACD" in r for r in result["reasons"])
    assert any("KD" in r for r in result["reasons"])


def test_one_failed_core_condition_disqualifies_and_is_named():
    """Intersection means AND: a single miss drops the stock, and the report
    has to say which one so the reader can judge whether they care."""
    result = screener.technical_intersection(_bull_metrics(kd={"k": 40.0, "d": 55.0}))
    assert result["is_candidate"] is False
    assert any("KD" in r for r in result["core_failed"])
    assert all("KD" not in r for r in result["reasons"])


def test_weekly_macd_is_a_core_condition():
    result = screener.technical_intersection(
        _bull_metrics(macd_weekly={"dif": 1.0, "dea": 1.4, "hist": -0.4, "cross": None}))
    assert result["is_candidate"] is False
    assert any("週" in r for r in result["core_failed"])


def test_overheated_rsi_disqualifies_even_with_every_core_condition():
    """RSI ≥ 80 is a veto, not a core condition: everything else can look
    perfect precisely because the stock has already run."""
    result = screener.technical_intersection(_bull_metrics(rsi=85.0))
    assert result["is_candidate"] is False
    assert any("RSI" in r for r in result["vetoes"])


def test_bonus_conditions_do_not_gate_but_are_reported():
    result = screener.technical_intersection(_bull_metrics(volume_ratio_20d=0.6))
    assert result["is_candidate"] is True
    assert any("量" in r for r in result["bonus_failed"])
    assert result["bonus_passed"] < result["bonus_total"]


def test_uncomputable_condition_is_not_silently_treated_as_a_pass():
    """A stock without enough history must not sneak through on missing data."""
    result = screener.technical_intersection(_bull_metrics(macd_weekly=None, ma60=None))
    assert result["is_candidate"] is False
    assert result["insufficient_data"] is True
    assert any("資料不足" in r for r in result["core_failed"])


def test_score_orders_candidates_by_total_conditions_met():
    strong = screener.technical_intersection(_bull_metrics())
    weaker = screener.technical_intersection(
        _bull_metrics(volume_ratio_20d=0.6, boll={"mid": 105.0, "upper": 115.0,
                                                  "lower": 95.0, "pctb": 0.3}))
    assert strong["score"] > weaker["score"]
    assert 0.0 <= weaker["score"] <= 1.0


def test_intersection_reports_the_numbers_behind_the_verdict():
    """Passing stocks all satisfy the same core conditions by definition, so the
    values are the only thing that tells them apart."""
    snap = screener.technical_intersection(_bull_metrics())["metrics_snapshot"]
    assert snap["rsi"] == 62.0
    assert snap["kd_k"] == 65.0 and snap["kd_d"] == 58.0
    assert snap["macd_hist"] == 0.3
    assert snap["pctb"] == 0.75
    assert snap["volume_ratio_20d"] == 1.4
    assert snap["close_vs_ma20"] == pytest.approx(110.0 / 105.0 - 1)


def test_core_condition_texts_are_stable_and_shared():
    """The header states the core conditions once, so they must not vary by stock."""
    a = screener.core_condition_texts()
    b = screener.technical_intersection(_bull_metrics())
    assert a == b["reasons"][:len(a)]
    assert len(a) == b["core_total"]


def test_price_outside_upper_bollinger_band_is_flagged_not_rewarded():
    """%b > 1 means the close sits ABOVE the upper band. It still satisfies the
    "upper half" bonus, but it is a stretch signal, so it must be surfaced —
    otherwise the shortlist quietly favours the most extended names."""
    r = screener.technical_intersection(
        _bull_metrics(boll={"mid": 105.0, "upper": 115.0, "lower": 95.0, "pctb": 1.37}))
    assert r["is_candidate"] is True
    assert any("布林" in w and "外" in w for w in r["warnings"])


def test_overbought_kd_is_flagged_below_the_rsi_veto():
    r = screener.technical_intersection(_bull_metrics(kd={"k": 88.0, "d": 80.0}))
    assert r["is_candidate"] is True
    assert any("K" in w for w in r["warnings"])


def test_large_gap_above_the_monthly_line_is_flagged():
    r = screener.technical_intersection(_bull_metrics(close=130.0))
    assert any("乖離" in w for w in r["warnings"])


def test_a_clean_setup_carries_no_warnings():
    assert screener.technical_intersection(_bull_metrics())["warnings"] == []
