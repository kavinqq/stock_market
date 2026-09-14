import json, math, os, tempfile, unittest

import pytest

from marketlib import model as ml_model
from marketlib.model import (
    load_model, predict_proba, proba_to_direction, proba_to_confidence,
    passes_deploy_gate,
)


def _metrics(hit_rate, random_prior, predicted_counts):
    """Build a metrics dict shaped like models/.../metrics_<mkt>.json."""
    return {
        "metrics": {
            "hit_rate": hit_rate,
            "classification": {
                "per_class": {
                    cls: {"predicted_count": n} for cls, n in predicted_counts.items()
                }
            },
        },
        "baselines": {"random_prior": {"expected_hit_rate": random_prior}},
    }

MODEL = {
    "market": "tw", "model_type": "logistic_regression",
    "feature_names": ["a", "b"],
    "standardization": {"mean": {"a": 0.0, "b": 0.0}, "std": {"a": 1.0, "b": 1.0}},
    "imputation_median": {"a": 0.0, "b": 0.0},
    "coefficients": {"a": 2.0, "b": 0.0}, "intercept": 0.0,
    "thresholds": {"bull": 0.6, "bear": 0.4},
    "confidence_bands": {"mid": 0.05, "high": 0.2},
}

class TestModel(unittest.TestCase):
    def test_predict_proba_logistic(self):
        p = predict_proba(MODEL, {"a": 1.0, "b": 0.0})  # z = 2 -> sigmoid(2)
        self.assertAlmostEqual(p, 1 / (1 + pow(2.718281828, -2.0)), places=4)

    def test_missing_feature_uses_median(self):
        p = predict_proba(MODEL, {"b": 0.0})  # a 缺 -> median 0 -> z=0 -> 0.5
        self.assertAlmostEqual(p, 0.5, places=6)

    def test_direction_and_confidence(self):
        self.assertEqual(proba_to_direction(MODEL, 0.7), "偏多")
        self.assertEqual(proba_to_direction(MODEL, 0.3), "偏空")
        self.assertEqual(proba_to_direction(MODEL, 0.5), "震盪")
        self.assertEqual(proba_to_confidence(MODEL, 0.95), "高")
        self.assertEqual(proba_to_confidence(MODEL, 0.5), "低")

    def test_load_model_roundtrip(self):
        d = tempfile.mkdtemp(); path = os.path.join(d, "m.json")
        with open(path, "w") as fh:
            json.dump(MODEL, fh)
        self.assertEqual(load_model(path)["market"], "tw")

    def test_calibration_applied(self):
        # With a Platt calibrator, predict_proba must return
        # 1/(1+exp(-(w*z+b))) and differ from the raw sigmoid(z).
        w, b = 2.0, 0.5
        cal_model = dict(MODEL, calibration={"method": "platt", "params": {"w": w, "b": b}})
        feats = {"a": 1.0, "b": 0.0}  # z = 2.0 (coef a=2.0)
        z = 2.0
        raw = 1.0 / (1.0 + math.exp(-z))
        expected = 1.0 / (1.0 + math.exp(-(w * z + b)))
        got = predict_proba(cal_model, feats)
        self.assertAlmostEqual(got, expected, places=10)
        self.assertNotAlmostEqual(got, raw, places=4)
        # method "none" must return the raw sigmoid (regression guard).
        none_model = dict(MODEL, calibration={"method": "none", "params": {}})
        self.assertAlmostEqual(predict_proba(none_model, feats), raw, places=10)
        # Missing calibration key also returns raw sigmoid.
        self.assertAlmostEqual(predict_proba(MODEL, feats), raw, places=10)

class TestDeployGate(unittest.TestCase):
    def test_passes_when_beats_random_and_non_degenerate(self):
        m = _metrics(0.363, 0.357, {"偏多": 156, "震盪": 67, "偏空": 55})
        ok, reason = passes_deploy_gate(m)
        self.assertTrue(ok)
        self.assertIn("0.36", reason)

    def test_blocks_degenerate_model(self):
        # US-style collapse: only 偏多 ever predicted.
        m = _metrics(0.90, 0.357, {"偏多": 274, "震盪": 0, "偏空": 0})
        ok, reason = passes_deploy_gate(m)
        self.assertFalse(ok)
        self.assertIn("震盪", reason)
        self.assertIn("偏空", reason)

    def test_blocks_when_below_random_prior(self):
        m = _metrics(0.30, 0.357, {"偏多": 100, "震盪": 100, "偏空": 78})
        ok, reason = passes_deploy_gate(m)
        self.assertFalse(ok)
        self.assertIn("隨機", reason)

    def test_passes_when_random_prior_missing(self):
        # No baseline to compare against -> only the non-degenerate check applies.
        m = _metrics(0.20, None, {"偏多": 100, "震盪": 100, "偏空": 78})
        ok, _ = passes_deploy_gate(m)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Two-stage three-class model (added 2026-09-14)
# ---------------------------------------------------------------------------

def _two_stage_model():
    """Minimal hand-built two-stage artifact.

    magnitude stage → P(震盪); direction stage → P(偏多 | 非震盪).
    Both use a single feature `x` standardized to itself, with an identity-ish
    isotonic table so the arithmetic is checkable by hand.
    """
    iso = {"method": "isotonic", "x": [-2.0, 0.0, 2.0], "y": [0.2, 0.5, 0.8]}
    stage = lambda coef, intercept: {
        "standardization": {"mean": {"x": 0.0}, "std": {"x": 1.0}},
        "coefficients": {"x": coef},
        "intercept": intercept,
        "calibration": iso,
    }
    return {
        "artifact": "production_refit",
        "market": "tw",
        "model_type": "two_stage_logistic",
        "feature_names": ["x"],
        "imputation_median": {"x": 0.0},
        "stages": {"magnitude": stage(1.0, 0.0), "direction": stage(1.0, 0.0)},
        "confidence_tiers": [
            {"min_prob": 0.60, "label": "高"},
            {"min_prob": 0.50, "label": "中"},
            {"min_prob": 0.00, "label": "低"},
        ],
    }


def test_three_class_probabilities_sum_to_one():
    m = _two_stage_model()
    for x in (-2.0, -0.5, 0.0, 1.0, 2.0):
        probs = ml_model.predict_three_class(m, {"x": x})
        assert set(probs) == {"偏多", "震盪", "偏空"}
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in probs.values())


def test_three_class_decomposition_matches_stage_outputs():
    """P(偏多) must equal (1 - P震盪) * P(偏多|非震盪) — the whole point of the
    two-stage split is that magnitude and direction stay separable."""
    m = _two_stage_model()
    probs = ml_model.predict_three_class(m, {"x": 2.0})
    # isotonic table maps z=2.0 → 0.8 for both stages
    assert probs["震盪"] == pytest.approx(0.8)
    assert probs["偏多"] == pytest.approx(0.2 * 0.8)
    assert probs["偏空"] == pytest.approx(0.2 * 0.2)


def test_isotonic_interpolates_between_breakpoints_and_clips_outside():
    m = _two_stage_model()
    assert ml_model.predict_three_class(m, {"x": 1.0})["震盪"] == pytest.approx(0.65)
    # beyond the table the value clips to the endpoint, never extrapolates
    assert ml_model.predict_three_class(m, {"x": 99.0})["震盪"] == pytest.approx(0.8)
    assert ml_model.predict_three_class(m, {"x": -99.0})["震盪"] == pytest.approx(0.2)


def test_missing_feature_falls_back_to_imputation_median():
    m = _two_stage_model()
    assert (ml_model.predict_three_class(m, {})
            == ml_model.predict_three_class(m, {"x": 0.0}))


def test_direction_is_argmax_of_three_class_probabilities():
    m = _two_stage_model()
    probs = {"偏多": 0.5, "震盪": 0.3, "偏空": 0.2}
    assert ml_model.direction_from_probs(probs) == "偏多"
    assert ml_model.direction_from_probs({"偏多": 0.1, "震盪": 0.7, "偏空": 0.2}) == "震盪"
    assert ml_model.direction_from_probs({"偏多": 0.1, "震盪": 0.2, "偏空": 0.7}) == "偏空"


def test_confidence_tier_uses_measured_bands():
    m = _two_stage_model()
    assert ml_model.confidence_from_probs(m, {"偏多": 0.7, "震盪": 0.2, "偏空": 0.1}) == "高"
    assert ml_model.confidence_from_probs(m, {"偏多": 0.55, "震盪": 0.3, "偏空": 0.15}) == "中"
    assert ml_model.confidence_from_probs(m, {"偏多": 0.4, "震盪": 0.35, "偏空": 0.25}) == "低"


# ---------------------------------------------------------------------------
# Deploy gate must not pass a model that merely tracks the class prior
# ---------------------------------------------------------------------------

def _gate_metrics(hit, per_class_predicted, baselines, brier=None, brier_base=None):
    return {
        "metrics": {
            "hit_rate": hit,
            "brier": brier,
            "brier_baseline": brier_base,
            "classification": {
                "per_class": {k: {"predicted_count": v}
                              for k, v in per_class_predicted.items()},
            },
        },
        "baselines": baselines,
    }


BASE = {
    "always_bull": {"hit_rate": 0.36},
    "always_neutral": {"hit_rate": 0.44},
    "always_bear": {"hit_rate": 0.22},
    "random_prior": {"expected_hit_rate": 0.35},
}


def test_gate_rejects_model_that_loses_to_always_neutral():
    """The deployed model scored 0.365 while always-震盪 scored 0.44. Clearing
    random_prior alone let that ship; the gate must now catch it."""
    ok, reason = ml_model.passes_deploy_gate(
        _gate_metrics(0.365, {"偏多": 286, "震盪": 32, "偏空": 60}, BASE,
                 brier=0.24, brier_base=0.24))
    assert ok is False
    assert "永遠震盪" in reason or "0.44" in reason


def test_gate_rejects_degenerate_collapse():
    ok, reason = ml_model.passes_deploy_gate(
        _gate_metrics(0.46, {"偏多": 0, "震盪": 378, "偏空": 0}, BASE,
                 brier=0.23, brier_base=0.246))
    assert ok is False
    assert "退化" in reason


def test_gate_rejects_uncalibrated_model_with_no_brier_skill():
    """AUC-only skill is not enough: if the probabilities are worse than a
    constant, the numbers we hand the human are not trustworthy."""
    ok, reason = ml_model.passes_deploy_gate(
        _gate_metrics(0.46, {"偏多": 130, "震盪": 200, "偏空": 48}, BASE,
                 brier=0.26, brier_base=0.246))
    assert ok is False
    assert "Brier" in reason


def test_gate_passes_calibrated_model_that_beats_every_constant():
    ok, reason = ml_model.passes_deploy_gate(
        _gate_metrics(0.459, {"偏多": 130, "震盪": 200, "偏空": 48}, BASE,
                 brier=0.2366, brier_base=0.2461))
    assert ok is True, reason


def test_stage_shrinkage_pulls_probability_toward_prior():
    """A stage whose ranking is informative but whose magnitudes overshoot is
    shrunk toward its prior: p = prior + λ(p_iso − prior).

    Measured on TW, the direction stage ranks well (AUC ≈ 0.59) but its raw
    isotonic output made the multiclass Brier worse than a constant, so the
    probabilities were not fit to publish without shrinkage.
    """
    m = _two_stage_model()
    m["stages"]["direction"]["shrinkage"] = 0.5
    m["stages"]["direction"]["prior"] = 0.5
    probs = ml_model.predict_three_class(m, {"x": 2.0})
    # direction: isotonic 0.8 → 0.5 + 0.5*(0.8-0.5) = 0.65; magnitude untouched
    assert probs["震盪"] == pytest.approx(0.8)
    assert probs["偏多"] == pytest.approx(0.2 * 0.65)
    assert probs["偏空"] == pytest.approx(0.2 * 0.35)


def test_zero_shrinkage_collapses_stage_to_its_prior():
    m = _two_stage_model()
    m["stages"]["direction"]["shrinkage"] = 0.0
    m["stages"]["direction"]["prior"] = 0.56
    probs = ml_model.predict_three_class(m, {"x": 2.0})
    assert probs["偏多"] == pytest.approx(0.2 * 0.56)


def test_absent_shrinkage_leaves_probability_untouched():
    m = _two_stage_model()
    assert "shrinkage" not in m["stages"]["direction"]
    assert ml_model.predict_three_class(m, {"x": 2.0})["偏多"] == pytest.approx(0.2 * 0.8)


def test_gate_rejects_when_rolling_validation_loses_to_constant():
    """One held-out window is a lucky-split away from any verdict. When the
    artifact carries a rolling-origin estimate, that is the one that governs:
    on TW the fixed split read 0.468 (p=0.008 vs always-震盪) while eight
    rolling windows read 0.441 (p=0.57) — the fixed split was period luck.
    """
    m = _gate_metrics(0.468, {"偏多": 65, "震盪": 297, "偏空": 16}, BASE,
                      brier=0.2366, brier_base=0.2461)
    m["rolling_validation"] = {"hit_rate": 0.402, "pooled_sample_size": 1008,
                               "baselines": {"always_neutral": 0.4325}}
    ok, reason = ml_model.passes_deploy_gate(m)
    assert ok is False
    assert "滾動" in reason


def test_gate_passes_when_rolling_validation_holds_up():
    m = _gate_metrics(0.468, {"偏多": 65, "震盪": 297, "偏空": 16}, BASE,
                      brier=0.2366, brier_base=0.2461)
    m["rolling_validation"] = {"hit_rate": 0.4405, "pooled_sample_size": 1008,
                               "baselines": {"always_neutral": 0.4325}}
    ok, reason = ml_model.passes_deploy_gate(m)
    assert ok is True, reason
