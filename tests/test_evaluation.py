import unittest

import pytest

from marketlib.evaluation import (
    DIRECTION_LABELS, direction_from_return, summarize_baselines,
    summarize_classification,
)

class TestEvaluation(unittest.TestCase):
    def test_direction_threshold(self):
        self.assertEqual(direction_from_return(0.01, 0.003), "偏多")
        self.assertEqual(direction_from_return(-0.01, 0.003), "偏空")
        self.assertEqual(direction_from_return(0.001, 0.003), "震盪")

    def test_macro_f1_present(self):
        records = [
            {"actual_direction": "偏多", "prediction": "偏多"},
            {"actual_direction": "偏空", "prediction": "偏多"},
        ]
        out = summarize_classification(records)
        self.assertIn("macro_f1", out)
        self.assertEqual(out["confusion_matrix"]["偏空"]["偏多"], 1)

    def test_labels(self):
        self.assertEqual(DIRECTION_LABELS, ["偏多", "震盪", "偏空"])

if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Degenerate-predictor guards (added 2026-09-14)
# ---------------------------------------------------------------------------

def test_macro_f1_counts_never_predicted_class_as_zero():
    """A class that really occurs but is NEVER predicted must score f1=0, not be
    dropped from the macro average.

    Regression guard: dropping it let a degenerate always-震盪 predictor score a
    higher macro_f1 (0.58) than a genuine three-class model (0.41), so the deploy
    gate rewarded collapse.
    """
    records = [{"prediction": "震盪", "actual_direction": a}
               for a in ["震盪"] * 5 + ["偏多"] * 3 + ["偏空"] * 2]
    summary = summarize_classification(records)
    per_class = summary["per_class"]

    assert per_class["偏多"]["f1"] == 0.0
    assert per_class["偏空"]["f1"] == 0.0
    # macro-F1 must average over all three classes, so it cannot exceed 1/3 here.
    assert summary["macro_f1"] == pytest.approx((per_class["震盪"]["f1"] + 0.0 + 0.0) / 3)
    assert summary["macro_f1"] < 0.34


def test_macro_f1_degenerate_scores_below_balanced_model():
    """The whole point of the fix: collapse must not out-score a real model."""
    actuals = ["震盪"] * 5 + ["偏多"] * 3 + ["偏空"] * 2
    degenerate = [{"prediction": "震盪", "actual_direction": a} for a in actuals]
    balanced = [{"prediction": p, "actual_direction": a} for p, a in
                zip(["震盪", "震盪", "震盪", "偏多", "偏空",
                     "偏多", "偏多", "震盪", "偏空", "偏多"], actuals)]
    assert (summarize_classification(balanced)["macro_f1"]
            > summarize_classification(degenerate)["macro_f1"])


def test_summarize_baselines_includes_always_neutral():
    """震盪 is the plurality class in TW; always-震盪 is the baseline any model
    must actually beat, so it has to be reported alongside random_prior."""
    records = [{"prediction": "偏多", "actual_direction": a}
               for a in ["震盪"] * 6 + ["偏多"] * 3 + ["偏空"]]
    baselines = summarize_baselines(records)
    assert baselines["always_neutral"]["hit_rate"] == pytest.approx(0.6)
    assert baselines["always_bull"]["hit_rate"] == pytest.approx(0.3)
    assert baselines["always_neutral"]["sample_size"] == 10
