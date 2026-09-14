"""train.py and marketlib.model implement the two-stage probability twice —
once in numpy for training and evaluation, once in dependency-free Python for
the daily report. A drift between them means the metrics we publish describe a
different model than the one making today's call, so pin them together.

The NaN-imputation bug this file now guards was exactly that: training replaced
NaN with the column median, inference did not, and the isotonic lookup turned
the resulting NaN score into the highest calibrated probability.
"""
from __future__ import annotations

import numpy as np
import pytest

import train as T
from marketlib import model as ml_model

FEATURES = ["a", "b", "c"]


def _artifact(coefs, intercept, xs, ys, prior=None, shrinkage=None):
    art = {
        "standardization": {"mean": {"a": 1.0, "b": -2.0, "c": 0.5},
                            "std": {"a": 2.0, "b": 0.5, "c": 1.5}},
        "coefficients": dict(zip(FEATURES, coefs)),
        "intercept": intercept,
        "calibration": {"method": "isotonic", "x": xs, "y": ys},
    }
    if prior is not None:
        art["prior"] = prior
    if shrinkage is not None:
        art["shrinkage"] = shrinkage
    return art


MAG = _artifact([0.4, -0.3, 0.2], 0.1, [-1.5, -0.4, 0.0, 0.7, 1.8],
                [0.15, 0.35, 0.48, 0.62, 0.85], prior=0.45, shrinkage=1.0)
DIR = _artifact([-0.2, 0.5, 0.1], -0.05, [-1.2, 0.0, 1.1],
                [0.25, 0.55, 0.8], prior=0.56, shrinkage=0.6)
MODEL = {"model_type": "two_stage_logistic", "feature_names": FEATURES,
         "imputation_median": {"a": 1.0, "b": -2.0, "c": 0.5},
         "stages": {"magnitude": MAG, "direction": DIR}}

ROWS = [
    {"a": 3.0, "b": -1.0, "c": 2.0},
    {"a": -4.0, "b": 5.0, "c": -3.0},
    {"a": 1.0, "b": -2.0, "c": 0.5},
    {"a": 99.0, "b": -99.0, "c": 50.0},      # far outside the isotonic table
    {"a": 0.0, "b": 0.0, "c": 0.0},
]


MEDIAN = {"a": 1.0, "b": -2.0, "c": 0.5}


def _train_side(rows):
    """The training-side computation, imputation included.

    Imputation has to happen here too, via training's own `_imp`, or the
    comparison is inference-against-inference and cannot detect the two sides
    drifting apart — which is exactly how the NaN bug survived the first pass.
    """
    X = np.array([[T._imp(float(r.get(n, float("nan")) if r.get(n) is not None
                                else float("nan")), MEDIAN[n])
                   for n in FEATURES] for r in rows], dtype=float)
    return T._three_class_probs(MAG, DIR, T._z_for(MAG, X, FEATURES),
                                T._z_for(DIR, X, FEATURES))


@pytest.mark.parametrize("row", ROWS)
def test_training_and_inference_agree_on_every_class(row):
    expected = _train_side([row])[0]
    got = ml_model.predict_three_class(MODEL, row)
    for i, label in enumerate(T.DIRECTION_LABELS):
        assert got[label] == pytest.approx(expected[i], abs=1e-12)


@pytest.mark.parametrize("row", [
    {},                                                    # every feature absent
    {"a": float("nan"), "b": None, "c": float("nan")},     # NaN and None mixed
    {"a": 3.0, "b": float("nan"), "c": None},              # partially present
])
def test_agreement_holds_for_missing_and_nan_features(row):
    """Compared against the TRAINING side, not against another inference call.

    The earlier version of this test compared inference to inference, so it
    would have passed even while training imputed NaN and inference did not —
    the defect it exists to catch.
    """
    expected = _train_side([row])[0]
    got = ml_model.predict_three_class(MODEL, row)
    for i, label in enumerate(T.DIRECTION_LABELS):
        assert got[label] == pytest.approx(expected[i], abs=1e-12)


def test_the_parity_harness_would_notice_an_imputation_divergence():
    """Guard on the guard: if inference stopped imputing NaN, this comparison
    must fail. Pin that by checking the un-imputed path gives a different
    answer, so a silently-passing test cannot go unnoticed."""
    nan_row = {"a": float("nan"), "b": float("nan"), "c": float("nan")}
    X_raw = np.array([[float("nan")] * len(FEATURES)])
    unimputed = T._three_class_probs(MAG, DIR, T._z_for(MAG, X_raw, FEATURES),
                                     T._z_for(DIR, X_raw, FEATURES))[0]
    imputed = _train_side([nan_row])[0]
    assert not np.allclose(np.nan_to_num(unimputed, nan=-1.0), imputed)


@pytest.mark.parametrize("row", ROWS[:3])
def test_parity_holds_on_the_uncalibrated_sigmoid_path(row):
    """Stages without an isotonic table fall back to a plain sigmoid on both
    sides; that branch needs pinning too."""
    plain_mag = {k: v for k, v in MAG.items() if k != "calibration"}
    plain_dir = {k: v for k, v in DIR.items() if k != "calibration"}
    model = dict(MODEL, stages={"magnitude": plain_mag, "direction": plain_dir})
    X = np.array([[row[n] for n in FEATURES]], dtype=float)
    expected = T._three_class_probs(plain_mag, plain_dir,
                                    T._z_for(plain_mag, X, FEATURES),
                                    T._z_for(plain_dir, X, FEATURES))[0]
    got = ml_model.predict_three_class(model, row)
    for i, label in enumerate(T.DIRECTION_LABELS):
        assert got[label] == pytest.approx(expected[i], abs=1e-12)


def test_shrinkage_is_applied_on_both_sides():
    """DIR carries λ=0.6; dropping it on either side would change the answer."""
    unshrunk = dict(MODEL, stages={"magnitude": MAG,
                                   "direction": {k: v for k, v in DIR.items()
                                                 if k != "shrinkage"}})
    assert (ml_model.predict_three_class(MODEL, ROWS[0])["偏多"]
            != pytest.approx(ml_model.predict_three_class(unshrunk, ROWS[0])["偏多"]))


def test_probabilities_stay_a_distribution_on_both_sides():
    for row in ROWS:
        got = ml_model.predict_three_class(MODEL, row)
        assert sum(got.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in got.values())
