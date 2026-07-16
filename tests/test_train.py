# tests/test_train.py  (run with: .venv/bin/python -m unittest -v tests.test_train)
import os, json, tempfile, unittest
import numpy as np
import train

def _synth_samples(n=400):
    # 造一個「ret_1d 正相關於隔日上漲」的可學訊號
    import random
    rng = random.Random(0)
    samples = []
    for i in range(n):
        sig = rng.gauss(0, 1)
        feats = {name: rng.gauss(0, 1) for name in train.MINIMAL_FEATURES}
        feats["ret_1d"] = sig
        label = 1 if (sig + rng.gauss(0, 0.5)) > 0 else 0
        samples.append({"date": f"2020-{(i%12)+1:02d}-{(i%28)+1:02d}",
                        "features": feats, "label": label,
                        "next_return": 0.01 if label else -0.01,
                        "actual_direction": "偏多" if label else "偏空",
                        "neutral_threshold_pct": 0.003})
    return samples

class TestTrain(unittest.TestCase):
    def test_artifacts_have_distinct_periods_and_metadata(self):
        samples = _synth_samples()
        out = tempfile.mkdtemp()
        model_path, metrics_path = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25, seed=0)
        with open(model_path) as fh:
            model = json.load(fh)
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        self.assertEqual(model["artifact"], "production_refit")
        self.assertEqual(metrics["artifact"], "final_test_evaluation")
        # production 訓練期間涵蓋到最後;final-test 訓練期間不含 test 窗
        self.assertNotEqual(model["train_period"], metrics["trained_on"])
        self.assertIn("baselines", metrics)
        self.assertIn("significance", metrics)

    def test_matrix_imputes_all_missing_feature_to_zero(self):
        # A feature that is None for EVERY sample in a slice (e.g. the US
        # features in a pre-2022 TW walk-forward fold) must not leak NaN into
        # the design matrix — np.nanmedian of an all-NaN column is NaN, so the
        # imputation value must fall back to a neutral 0.0.
        samples = [{"features": {"ret_1d": 0.1, "us_vix_level": None}, "label": 1},
                   {"features": {"ret_1d": -0.2, "us_vix_level": None}, "label": 0}]
        X, y, med = train._matrix(samples, ["ret_1d", "us_vix_level"])
        self.assertFalse(np.isnan(X).any())
        self.assertEqual(med["us_vix_level"], 0.0)

    def test_learns_signal_beats_random(self):
        samples = _synth_samples()
        out = tempfile.mkdtemp()
        _, metrics_path = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25, seed=0)
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        self.assertGreater(metrics["metrics"]["hit_rate"], 0.55)

    def test_embargo_gap(self):
        # Fold boundary arithmetic: the train slice excludes the EMBARGO samples
        # adjacent to the val fold. With n samples and n_folds folds, fold size is
        # n // (n_folds + 1); for fold k the train slice ends at fold*k - EMBARGO
        # and the val fold begins at fold*k -> a gap of exactly EMBARGO samples.
        n, n_folds = 400, 4
        fold = n // (n_folds + 1)
        for k in range(1, n_folds + 1):
            tr_end = max(0, fold * k - train.EMBARGO)
            va_start = fold * k
            self.assertEqual(va_start - tr_end, train.EMBARGO)

        # train_from_samples leaves an EMBARGO index gap between trainval[-1] and
        # test[0]. We verify via dates tagged with their original index.
        samples = _synth_samples()
        for idx, s in enumerate(samples):
            s["date"] = f"idx-{idx:04d}"
        out = tempfile.mkdtemp()
        model_path, metrics_path = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25, seed=0)
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        trainval_last_idx = int(metrics["trained_on"][1].split("-")[1])
        test_first_idx = int(metrics["tested_on"][0].split("-")[1])
        # cut = floor(400 * 0.75) = 300; trainval = labeled[:cut-EMBARGO] ends at
        # index 294, test = labeled[cut:] starts at 300 -> exactly EMBARGO samples
        # (indices 295..299) are purged in the gap between them.
        dropped = test_first_idx - trainval_last_idx - 1
        self.assertEqual(dropped, train.EMBARGO)
        self.assertGreater(test_first_idx, trainval_last_idx)

    def test_confidence_bands_data_derived(self):
        samples = _synth_samples()
        out = tempfile.mkdtemp()
        model_path, _ = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25, seed=0)
        with open(model_path) as fh:
            model = json.load(fh)
        bands = model["confidence_bands"]
        # NOT the old hardcoded constant.
        self.assertNotEqual((bands["mid"], bands["high"]), (0.04, 0.10))
        # Valid tertiles of |p - 0.5|.
        self.assertLessEqual(0.0, bands["mid"])
        self.assertLessEqual(bands["mid"], bands["high"])
        self.assertLessEqual(bands["high"], 0.5)

    def test_platt_calibration_fit_and_store(self):
        # Focused unit test of the Platt fit + store path (selection of balanced
        # params is data-dependent, so test the calibrator directly).
        rng = np.random.RandomState(0)
        n = 500
        z = rng.normal(0, 2, n)
        # y strongly tied to z so a logistic of y~z is well-defined.
        prob = 1.0 / (1.0 + np.exp(-(0.8 * z + 0.3)))
        y = (rng.uniform(size=n) < prob).astype(int)
        w, b = train._platt_calibrate(z, y)
        self.assertIsInstance(w, float)
        self.assertIsInstance(b, float)
        cal = {"method": "platt", "params": {"w": w, "b": b}}
        self.assertEqual(cal["method"], "platt")
        self.assertIn("w", cal["params"])
        self.assertIn("b", cal["params"])
        # _apply_platt matches the stored-params formula.
        p_cal = train._apply_platt(z[:5], w, b)
        expected = 1.0 / (1.0 + np.exp(-(w * z[:5] + b)))
        self.assertTrue(np.allclose(p_cal, expected))
        # Recovered slope should be positive (z positively predicts y).
        self.assertGreater(w, 0.0)

    def test_should_calibrate_respects_use_platt_flag(self):
        y = np.array([0, 1, 0, 1])  # 兩類齊全
        # balanced + use_platt=True -> 校準;use_platt=False(台股)-> 不校準
        self.assertTrue(train._should_calibrate({"class_weight": "balanced"}, y, True))
        self.assertFalse(train._should_calibrate({"class_weight": "balanced"}, y, False))
        # 非 balanced -> 本來就不校準
        self.assertFalse(train._should_calibrate({"class_weight": None}, y, True))
        # 單一類別 -> 不校準(Platt 無意義)
        self.assertFalse(train._should_calibrate({"class_weight": "balanced"},
                                                 np.array([1, 1, 1]), True))

    def test_tw_training_skips_platt(self):
        # 台股整條訓練流程應產出未校準模型(calibration.method == "none"),
        # 讓原始 sigmoid 機率保有分散度以通過部署 gate。
        samples = _synth_samples()
        out = tempfile.mkdtemp()
        model_path, _ = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25,
            seed=0, use_platt=False)
        with open(model_path) as fh:
            model = json.load(fh)
        self.assertEqual(model["calibration"]["method"], "none")

    def test_collect_oos_predictions_pooled(self):
        samples = _synth_samples()
        params = {"C": 1.0, "class_weight": None}
        z, p, y = train.collect_oos_predictions(
            samples, train.MINIMAL_FEATURES, params)
        self.assertEqual(len(z), len(p))
        self.assertEqual(len(p), len(y))
        self.assertGreater(len(y), 0)
        # p must be the sigmoid of z.
        self.assertTrue(np.allclose(p, 1.0 / (1.0 + np.exp(-z)), atol=1e-6))

if __name__ == "__main__":
    unittest.main()
