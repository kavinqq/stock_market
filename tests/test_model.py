import json, math, os, tempfile, unittest
from marketlib.model import load_model, predict_proba, proba_to_direction, proba_to_confidence

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

if __name__ == "__main__":
    unittest.main()
