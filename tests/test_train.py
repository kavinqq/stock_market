# tests/test_train.py  (run with: .venv/bin/python -m unittest -v tests.test_train)
import os, json, tempfile, unittest
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
        model = json.load(open(model_path)); metrics = json.load(open(metrics_path))
        self.assertEqual(model["artifact"], "production_refit")
        self.assertEqual(metrics["artifact"], "final_test_evaluation")
        # production 訓練期間涵蓋到最後;final-test 訓練期間不含 test 窗
        self.assertNotEqual(model["train_period"], metrics["trained_on"])
        self.assertIn("baselines", metrics)
        self.assertIn("significance", metrics)

    def test_learns_signal_beats_random(self):
        samples = _synth_samples()
        out = tempfile.mkdtemp()
        _, metrics_path = train.train_from_samples(
            samples, train.MINIMAL_FEATURES, "tw", out, test_frac=0.25, seed=0)
        metrics = json.load(open(metrics_path))
        self.assertGreater(metrics["metrics"]["hit_rate"], 0.55)

if __name__ == "__main__":
    unittest.main()
