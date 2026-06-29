import unittest
from marketlib.stats import mcnemar_test, block_bootstrap_diff

class TestStats(unittest.TestCase):
    def test_mcnemar_counts(self):
        model =    [True,  True,  False, True]
        baseline = [False, True,  False, False]
        out = mcnemar_test(model, baseline)
        self.assertEqual(out["b"], 2)   # model right, baseline wrong (idx 0,3)
        self.assertEqual(out["c"], 0)
        self.assertIn("p_value", out)

    def test_bootstrap_positive_when_model_better(self):
        model = [True] * 80 + [False] * 20
        baseline = [True] * 50 + [False] * 50
        out = block_bootstrap_diff(model, baseline, block=5, n_resamples=500, seed=1)
        self.assertGreater(out["mean_diff"], 0)
        self.assertGreater(out["ci_low"], 0)  # 顯著為正

    def test_bootstrap_deterministic(self):
        m = [True, False] * 50
        b = [False, True] * 50
        a = block_bootstrap_diff(m, b, seed=7, n_resamples=300)
        c = block_bootstrap_diff(m, b, seed=7, n_resamples=300)
        self.assertEqual(a, c)  # 同 seed 結果一致

if __name__ == "__main__":
    unittest.main()
