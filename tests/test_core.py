import unittest
from marketlib.core import (
    MarketBar, clean_num, pct_change, simple_moving_average, compute_indicators,
)

class TestCore(unittest.TestCase):
    def test_clean_num_strips_symbols(self):
        self.assertEqual(clean_num("1,234.5"), 1234.5)
        self.assertIsNone(clean_num("--"))

    def test_pct_change(self):
        self.assertAlmostEqual(pct_change(110, 100), 0.1)
        self.assertIsNone(pct_change(110, 0))

    def test_sma(self):
        self.assertEqual(simple_moving_average([1, 2, 3, 4], 2), 3.5)
        self.assertIsNone(simple_moving_average([1], 2))

    def test_volume_ratio_excludes_today(self):
        # 21 根:前 20 根量=100,今日量=200 -> ratio=2.0(分母不含今日)
        bars = [MarketBar(date=f"2026-01-{i+1:02d}", close=10.0, volume=100.0) for i in range(20)]
        bars.append(MarketBar(date="2026-02-01", close=10.0, volume=200.0))
        ind = compute_indicators(bars, volume_attr="volume")
        self.assertAlmostEqual(ind.volume_ratio_20d, 2.0)

if __name__ == "__main__":
    unittest.main()
