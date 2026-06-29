import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from marketlib.features import decision_time_for, is_available, usable_value, zscore_excluding_t, realized_vol, volume_ratio_excluding_t

TZ = ZoneInfo("Asia/Taipei")

class TestGating(unittest.TestCase):
    def test_decision_time_is_1830(self):
        dt = decision_time_for("2026-06-26")
        self.assertEqual((dt.hour, dt.minute), (18, 30))

    def test_is_available(self):
        dt = decision_time_for("2026-06-26")
        self.assertTrue(is_available("2026-06-26T16:00:00+08:00", dt))
        self.assertFalse(is_available("2026-06-26T19:00:00+08:00", dt))  # 晚於 cron
        self.assertFalse(is_available(None, dt))

    def test_late_inst_does_not_drop_price(self):
        # Guardrail #1:法人晚發布(19:00),但價量(14:30)仍應可用
        dt = decision_time_for("2026-06-26")
        row = {
            "close": 100.0, "price_asof_taipei": "2026-06-26T14:30:00+08:00",
            "inst_total_net_ntd": 9e8, "inst_asof_taipei": "2026-06-26T19:00:00+08:00",
        }
        self.assertEqual(usable_value(row, "close", "price_asof_taipei", dt), 100.0)
        self.assertIsNone(usable_value(row, "inst_total_net_ntd", "inst_asof_taipei", dt))

class TestWindowPolicy(unittest.TestCase):
    def test_zscore_excludes_t(self):
        # 基準前 4 個都=10(std=0 -> 回 None 以免除以零);改成有變異
        series = [10, 12, 8, 10, 20]  # 基準=[10,12,8,10] mean=10 std≈1.414
        z = zscore_excluding_t(series, 4)
        self.assertAlmostEqual(z, (20 - 10) / pstdev_helper([10, 12, 8, 10]), places=4)

    def test_realized_vol_includes_t(self):
        rets = [0.01, -0.01, 0.02, -0.02]
        self.assertAlmostEqual(realized_vol(rets, 4), pstdev_helper(rets), places=6)

    def test_volume_ratio_excludes_t(self):
        vols = [100, 100, 100, 200]  # 前 3 均=100,今日=200
        self.assertAlmostEqual(volume_ratio_excluding_t(vols, 3), 2.0)

def pstdev_helper(xs):
    from statistics import pstdev
    return pstdev(xs)

if __name__ == "__main__":
    unittest.main()
