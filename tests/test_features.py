import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from marketlib.features import decision_time_for, is_available, usable_value, zscore_excluding_t, realized_vol, volume_ratio_excluding_t
from marketlib.features import build_dataset, latest_usable_us, build_feature_row

TZ = ZoneInfo("Asia/Taipei")

class TestGating(unittest.TestCase):
    def test_decision_time_is_1830(self):
        dt = decision_time_for("2026-06-26")
        self.assertEqual((dt.hour, dt.minute), (18, 30))
        # decision_time must be anchored to Asia/Taipei (+08:00), not naive/UTC.
        from datetime import timedelta
        self.assertEqual(dt.utcoffset(), timedelta(hours=8))
        self.assertEqual(dt.tzinfo.key, "Asia/Taipei")

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

def _tw_row(d, close, vol=1e9, turnover=5e11):
    return {"date": d, "close": close, "volume": vol, "turnover": turnover,
            "price_asof_taipei": f"{d}T14:30:00+08:00",
            "available_at_taipei": f"{d}T14:30:00+08:00"}

def _us_row(d, spy):
    from marketlib.fetchers import us_price_asof
    return {"date": d, "spy_close": spy, "qqq_close": spy, "spy_volume": 1e8,
            "vix_close": 18.0,
            "price_asof_taipei": us_price_asof(d)}

class TestDataset(unittest.TestCase):
    def test_us_lag_one_session(self):
        # 台股 2026-06-26 預測,US 2026-06-26 收盤次日凌晨才可用 -> 只能用 06-25
        dt = decision_time_for("2026-06-26")
        us = [_us_row("2026-06-25", 100.0), _us_row("2026-06-26", 200.0)]
        latest = latest_usable_us(us, dt)
        self.assertEqual(latest["date"], "2026-06-25")

    def test_label_is_next_day_up(self):
        rows = [_tw_row(f"2026-06-{d:02d}", close) for d, close in
                [(1, 100), (2, 101), (3, 100)]]
        ds = build_dataset("tw", rows, [])
        # 第 0 筆:隔日 101>100 -> label 1;最後一筆 label None
        self.assertEqual(ds[0]["label"], 1)
        self.assertIsNone(ds[-1]["label"])

    def test_feature_row_has_named_keys(self):
        rows = [_tw_row(f"2026-06-{d:02d}", 100 + d) for d in range(1, 30)]
        dt = decision_time_for(rows[-1]["date"])
        feats = build_feature_row("tw", rows, len(rows) - 1, [], dt)
        self.assertIn("ret_1d", feats)
        self.assertIn("close_vs_ma20", feats)

if __name__ == "__main__":
    unittest.main()
