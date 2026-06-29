import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from marketlib.features import decision_time_for, is_available, usable_value

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

if __name__ == "__main__":
    unittest.main()
