import unittest
from marketlib.fetchers import twse_asof, us_price_asof, finra_asof
from datetime import datetime
from zoneinfo import ZoneInfo

def _dt(s): return datetime.fromisoformat(s)

class TestAsof(unittest.TestCase):
    def test_twse_price_before_1830(self):
        self.assertLess(_dt(twse_asof("2026-06-26", "price")),
                        _dt("2026-06-26T18:30:00+08:00"))

    def test_us_price_next_morning_taipei(self):
        # 美股 2026-06-25 收盤 -> Taipei 2026-06-26 凌晨,故 > 2026-06-25 18:30
        self.assertGreater(_dt(us_price_asof("2026-06-25")),
                           _dt("2026-06-25T18:30:00+08:00"))

    def test_finra_lag_approx(self):
        # 2026-05 月資料 -> 約 2026-06-20 後才可用
        self.assertGreaterEqual(_dt(finra_asof("2026-05")),
                                _dt("2026-06-20T00:00:00+08:00"))

if __name__ == "__main__":
    unittest.main()
