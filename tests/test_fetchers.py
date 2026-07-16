import unittest
from marketlib.fetchers import (
    twse_asof, us_price_asof, finra_asof, taifex_asof, parse_taifex_inst_futures,
)
from datetime import datetime
from zoneinfo import ZoneInfo

def _dt(s): return datetime.fromisoformat(s)

# 三大法人-區分各期貨契約 CSV(已由 big5 轉 utf-8),取自 futContractsDateDown 實際格式。
# 多空未平倉口數淨額 為欄位索引 13。臺股期貨:自營商 2352、投信 73873、外資及陸資 -83390。
_TAIFEX_CSV = (
    "日期,商品名稱,身份別,多方交易口數,多方交易契約金額(千元),空方交易口數,"
    "空方交易契約金額(千元),多空交易口數淨額,多空交易契約金額淨額(千元),"
    "多方未平倉口數,多方未平倉契約金額(千元),空方未平倉口數,空方未平倉契約金額(千元),"
    "多空未平倉口數淨額,多空未平倉契約金額淨額(千元)\n"
    "2026/07/14,臺股期貨,自營商,18415,165031400,17580,158013049,835,7018351,"
    "7573,68237918,5221,47125630,2352,21112288\n"
    "2026/07/14,臺股期貨,投信,30931,277031117,28852,256845189,2079,20185928,"
    "79928,721505121,6055,54537257,73873,666967864\n"
    "2026/07/14,臺股期貨,外資及陸資,130107,1166274183,132326,1187371511,-2219,-21097328,"
    "10732,96522287,94122,849133079,-83390,-752610792\n"
    "2026/07/14,電子期貨,外資及陸資,100,200,100,200,0,0,50,100,50,100,999,999\n"
)


class TestTaifexAsof(unittest.TestCase):
    def test_available_before_decision_1830(self):
        # TAIFEX 三大法人約 15:00 收盤後公布,決策時點 18:30 前即可用,無洩漏。
        self.assertLess(_dt(taifex_asof("2026-07-14")),
                        _dt("2026-07-14T18:30:00+08:00"))

    def test_asof_is_taipei_same_day(self):
        asof = _dt(taifex_asof("2026-07-14"))
        self.assertEqual(asof.date().isoformat(), "2026-07-14")
        self.assertEqual(asof.utcoffset().total_seconds(), 8 * 3600)


class TestTaifexParse(unittest.TestCase):
    def test_foreign_net_oi_lots(self):
        parsed = parse_taifex_inst_futures(_TAIFEX_CSV)
        self.assertEqual(parsed["2026-07-14"]["tx_foreign_oi_net_lots"], -83390.0)

    def test_inst_total_net_oi_is_sum_of_three(self):
        parsed = parse_taifex_inst_futures(_TAIFEX_CSV)
        # 自營商 2352 + 投信 73873 + 外資及陸資 -83390 = -7165
        self.assertEqual(parsed["2026-07-14"]["tx_inst_oi_net_lots"], -7165.0)

    def test_ignores_non_tx_products(self):
        # 電子期貨 那列(淨額 999)不得混入臺股期貨統計。
        parsed = parse_taifex_inst_futures(_TAIFEX_CSV)
        self.assertNotIn(999.0, parsed["2026-07-14"].values())

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
