import unittest
from marketlib.stocks import _parse_stock_day, _parse_t86, _parse_bwibbu, _parse_monthly_revenue


class TestStockParsers(unittest.TestCase):
    def test_stock_day(self):
        payload = {
            "stat": "OK",
            "fields": ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"],
            "data": [["115/07/01", "30,000,000", "3.0E10", "1000", "1010", "995", "1005", "+5", "50000"]],
        }
        bars = _parse_stock_day(payload)
        self.assertEqual(bars[0].date, "2026-07-01")
        self.assertEqual(bars[0].close, 1005.0)
        self.assertEqual(bars[0].volume, 30000000.0)

    def test_stock_day_bad_stat_returns_empty(self):
        payload = {"stat": "FAIL", "fields": [], "data": []}
        bars = _parse_stock_day(payload)
        self.assertEqual(bars, [])

    def test_t86_maps_by_header(self):
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外陸資買賣超股數(不含外資自營商)",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [["2330", "台積電", "5000000", "1000000", "-2000000", "4000000"]],
        }
        out = _parse_t86(payload, "2330")
        self.assertEqual(out["foreign_net_lots"], 5000.0)   # 5,000,000 / 1000
        self.assertEqual(out["trust_net_lots"], 1000.0)
        self.assertEqual(out["dealer_net_lots"], -2000.0)
        self.assertEqual(out["inst_total_net_lots"], 4000.0)
        self.assertIsNone(_parse_t86(payload, "9999"))

    def test_t86_fallback_foreign_field(self):
        # Test with alternative foreign field name
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "證券名稱",
                "外資買賣超股數",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [["0050", "元大台灣50", "2000000", "500000", "300000", "2800000"]],
        }
        out = _parse_t86(payload, "0050")
        self.assertEqual(out["foreign_net_lots"], 2000.0)
        self.assertEqual(out["inst_total_net_lots"], 2800.0)

    def test_t86_bad_stat_returns_none(self):
        payload = {"stat": "FAIL", "fields": [], "data": []}
        self.assertIsNone(_parse_t86(payload, "2330"))

    def test_bwibbu(self):
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [["2330", "台積電", "1.80", "2025", "20.5", "5.2", "2025/1"]],
        }
        out = _parse_bwibbu(payload, "2330")
        self.assertEqual(out["pe"], 20.5)
        self.assertEqual(out["pb"], 5.2)
        self.assertEqual(out["dividend_yield"], 1.80)

    def test_bwibbu_stock_not_found_returns_none(self):
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
            "data": [["2330", "台積電", "1.80", "2025", "20.5", "5.2", "2025/1"]],
        }
        self.assertIsNone(_parse_bwibbu(payload, "9999"))

    def test_bwibbu_bad_stat_returns_none(self):
        payload = {"stat": "FAIL", "fields": [], "data": []}
        self.assertIsNone(_parse_bwibbu(payload, "2330"))

    def test_monthly_revenue(self):
        items = [
            {
                "公司代號": "2330",
                "資料年月": "11506",
                "營業收入-當月營收": "250000000",
                "去年同月增減(%)": "30.5",
                "上月比較增減(%)": "2.1",
                "前期比較增減(%)": "28.0",
            }
        ]
        out = _parse_monthly_revenue(items, "2330")
        self.assertEqual(out["month"], "2026-06")
        self.assertEqual(out["revenue_thousand"], 250000000.0)
        self.assertEqual(out["yoy_pct"], 30.5)
        self.assertEqual(out["mom_pct"], 2.1)
        self.assertEqual(out["cum_yoy_pct"], 28.0)

    def test_monthly_revenue_stock_not_found_returns_none(self):
        items = [{"公司代號": "2330", "資料年月": "11506", "營業收入-當月營收": "250000000",
                  "去年同月增減(%)": "30.5", "上月比較增減(%)": "2.1", "前期比較增減(%)": "28.0"}]
        self.assertIsNone(_parse_monthly_revenue(items, "9999"))

    def test_monthly_revenue_roc_date_conversion(self):
        # ROC 11506 -> year=115+1911=2026, month=06 -> "2026-06"
        items = [{"公司代號": "0050", "資料年月": "11412", "營業收入-當月營收": "1000",
                  "去年同月增減(%)": "5.0", "上月比較增減(%)": "1.0", "前期比較增減(%)": "4.0"}]
        out = _parse_monthly_revenue(items, "0050")
        self.assertEqual(out["month"], "2025-12")


if __name__ == "__main__":
    unittest.main()
