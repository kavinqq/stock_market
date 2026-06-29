import unittest
from marketlib.features import build_dataset, build_feature_row, decision_time_for
from marketlib.features import FEATURE_NAMES_TW, FEATURE_NAMES_US


def _tw_rows():
    return [{"date": f"2026-06-{d:02d}", "close": 100 + (d % 7), "volume": 1e9 + d,
             "turnover": 5e11, "price_asof_taipei": f"2026-06-{d:02d}T14:30:00+08:00",
             "available_at_taipei": f"2026-06-{d:02d}T14:30:00+08:00"}
            for d in range(1, 28)]


def _us_rows():
    from marketlib.fetchers import us_price_asof
    return [{"date": f"2026-06-{d:02d}", "spy_close": 400 + (d % 5),
             "qqq_close": 350 + (d % 5), "iwm_close": 180 + (d % 5),
             "vix_close": 18.0 + (d % 3), "spy_volume": 1e8 + d,
             "price_asof_taipei": us_price_asof(f"2026-06-{d:02d}")}
            for d in range(1, 28)]


class TestParity(unittest.TestCase):
    def test_offline_equals_online_for_last_day(self):
        rows = _tw_rows()
        offline = build_dataset("tw", rows, [])[-1]["features"]
        dt = decision_time_for(rows[-1]["date"])
        online = build_feature_row("tw", rows, len(rows) - 1, [], dt)
        self.assertEqual(offline, online)  # 同一份程式 -> 完全一致

    def test_tw_feature_keys_match_names(self):
        rows = _tw_rows()
        dt = decision_time_for(rows[-1]["date"])
        feats = build_feature_row("tw", rows, len(rows) - 1, [], dt)
        self.assertEqual(set(feats.keys()), set(FEATURE_NAMES_TW))

    def test_us_feature_keys_match_names(self):
        rows = _us_rows()
        dt = decision_time_for(rows[-1]["date"])
        feats = build_feature_row("us", rows, len(rows) - 1, [], dt)
        self.assertEqual(set(feats.keys()), set(FEATURE_NAMES_US))


if __name__ == "__main__":
    unittest.main()
