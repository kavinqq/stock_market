import unittest
from marketlib.universe import TW50_CONSTITUENTS, TW50_ASOF, TW50_NAMES


class TestUniverse(unittest.TestCase):
    def test_nonempty_codes(self):
        self.assertGreaterEqual(len(TW50_CONSTITUENTS), 40)
        self.assertTrue(all(isinstance(c, str) and c.isdigit() for c in TW50_CONSTITUENTS))
        self.assertIn("2330", TW50_CONSTITUENTS)
        self.assertEqual(len(TW50_CONSTITUENTS), len(set(TW50_CONSTITUENTS)))  # no dups

    def test_asof_present(self):
        self.assertRegex(TW50_ASOF, r"\d{4}-\d{2}-\d{2}")


class TestTW50Names(unittest.TestCase):
    """Task 2: TW50_NAMES static fallback map."""

    def test_tsmc_name(self):
        self.assertEqual(TW50_NAMES["2330"], "台積電")

    def test_keys_subset_of_constituents(self):
        constituents_set = set(TW50_CONSTITUENTS)
        for code in TW50_NAMES:
            self.assertIn(code, constituents_set, f"{code} in TW50_NAMES but not in TW50_CONSTITUENTS")

    def test_all_values_nonempty_strings(self):
        for code, name in TW50_NAMES.items():
            self.assertIsInstance(name, str, f"TW50_NAMES[{code!r}] is not a str")
            self.assertTrue(name.strip(), f"TW50_NAMES[{code!r}] is blank")

    def test_known_names(self):
        expected = {
            "2317": "鴻海",
            "2454": "聯發科",
            "2412": "中華電",
            "2881": "富邦金",
        }
        for code, name in expected.items():
            self.assertEqual(TW50_NAMES.get(code), name, f"Expected TW50_NAMES[{code!r}]=={name!r}")


if __name__ == "__main__":
    unittest.main()
