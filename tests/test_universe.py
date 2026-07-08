import unittest
from marketlib.universe import TW50_CONSTITUENTS, TW50_ASOF


class TestUniverse(unittest.TestCase):
    def test_nonempty_codes(self):
        self.assertGreaterEqual(len(TW50_CONSTITUENTS), 40)
        self.assertTrue(all(isinstance(c, str) and c.isdigit() for c in TW50_CONSTITUENTS))
        self.assertIn("2330", TW50_CONSTITUENTS)
        self.assertEqual(len(TW50_CONSTITUENTS), len(set(TW50_CONSTITUENTS)))  # no dups

    def test_asof_present(self):
        self.assertRegex(TW50_ASOF, r"\d{4}-\d{2}-\d{2}")


if __name__ == "__main__":
    unittest.main()
