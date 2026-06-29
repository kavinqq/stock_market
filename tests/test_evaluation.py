import unittest
from marketlib.evaluation import (
    DIRECTION_LABELS, direction_from_return, summarize_classification,
)

class TestEvaluation(unittest.TestCase):
    def test_direction_threshold(self):
        self.assertEqual(direction_from_return(0.01, 0.003), "偏多")
        self.assertEqual(direction_from_return(-0.01, 0.003), "偏空")
        self.assertEqual(direction_from_return(0.001, 0.003), "震盪")

    def test_macro_f1_present(self):
        records = [
            {"actual_direction": "偏多", "prediction": "偏多"},
            {"actual_direction": "偏空", "prediction": "偏多"},
        ]
        out = summarize_classification(records)
        self.assertIn("macro_f1", out)
        self.assertEqual(out["confusion_matrix"]["偏空"]["偏多"], 1)

    def test_labels(self):
        self.assertEqual(DIRECTION_LABELS, ["偏多", "震盪", "偏空"])

if __name__ == "__main__":
    unittest.main()
