import unittest

import pandas as pd

from anomalytics.stats import get_pot_threshold


class TestPeaksOverThreshold(unittest.TestCase):
    def setUp(self):
        self.ts = pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range(start="2023-01-01", periods=10))

    def test_high_anomaly_type(self):
        pot_threshold = get_pot_threshold(self.ts, t0=3, anomaly_type="high", q=0.90)
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.ts))

    def test_low_anomaly_type(self):
        pot_threshold = get_pot_threshold(self.ts, t0=3, anomaly_type="low", q=0.10)
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.ts))

    def test_invalid_anomaly_type(self):
        with self.assertRaises(ValueError):
            get_pot_threshold(self.ts, t0=3, anomaly_type="invalid", q=0.90)

    def test_invalid_ts_type(self):
        with self.assertRaises(TypeError):
            get_pot_threshold([1, 2, 3, 4], t0=3, anomaly_type="high", q=0.90)

    def test_invalid_t0_value(self):
        with self.assertRaises(ValueError):
            get_pot_threshold(self.ts, t0=None, anomaly_type="high", q=0.90)

    def tearDown(self) -> None:
        return super().tearDown()
