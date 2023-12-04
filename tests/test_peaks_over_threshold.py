import unittest

import numpy as np
import pandas as pd

from anomalytics import get_exceedance_peaks_over_threshold, fit_exceedance
from anomalytics.stats import get_threshold_peaks_over_threshold


class TestPeaksOverThreshold(unittest.TestCase):
    def setUp(self):
        self.sample_1_ts = pd.Series(
            data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range(start="2023-01-01", periods=10)
        )
        self.sample_2_ts = pd.Series(np.random.rand(100), index=pd.date_range(start="2023-01-01", periods=100))

    def test_calculate_threshold_for_high_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(ts=self.sample_1_ts, t0=3, anomaly_type="high", q=0.90)
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))

    def test__calculate_threshold_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(ts=self.sample_1_ts, t0=3, anomaly_type="low", q=0.10)
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))

    def test_invalid_anomaly_type_in_threshold_calculation_function(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(ts=self.sample_1_ts, t0=3, anomaly_type="invalid", q=0.90)  # type: ignore

    def test_invalid_ts_type_in_threshold_calculation_function(self):
        with self.assertRaises(TypeError):
            get_threshold_peaks_over_threshold(ts=[1, 2, 3, 4], t0=3, anomaly_type="high", q=0.90)

    def test_invalid_t0_value_in_threshold_calculation_function(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(ts=self.sample_1_ts, t0=None, anomaly_type="high", q=0.90)  # type: ignore

    def test_extract_exceedance_for_high_anomaly_type(self):
        pot_exceedance = get_exceedance_peaks_over_threshold(ts=self.sample_2_ts, t0=5, anomaly_type="high", q=0.90)
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.sample_2_ts))
        self.assertTrue((pot_exceedance >= 0).all())

    def test_extract_exceedance_for_low_anomaly_type(self):
        pot_exceedance = get_exceedance_peaks_over_threshold(ts=self.sample_2_ts, t0=5, anomaly_type="low", q=0.10)
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.sample_2_ts))
        # Check if all exceedances are non-negative
        self.assertTrue((pot_exceedance >= 0).all())

    def test_invalid_anomaly_type_in_exceedance_extraction_function(self):
        with self.assertRaises(ValueError):
            get_exceedance_peaks_over_threshold(ts=self.sample_2_ts, t0=5, anomaly_type="invalid", q=0.90)  # type: ignore

    def test_invalid_ts_type_in_exceedance_extraction_function(self):
        with self.assertRaises(TypeError):
            get_exceedance_peaks_over_threshold(ts="Not a series", t0=5, anomaly_type="high", q=0.90)  # type: ignore

    def test_invalid_t0_value_in_exceedance_extraction_function(self):
        with self.assertRaises(ValueError):
            get_exceedance_peaks_over_threshold(ts=self.sample_2_ts, t0=None, anomaly_type="high", q=0.90)  # type: ignore

    def test_fit_exceedance_with_valid_input(self):
        ts = pd.Series(np.random.rand(100) * 2, index=pd.date_range("2020-01-01", periods=100))
        t0 = 10
        gpd_params = {}
        exceedances = get_exceedance_peaks_over_threshold(
            ts=ts,
            t0=t0,
            anomaly_type="high",
            q=0.9
        )
        anomaly_scores = fit_exceedance(exceedances, t0, gpd_params)

        self.assertIsInstance(anomaly_scores, pd.Series)
        self.assertEqual(len(anomaly_scores), len(ts) - t0)
        self.assertTrue(all(isinstance(gpd_params[i], dict) for i in gpd_params))

    def test_fit_exceedance_with_invalid_ts(self):
        t0 = 10
        gpd_params = {}

        with self.assertRaises(TypeError):
            fit_exceedance("not a series", t0, gpd_params)

    def test_fit_exceedance_with_invalid_t0(self):
        ts = pd.Series(np.random.rand(100) * 2, index=pd.date_range("2020-01-01", periods=100))
        gpd_params = {}

        with self.assertRaises(ValueError):
            fit_exceedance(ts, None, gpd_params)

    def tearDown(self) -> None:
        return super().tearDown()
