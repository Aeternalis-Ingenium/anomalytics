import unittest

import numpy as np
import pandas as pd
import pytest

from anomalytics import get_anomaly, get_anomaly_score, get_exceedance_peaks_over_threshold
from anomalytics.stats import get_anomaly_threshold, get_threshold_peaks_over_threshold


@pytest.mark.usefixtures("get_sample_1_ts")
class TestPeaksOverThreshold(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=42)
        self.random_sample_2_ts = pd.Series(
            data=np.random.rand(100), index=pd.date_range(start="2023-01-01", periods=100)
        )
        self.random_sample_3_ts = pd.Series(
            data=np.random.rand(100) * 2, index=pd.date_range("2020-01-01", periods=100)
        )

    def test_calculate_threshold_for_high_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="high", q=0.90)  # type: ignore
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))  # type: ignore

    def test__calculate_threshold_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="low", q=0.10)  # type: ignore
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))  # type: ignore

    def test_invalid_anomaly_type_in_threshold_calculation_function(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="invalid", q=0.90)  # type: ignore

    def test_invalid_ts_type_in_threshold_calculation_function(self):
        with self.assertRaises(TypeError):
            get_threshold_peaks_over_threshold(dataset=[1, 2, 3, 4], t0=3, anomaly_type="high", q=0.90)

    def test_invalid_t0_value_in_threshold_calculation_function(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=None, anomaly_type="high", q=0.90)  # type: ignore

    def test_extract_exceedance_for_high_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_2_ts, t0=5, anomaly_type="high", q=0.90)  # type: ignore
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_2_ts, threshold_dataset=pot_threshold, t0=5, anomaly_type="high", q=0.90
        )
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.random_sample_2_ts))
        self.assertTrue((pot_exceedance >= 0).all())

    def test_extract_exceedance_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_2_ts, t0=5, anomaly_type="low", q=0.10)  # type: ignore
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_2_ts, t0=5, threshold_dataset=pot_threshold, anomaly_type="low", q=0.10
        )
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.random_sample_2_ts))
        self.assertTrue((pot_exceedance >= 0).all())

    def test_invalid_anomaly_type_in_exceedance_extraction_function(self):
        with self.assertRaises(ValueError):
            get_exceedance_peaks_over_threshold(
                dataset=self.random_sample_2_ts,
                threshold_dataset=self.random_sample_2_ts,
                t0=5,
                anomaly_type="invalid",
                q=0.90,
            )  # type: ignore

    def test_invalid_ts_type_in_exceedance_extraction_function(self):
        with self.assertRaises(TypeError):
            get_exceedance_peaks_over_threshold(
                dataset="Not aseries", threshold_dataset="Not aseries", t0=5, anomaly_type="high", q=0.90
            )  # type: ignore

    def test_invalid_t0_value_in_exceedance_extraction_function(self):
        with self.assertRaises(ValueError):
            get_exceedance_peaks_over_threshold(
                dataset=self.random_sample_2_ts, threshold_dataset=[1, 2, 3, 4], t0=None, anomaly_type="high", q=0.90
            )  # type: ignore

    def test_fit_exceedance_with_valid_input(self):
        t0 = 10
        gpd_params: dict = {}
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_3_ts, t0=t0, anomaly_type="high", q=0.90)  # type: ignore
        exceedances = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_3_ts, threshold_dataset=pot_threshold, t0=t0, anomaly_type="high", q=0.90
        )
        anomaly_scores = get_anomaly_score(ts=exceedances, t0=t0, gpd_params=gpd_params)

        self.assertIsInstance(anomaly_scores, pd.Series)
        self.assertEqual(len(anomaly_scores), len(self.random_sample_3_ts.values) - t0)
        self.assertTrue(all(isinstance(gpd_params[i], dict) for i in gpd_params))

    def test_fit_exceedance_with_invalid_ts(self):
        t0 = 10
        gpd_params: dict = {}

        with self.assertRaises(TypeError):
            get_anomaly_score(ts="not a series", t0=t0, gpd_params=gpd_params)

    def test_fit_exceedance_with_invalid_t0(self):
        gpd_params: dict = {}

        with self.assertRaises(ValueError):
            get_anomaly_score(ts=self.random_sample_3_ts, t0=None, gpd_params=gpd_params)  # type: ignore

    def test_get_anomaly_threshold_with_valid_input(self):
        t1 = 50
        q = 0.90
        anomaly_threshold = get_anomaly_threshold(ts=self.random_sample_2_ts, t1=t1, q=q)

        self.assertIsInstance(anomaly_threshold, float)
        self.assertTrue(0 <= anomaly_threshold <= 1)

    def test_get_anomaly_threshold_with_invalid_ts(self):
        t1 = 50
        q = 0.90

        with self.assertRaises(TypeError):
            get_anomaly_threshold(ts="not a series", t1=t1, q=q)

    def test_confirm_correct_quantile_calculation_for_anomaly_threshold(self):
        ts = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=pd.date_range("2020-01-01", periods=10)
        )
        t1 = 5
        q = 0.90
        t1_ts = ts.iloc[:t1]

        expected_anomaly_threshold = np.quantile(a=t1_ts.values, q=q)
        anomaly_threshold = get_anomaly_threshold(ts=ts, t1=t1, q=q)

        self.assertEqual(anomaly_threshold, expected_anomaly_threshold)

    def test_get_anomaly_with_valid_input(self):
        self.random_sample_2_ts.iloc[75:] = self.random_sample_2_ts.iloc[75:] * 5

        t1 = 50
        q = 0.90
        anomalies = get_anomaly(ts=self.random_sample_2_ts, t1=t1, q=q)

        self.assertIsInstance(anomalies, pd.Series)

        expected_anomalies = self.random_sample_2_ts.iloc[t1:] > get_anomaly_threshold(
            ts=self.random_sample_2_ts, t1=t1, q=q
        )

        self.assertTrue((anomalies == expected_anomalies).all())

    def test_get_anomaly_with_invalid_ts(self):
        t1 = 50
        q = 0.90

        with self.assertRaises(TypeError):
            get_anomaly(ts="not a series", t1=t1, q=q)

    def tearDown(self) -> None:
        return super().tearDown()
