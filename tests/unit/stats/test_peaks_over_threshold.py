import typing
import unittest

import numpy as np
import pandas as pd
import pytest

from anomalytics import get_anomaly, get_anomaly_score, get_exceedance_peaks_over_threshold
from anomalytics.stats import get_anomaly_threshold, get_threshold_peaks_over_threshold


@pytest.mark.usefixtures("get_sample_1_ts", "get_sample_1_df")
class TestPeaksOverThreshold(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        np.random.seed(seed=42)
        self.random_sample_2_ts = pd.Series(
            data=np.random.rand(100), index=pd.date_range(start="2023-01-01", periods=100)
        )
        self.random_sample_3_ts = pd.Series(
            data=np.random.rand(100) * 2, index=pd.date_range("2020-01-01", periods=100)
        )

    def test_calculate_threshold_dataframe_for_high_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_df, t0=6, anomaly_type="high", q=0.99)  # type: ignore

        expected_pot_threshold = pd.DataFrame(
            data={
                "feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )
        pd.testing.assert_frame_equal(left=pot_threshold, right=expected_pot_threshold)
        pd.testing.assert_series_equal(
            left=pot_threshold["feature_1"], right=expected_pot_threshold["feature_1"]  # type: ignore
        )
        pd.testing.assert_series_equal(
            left=pot_threshold["feature_2"], right=expected_pot_threshold["feature_2"]  # type: ignore
        )

    def test_calculate_threshold_dataframe_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_df, t0=6, anomaly_type="low", q=0.01)  # type: ignore

        expected_pot_threshold = pd.DataFrame(
            data={
                "feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )
        pd.testing.assert_frame_equal(left=pot_threshold, right=expected_pot_threshold)
        pd.testing.assert_series_equal(
            left=pot_threshold["feature_1"], right=expected_pot_threshold["feature_1"]  # type: ignore
        )
        pd.testing.assert_series_equal(
            left=pot_threshold["feature_2"], right=expected_pot_threshold["feature_2"]  # type: ignore
        )

    def test_exceedance_dataframe_for_high_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_df, t0=6, anomaly_type="high", q=0.99)  # type: ignore
        expected_pot_exceedance = pd.DataFrame(
            data={
                "feature_1": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.5,
                    0.6000000000000085,
                    0.7000000000000028,
                    0.7999999999999972,
                    0.9000000000000057,
                ],
                "feature_2": [
                    0,
                    0,
                    0,
                    0.5999999999999943,
                    0,
                    0,
                    2.3400000000000176,
                    0,
                    1.1200000000000045,
                    1.4399999999999977,
                ],
            }
        )
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.sample_1_df, threshold_dataset=pot_threshold, anomaly_type="high"  # type: ignore
        )

        pd.testing.assert_frame_equal(left=pot_exceedance, right=expected_pot_exceedance)
        pd.testing.assert_series_equal(left=pot_exceedance["feature_1"], right=expected_pot_exceedance["feature_1"])
        pd.testing.assert_series_equal(left=pot_exceedance["feature_2"], right=expected_pot_exceedance["feature_2"])

    def test_exceedance_dataframe_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_df, t0=6, anomaly_type="low", q=0.01)  # type: ignore
        expected_pot_exceedance = pd.DataFrame(
            data={
                "feature_1": [49.5, 39.5, 29.5, 19.5, 9.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                "feature_2": [
                    20.400000000000006,
                    18.400000000000006,
                    11.400000000000006,
                    0.0,
                    12.400000000000006,
                    20.400000000000006,
                    0.0,
                    17.669999999999987,
                    0.0,
                    0.0,
                ],
            }
        )
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.sample_1_df, threshold_dataset=pot_threshold, anomaly_type="low"  # type: ignore
        )

        pd.testing.assert_frame_equal(left=pot_exceedance, right=expected_pot_exceedance)
        pd.testing.assert_series_equal(left=pot_exceedance["feature_1"], right=expected_pot_exceedance["feature_1"])
        pd.testing.assert_series_equal(left=pot_exceedance["feature_2"], right=expected_pot_exceedance["feature_2"])

    def test_calculate_threshold_series_for_high_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="high", q=0.90)  # type: ignore
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))  # type: ignore

    def test__calculate_threshold_series_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="low", q=0.10)  # type: ignore
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))  # type: ignore

    def test_invalid_anomaly_type_in_threshold_series_calculation_function(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="invalid", q=0.90)  # type: ignore

    def test_invalid_ts_type_in_threshold_series_calculation_function(self):
        with self.assertRaises(TypeError):
            get_threshold_peaks_over_threshold(dataset=[1, 2, 3, 4], t0=3, anomaly_type="high", q=0.90)

    def test_invalid_t0_value_in_threshold_series_calculation_function(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=None, anomaly_type="high", q=0.90)  # type: ignore

    def test_extract_exceedance_series_for_high_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_2_ts, t0=5, anomaly_type="high", q=0.90)  # type: ignore
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_2_ts, threshold_dataset=pot_threshold, anomaly_type="high"
        )
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.random_sample_2_ts))
        self.assertTrue((pot_exceedance >= 0).all())

    def test_extract_exceedance_series_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_2_ts, t0=5, anomaly_type="low", q=0.10)  # type: ignore
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_2_ts, threshold_dataset=pot_threshold, anomaly_type="low"
        )
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.random_sample_2_ts))
        self.assertTrue((pot_exceedance >= 0).all())

    def test_invalid_anomaly_type_in_exceedance_series_extraction_function(self):
        with self.assertRaises(ValueError):
            get_exceedance_peaks_over_threshold(
                dataset=self.random_sample_2_ts,
                threshold_dataset=self.random_sample_2_ts,
                anomaly_type="invalid",
            )  # type: ignore

    def test_invalid_dataset_type_in_exceedance_series_extraction_function(self):
        with self.assertRaises(TypeError):
            get_exceedance_peaks_over_threshold(
                dataset="Not aseries", threshold_dataset="Not aseries", anomaly_type="high"
            )  # type: ignore

    def test_fit_exceedance_dataframe_with_valid_input(self):
        expected_anomaly_score = pd.DataFrame(
            data={
                "feature_1_anomaly_score": [float("inf"), float("inf"), float("inf"), float("inf")],
                "feature_2_anomaly_score": [float("inf"), 0.0, 1.6577266220223992, 2.099151208696442],
                "total_anomaly_score": [float("inf"), float("inf"), float("inf"), float("inf")],
            }
        )
        expected_params = {
            0: {
                "feature_1": {
                    "c": -2.687778724221391,
                    "loc": 0,
                    "scale": 1.3438893621106958,
                    "p_value": 0.0,
                    "anomaly_score": float("inf"),
                },
                "feature_2": {
                    "c": -2.793066758890821,
                    "loc": 0,
                    "scale": 1.675840055334477,
                    "p_value": 0.0,
                    "anomaly_score": float("inf"),
                },
                "total_anomaly_score": float("inf"),
            },
            1: {
                "feature_1": {
                    "c": -1.9076907886853052,
                    "loc": 0,
                    "scale": 1.1446144732111996,
                    "p_value": 0.0,
                    "anomaly_score": float("inf"),
                },
                "feature_2": {"c": 0.0, "loc": 0.0, "scale": 0.0, "p_value": 0.0, "anomaly_score": 0.0},
                "total_anomaly_score": float("inf"),
            },
            2: {
                "feature_1": {
                    "c": -2.352491882657089,
                    "loc": 0,
                    "scale": 1.646744317859969,
                    "p_value": 0.0,
                    "anomaly_score": float("inf"),
                },
                "feature_2": {
                    "c": -1.2885621357496342,
                    "loc": 0,
                    "scale": 3.015235397654167,
                    "p_value": 0.6032357728441475,
                    "anomaly_score": 1.6577266220223992,
                },
                "total_anomaly_score": float("inf"),
            },
            3: {
                "feature_1": {
                    "c": -1.8227953175299825,
                    "loc": 0,
                    "scale": 1.4582362540239808,
                    "p_value": 0.0,
                    "anomaly_score": float("inf"),
                },
                "feature_2": {
                    "c": -1.2885621357496342,
                    "loc": 0,
                    "scale": 3.015235397654167,
                    "p_value": 0.47638302369889446,
                    "anomaly_score": 2.099151208696442,
                },
                "total_anomaly_score": float("inf"),
            },
        }

        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_df, t0=6, anomaly_type="high", q=0.99)  # type: ignore
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.sample_1_df, threshold_dataset=pot_threshold, anomaly_type="high"  # type: ignore
        )

        gpd_params: typing.Dict = {}
        anomaly_scores = get_anomaly_score(exceedance_dataset=pot_exceedance, t0=6, gpd_params=gpd_params)

        pd.testing.assert_frame_equal(left=anomaly_scores, right=expected_anomaly_score)
        self.assertEqual(first=gpd_params, second=expected_params)

    def test_fit_exceedance_series_with_valid_input(self):
        t0 = 10
        gpd_params: dict = {}
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_3_ts, t0=t0, anomaly_type="high", q=0.90)  # type: ignore
        pot_exceedances = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_3_ts, threshold_dataset=pot_threshold, anomaly_type="high"
        )
        anomaly_scores = get_anomaly_score(exceedance_dataset=pot_exceedances, t0=t0, gpd_params=gpd_params)

        self.assertIsInstance(anomaly_scores, pd.Series)
        self.assertEqual(len(anomaly_scores), len(self.random_sample_3_ts.values) - t0)
        self.assertTrue(all(isinstance(gpd_params[i], dict) for i in gpd_params))

    def test_fit_exceedance_series_with_invalid_ts(self):
        t0 = 10
        gpd_params: dict = {}

        with self.assertRaises(TypeError):
            get_anomaly_score(ts="not a series", t0=t0, gpd_params=gpd_params)

    def test_fit_exceedance_series_with_invalid_t0(self):
        gpd_params: dict = {}

        with self.assertRaises(ValueError):
            get_anomaly_score(exceedance_dataset=self.random_sample_3_ts, t0=None, gpd_params=gpd_params)  # type: ignore

    def test_get_anomaly_threshold_series_with_valid_input(self):
        t1 = 50
        q = 0.90
        anomaly_threshold = get_anomaly_threshold(ts=self.random_sample_2_ts, t1=t1, q=q)

        self.assertIsInstance(anomaly_threshold, float)
        self.assertTrue(0 <= anomaly_threshold <= 1)

    def test_get_anomaly_threshold_series_with_invalid_ts(self):
        t1 = 50
        q = 0.90

        with self.assertRaises(TypeError):
            get_anomaly_threshold(ts="not a series", t1=t1, q=q)

    def test_confirm_correct_quantile_calculation_for_anomaly_threshold_series(self):
        ts = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=pd.date_range("2020-01-01", periods=10)
        )
        t1 = 5
        q = 0.90
        t1_ts = ts.iloc[:t1]

        expected_anomaly_threshold = np.quantile(a=t1_ts.values, q=q)
        anomaly_threshold = get_anomaly_threshold(ts=ts, t1=t1, q=q)

        self.assertEqual(anomaly_threshold, expected_anomaly_threshold)

    def test_get_anomaly_series_with_valid_input(self):
        self.random_sample_2_ts.iloc[75:] = self.random_sample_2_ts.iloc[75:] * 5

        t1 = 50
        q = 0.90
        anomalies = get_anomaly(ts=self.random_sample_2_ts, t1=t1, q=q)

        self.assertIsInstance(anomalies, pd.Series)

        expected_anomalies = self.random_sample_2_ts.iloc[t1:] > get_anomaly_threshold(
            ts=self.random_sample_2_ts, t1=t1, q=q
        )

        self.assertTrue((anomalies == expected_anomalies).all())

    def test_get_anomaly_series_with_invalid_ts(self):
        t1 = 50
        q = 0.90

        with self.assertRaises(TypeError):
            get_anomaly(ts="not a series", t1=t1, q=q)

    def tearDown(self) -> None:
        return super().tearDown()
