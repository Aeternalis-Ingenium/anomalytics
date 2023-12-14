import typing
import unittest

import numpy as np
import pandas as pd
import pytest

from anomalytics import get_anomaly, get_anomaly_score, get_exceedance_peaks_over_threshold
from anomalytics.stats import get_anomaly_threshold, get_threshold_peaks_over_threshold


@pytest.mark.usefixtures("get_sample_1_ts", "get_sample_1_df", "get_sample_2_df")
class TestPeaksOverThreshold(unittest.TestCase):
    def setUp(self):
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
        pd.testing.assert_frame_equal(pot_threshold, expected_pot_threshold)
        pd.testing.assert_series_equal(pot_threshold["feature_1"], expected_pot_threshold["feature_1"])  # type: ignore
        pd.testing.assert_series_equal(pot_threshold["feature_2"], expected_pot_threshold["feature_2"])  # type: ignore

    def test_calculate_threshold_dataframe_for_low_anomaly_type(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_df, t0=6, anomaly_type="low", q=0.01)  # type: ignore

        expected_pot_threshold = pd.DataFrame(
            data={
                "feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )
        pd.testing.assert_frame_equal(pot_threshold, expected_pot_threshold)
        pd.testing.assert_series_equal(pot_threshold["feature_1"], expected_pot_threshold["feature_1"])  # type: ignore
        pd.testing.assert_series_equal(pot_threshold["feature_2"], expected_pot_threshold["feature_2"])  # type: ignore

    def test_get_exceedance_peaks_over_threshold_dataframe_for_high_anomaly_type(self):
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

        pd.testing.assert_frame_equal(pot_exceedance, expected_pot_exceedance)
        pd.testing.assert_series_equal(pot_exceedance["feature_1"], expected_pot_exceedance["feature_1"])
        pd.testing.assert_series_equal(pot_exceedance["feature_2"], expected_pot_exceedance["feature_2"])

    def test_get_exceedance_peaks_over_threshold_dataframe_for_low_anomaly_type(self):
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

        pd.testing.assert_frame_equal(pot_exceedance, expected_pot_exceedance)
        pd.testing.assert_series_equal(pot_exceedance["feature_1"], expected_pot_exceedance["feature_1"])
        pd.testing.assert_series_equal(pot_exceedance["feature_2"], expected_pot_exceedance["feature_2"])

    def test_get_anomaly_score_dataframe_successful(self):
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

        pd.testing.assert_frame_equal(anomaly_scores, expected_anomaly_score)
        self.assertEqual(gpd_params, expected_params)

    def test_get_anomaly_threshold_dataframe_successful(self):
        expected_anomaly_threshold = 2.04403430931313
        t0, t1 = 30, 13
        # t2 = 7
        expected_pot_thresholds = pd.DataFrame(
            data={
                "feature_1": [
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    453.0,
                    449.50000000000006,
                    446.0,
                    442.5,
                    439.00000000000006,
                    435.5,
                    431.99999999999994,
                    428.50000000000017,
                    438.6,
                    454.90000000000003,
                    453.0,
                    451.5,
                    450.00000000000006,
                    448.50000000000006,
                    447.0,
                    445.5,
                    444.0,
                    442.50000000000006,
                    441.00000000000006,
                    454.90000000000003,
                ],
                "feature_2": [
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    535.0,
                    534.0,
                    533.0,
                    532.0,
                    531.0,
                    530.0,
                    529.0,
                    528.0,
                    527.0,
                    526.0,
                    525.0,
                    532.2,
                    531.4000000000001,
                    530.6,
                    534.2,
                    534.0,
                    533.8,
                    535.0,
                    535.0,
                    535.0,
                ],
            }
        )
        expected_pot_exceedance = pd.DataFrame(
            data={
                "feature_1": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    33.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    17.099999999999966,
                    0.0,
                    0.0,
                    25.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    26.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    203545.1,
                ],
                "feature_2": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    344460.3,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    59.299999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    42.299999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.7999999999999545,
                    0.0,
                    0.0,
                    61.799999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )
        expected_anomaly_scores = pd.DataFrame(
            data={
                "feature_1_anomaly_score": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5947881128257808,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float("inf"),
                ],
                "feature_2_anomaly_score": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0092577188337257,
                    0.0,
                    0.0,
                    2.1563458584349675,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "total_anomaly_score": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5947881128257808,
                    0.0,
                    1.0092577188337257,
                    0.0,
                    0.0,
                    2.1563458584349675,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float("inf"),
                ],
            }
        )

        pot_threshold = get_threshold_peaks_over_threshold(
            dataset=self.sample_2_df, t0=t0, anomaly_type="high", q=0.90  # type: ignore
        )

        pd.testing.assert_frame_equal(pot_threshold, expected_pot_thresholds)

        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.sample_2_df, threshold_dataset=pot_threshold, anomaly_type="high"  # type: ignore
        )

        pd.testing.assert_frame_equal(pot_exceedance, expected_pot_exceedance)

        params: typing.Dict = {}
        anomaly_score = get_anomaly_score(exceedance_dataset=pot_exceedance, t0=t0, gpd_params=params)

        pd.testing.assert_frame_equal(anomaly_score, expected_anomaly_scores)
        self.assertEqual(first=type(params), second=dict)

        anomaly_threshold = get_anomaly_threshold(anomaly_score_dataset=anomaly_score, t1=t1, q=0.90)

        self.assertEqual(anomaly_threshold, expected_anomaly_threshold)

    def test_get_anomaly_dataframe_successful(self):
        t0, t1 = 30, 13
        # t2 = 7

        pot_threshold = get_threshold_peaks_over_threshold(
            dataset=self.sample_2_df, t0=t0, anomaly_type="high", q=0.90  # type: ignore
        )
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.sample_2_df, threshold_dataset=pot_threshold, anomaly_type="high"  # type: ignore
        )
        params: typing.Dict = {}
        anomaly_score = get_anomaly_score(exceedance_dataset=pot_exceedance, t0=t0, gpd_params=params)
        anomaly_threshold = get_anomaly_threshold(anomaly_score_dataset=anomaly_score, t1=t1, q=0.90)

        detected_data = get_anomaly(anomaly_score_dataset=anomaly_score, threshold=anomaly_threshold, t1=t1)

        expected_detected_data = pd.Series(
            index=detected_data.index,
            data=[
                False,
                True,
                False,
                False,
                False,
                False,
                True,
            ],
            name="detected data",
        )

        self.assertIsInstance(detected_data, pd.Series)
        pd.testing.assert_series_equal(detected_data, expected_detected_data)

    def test_get_threshold_peaks_over_threshold_series_for_high_anomaly_type_successful(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="high", q=0.90)  # type: ignore
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))  # type: ignore

    def test_get_threshold_peaks_over_threshold_series_for_low_anomaly_type_successful(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="low", q=0.10)  # type: ignore
        self.assertIsInstance(pot_threshold, pd.Series)
        self.assertEqual(len(pot_threshold), len(self.sample_1_ts))  # type: ignore

    def test_get_exceedance_peaks_over_threshold_series_for_high_anomaly_successful(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_2_ts, t0=5, anomaly_type="high", q=0.90)  # type: ignore
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_2_ts, threshold_dataset=pot_threshold, anomaly_type="high"
        )
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.random_sample_2_ts))
        self.assertTrue((pot_exceedance >= 0).all())

    def test_get_exceedance_peaks_over_threshold_series_for_low_anomaly_successful(self):
        pot_threshold = get_threshold_peaks_over_threshold(dataset=self.random_sample_2_ts, t0=5, anomaly_type="low", q=0.10)  # type: ignore
        pot_exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.random_sample_2_ts, threshold_dataset=pot_threshold, anomaly_type="low"
        )
        self.assertIsInstance(pot_exceedance, pd.Series)
        self.assertEqual(len(pot_exceedance), len(self.random_sample_2_ts))
        self.assertTrue((pot_exceedance >= 0).all())

    def test_get_anomaly_score_series_successful(self):
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

    def test_get_anomaly_threshold_series_successfuls(self):
        t1 = 50
        q = 0.90
        anomaly_threshold = get_anomaly_threshold(anomaly_score_dataset=self.random_sample_2_ts, t1=t1, q=q)

        self.assertIsInstance(anomaly_threshold, float)
        self.assertTrue(0 <= anomaly_threshold <= 1)

    def test_confirm_correct_quantile_calculation_for_anomaly_threshold_series_successful(self):
        t1 = 5
        q = 0.90
        anomaly_score_dataset = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=pd.date_range("2020-01-01", periods=10)
        )
        t1_ts = anomaly_score_dataset.iloc[:t1]
        expected_anomaly_threshold = np.quantile(a=t1_ts.values, q=q)
        anomaly_threshold = get_anomaly_threshold(anomaly_score_dataset=anomaly_score_dataset, t1=t1, q=q)

        self.assertEqual(anomaly_threshold, expected_anomaly_threshold)

    def test_get_detected_data_series_successful(self):
        self.random_sample_2_ts.iloc[75:] = self.random_sample_2_ts.iloc[75:] * 5
        t1 = 50
        q = 0.90
        anomaly_threshold = get_anomaly_threshold(anomaly_score_dataset=self.random_sample_2_ts, t1=t1, q=q)
        expected_detected_data = self.random_sample_2_ts.iloc[t1:] > anomaly_threshold

        detected_data = get_anomaly(anomaly_score_dataset=self.random_sample_2_ts, threshold=anomaly_threshold, t1=t1)

        self.assertIsInstance(detected_data, pd.Series)
        self.assertTrue((detected_data == expected_detected_data).all())

    def test_get_threshold_peaks_over_threshold_with_invalid_dataset(self):
        with self.assertRaises(TypeError):
            get_threshold_peaks_over_threshold(dataset=[1, 2, 3, 4], t0=3, anomaly_type="high", q=0.90)

    def test_get_threshold_peaks_over_threshold_with_invalid_t0(self):
        with self.assertRaises(TypeError):
            get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=None, anomaly_type="high", q=0.90)  # type: ignore

    def test_get_threshold_peaks_over_threshold_with_invalid_anomaly_type(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="invalid", q=0.90)  # type: ignore

    def test_get_threshold_peaks_over_threshold_with_invalid_q(self):
        with self.assertRaises(ValueError):
            get_threshold_peaks_over_threshold(dataset=self.sample_1_ts, t0=3, anomaly_type="invalid", q=90)  # type: ignore

    def test_get_exceedance_peaks_over_threshold_with_invalid_dataset(self):
        with self.assertRaises(TypeError):
            get_exceedance_peaks_over_threshold(
                dataset="Not aseries", threshold_dataset=self.random_sample_3_ts, anomaly_type="high"  # type: ignore
            )

    def test_get_exceedance_peaks_over_threshold_with_invalid_threshold_dataset(self):
        with self.assertRaises(TypeError):
            get_exceedance_peaks_over_threshold(
                dataset=self.random_sample_2_ts,
                threshold_dataset="not threshold dataset",  # type: ignore
                anomaly_type="high",
            )

    def test_get_exceedance_peaks_over_threshold_with_invalid_anomaly_type(self):
        with self.assertRaises(ValueError):
            get_exceedance_peaks_over_threshold(
                dataset=self.random_sample_2_ts,
                threshold_dataset=self.random_sample_2_ts,
                anomaly_type="invalid",  # type: ignore
            )

    def test_get_anomaly_score_with_invalid_anomaly_score_dataset(self):
        with self.assertRaises(TypeError):
            get_anomaly_score(anomaly_score_dataset="not a series", t0=10, gpd_params={})  # type: ignore

    def test_get_anomaly_score_with_invalid_t0(self):
        with self.assertRaises(TypeError):
            get_anomaly_score(exceedance_dataset=self.random_sample_3_ts, t0=None, gpd_params={})  # type: ignore

    def test_get_anomaly_score_with_invalid_gpd_params(self):
        with self.assertRaises(TypeError):
            get_anomaly_score(exceedance_dataset=self.random_sample_3_ts, t0=10, gpd_params=[])  # type: ignore

    def test_get_anomaly_threshold_with_invalid_anomaly_score_dataset(self):
        with self.assertRaises(TypeError):
            get_anomaly_threshold(anomaly_score_dataset="not a series", t1=50, q=0.90)  # type: ignore

    def test_get_anomaly_threshold_with_invalid_t1(self):
        with self.assertRaises(TypeError):
            get_anomaly_threshold(anomaly_score_dataset=self.sample_1_df, t1="not t1", q=0.90)  # type: ignore

    def test_get_anomaly_threshold_with_invalid_q(self):
        with self.assertRaises(TypeError):
            get_anomaly_threshold(anomaly_score_dataset=self.sample_1_df, t1=50, q="not q")  # type: ignore

    def test_get_detected_data_with_invalid_anomaly_score_dataset(self):
        with self.assertRaises(TypeError):
            get_anomaly(anomaly_score_dataset="not a series", threshold=5.234, t1=50)  # type: ignore

    def test_get_detected_data_with_invalid_anomaly_threshold(self):
        with self.assertRaises(TypeError):
            get_anomaly(anomaly_score_dataset=self.sample_1_df, threshold="not a threshold", t1=50)  # type: ignore

    def test_get_detected_data_with_invalid_t1(self):
        with self.assertRaises(TypeError):
            get_anomaly(anomaly_score_dataset=self.sample_1_df, threshold=53.245, t1=0.5)  # type: ignore

    def tearDown(self) -> None:
        return super().tearDown()
