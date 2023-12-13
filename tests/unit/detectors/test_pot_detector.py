import unittest

import pandas as pd
import pytest

import anomalytics as atics
from anomalytics.models.peaks_over_threshold import POTDetector


@pytest.mark.usefixtures("get_sample_1_ts", "get_sample_2_ts", "get_sample_3_df")
class TestPOTDetector(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pot1_series_detector = atics.get_detector(method="POT", dataset=self.sample_1_ts)  # type: ignore
        self.pot2_series_detector = atics.get_detector(method="POT", dataset=self.sample_2_ts, anomaly_type="low")  # type: ignore
        self.pot3_dataframe_detector = atics.get_detector(method="POT", dataset=self.sample_3_df)  # type: ignore
        self.pot4_dataframe_detector = atics.get_detector(method="POT", dataset=self.sample_3_df, anomaly_type="low")  # type: ignore

    def test_instance_is_pot_detector_class_successful(self):
        self.assertIsInstance(obj=self.pot1_series_detector, cls=POTDetector)

    def test_detector_string_method_successful(self):
        self.assertEqual(first=str(self.pot1_series_detector), second=str(POTDetector(dataset=self.sample_1_ts)))  # type: ignore

    def test_reset_time_window_to_historical_successful(self):
        t0 = self.pot1_series_detector.t0
        t1 = self.pot1_series_detector.t1
        t2 = self.pot1_series_detector.t2

        self.pot1_series_detector.reset_time_window(analysis_type="historical", t0_pct=0.80, t1_pct=0.15, t2_pct=0.05)

        self.assertNotEqual(t0, self.pot1_series_detector.t0)
        self.assertNotEqual(t1, self.pot1_series_detector.t1)
        self.assertNotEqual(t2, self.pot1_series_detector.t2)

    def test_exceedance_thresholds_dataframe_for_high_anomaly_type_successful(self):
        expected_exceedance_thresholds = pd.DataFrame(
            data={
                "feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
                "datetime": pd.date_range(start="2023-01-01", periods=10),
            }
        )

        self.pot3_dataframe_detector.get_extremes(q=0.99)

        pd.testing.assert_frame_equal(
            self.pot3_dataframe_detector.exceedance_thresholds, expected_exceedance_thresholds
        )
        pd.testing.assert_series_equal(
            self.pot3_dataframe_detector.exceedance_thresholds["feature_1"], expected_exceedance_thresholds["feature_1"]  # type: ignore
        )
        pd.testing.assert_series_equal(
            self.pot3_dataframe_detector.exceedance_thresholds["feature_2"], expected_exceedance_thresholds["feature_2"]  # type: ignore
        )

    def test_get_extremes_dataframe_with_high_anomaly_type_successful(self):
        expected_exceedances = pd.DataFrame(
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
                "datetime": pd.date_range(start="2023-01-01", periods=10),
            }
        )

        self.pot3_dataframe_detector.get_extremes(q=0.99)

        pd.testing.assert_frame_equal(self.pot3_dataframe_detector.exceedances, expected_exceedances)
        pd.testing.assert_series_equal(
            self.pot3_dataframe_detector.exceedances["feature_1"], expected_exceedances["feature_1"]  # type: ignore
        )
        pd.testing.assert_series_equal(
            self.pot3_dataframe_detector.exceedances["feature_2"], expected_exceedances["feature_2"]  # type: ignore
        )

    def test_fit_dataframe_with_high_anomaly_type_successful(self):
        expected_anomaly_scores = pd.DataFrame(
            data={
                "feature_1_anomaly_score": [float("inf"), float("inf"), float("inf"), float("inf")],
                "feature_2_anomaly_score": [float("inf"), 0.0, 1.2988597467759642, 2.129427676525411],
                "total_anomaly_score": [float("inf"), float("inf"), float("inf"), float("inf")],
                "datetime": pd.date_range(start="2023-01-01", periods=10)[6:],
            }
        )

        self.pot3_dataframe_detector.get_extremes(q=0.90)
        self.pot3_dataframe_detector.fit()

        pd.testing.assert_frame_equal(left=self.pot3_dataframe_detector.fit_result, right=expected_anomaly_scores)

    def test_get_extremes_series_for_high_anomaly_type_successful(self):
        self.pot1_series_detector.get_extremes(q=0.9)

        expected_exceedance_threshold = pd.Series(
            [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6.4, 7.3, 8.2, 9.1], index=self.sample_1_ts.index  # type: ignore
        )
        expected_exceedance = pd.Series(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.5999999999999996,
                0.7000000000000002,
                0.8000000000000007,
                0.9000000000000004,
            ],
            index=self.sample_1_ts.index,  # type: ignore
            name="exceedances",
        )

        pd.testing.assert_series_equal(self.pot1_series_detector.exceedance_thresholds, expected_exceedance_threshold)
        pd.testing.assert_series_equal(self.pot1_series_detector.exceedances, expected_exceedance)

    def test_fit_series_for_high_anomaly_type_successful(self):
        self.pot1_series_detector.get_extremes(q=0.90)
        self.pot1_series_detector.fit()

        expected_anomaly_scores = pd.Series(
            data=[float("inf"), float("inf"), float("inf"), float("inf")],
            index=self.sample_1_ts.index[6:],  # type: ignore
            name="anomaly scores",
        )
        expected_params = {
            0: {
                "index": pd.Timestamp("2023-01-07 00:00:00"),
                "c": -2.687778724221391,
                "loc": 0,
                "scale": 1.3438893621106958,
                "p_value": 0.0,
                "anomaly_score": float("inf"),
            },
        }

        pd.testing.assert_series_equal(self.pot1_series_detector.fit_result, expected_anomaly_scores)
        self.assertEqual(self.pot1_series_detector.params[0], expected_params[0])

    def test_detect_data_series_for_low_anomaly_type_successful(self):
        expected_detected_data = False
        expected_anomaly_threshold = 1.6609084761335131

        self.pot2_series_detector.get_extremes(q=0.90)
        self.pot2_series_detector.fit()
        self.pot2_series_detector.detect(q=0.90)

        self.assertEqual(self.pot2_series_detector.anomaly_threshold, expected_anomaly_threshold)
        self.assertEqual(self.pot2_series_detector.detection_result.iloc[0], expected_detected_data)

    def test_evaluation_with_ks_1sample_series_for_low_anomaly_type_successful(self):
        self.pot2_series_detector.get_extremes(q=0.90)
        self.pot2_series_detector.fit()
        self.pot2_series_detector.detect(q=0.90)
        self.pot2_series_detector.evaluate(method="ks")

        expected_kstest_result = pd.DataFrame(
            data={
                "total_nonzero_exceedances": [50],
                "stats_distance": [0.4489148721482511],
                "p_value": [0.08363867527012836],
                "c": [-2.3994051567842636],
                "loc": [0],
                "scale": [120.69007938624844],
            }
        )

        pd.testing.assert_frame_equal(self.pot2_series_detector.evaluation_result, expected_kstest_result)

    def tearDown(self) -> None:
        return super().tearDown()
