import unittest

import pandas as pd
import pytest

import anomalytics as atics
from anomalytics.models.peaks_over_threshold import POTDetector


@pytest.mark.usefixtures("get_sample_1_ts", "get_sample_2_ts")
class TestPOTDetector(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pot1_detector = atics.get_detector(method="POT", dataset=self.sample_1_ts)  # type: ignore
        self.pot2_detector = atics.get_detector(method="POT", dataset=self.sample_2_ts, anomaly_type="high")  # type: ignore

    def test_instance_is_pot_detector_class(self):
        self.assertIsInstance(obj=self.pot1_detector, cls=POTDetector)

    def test_detector_string_method(self):
        self.assertEqual(first=str(self.pot1_detector), second=str(POTDetector(dataset=self.sample_1_ts)))  # type: ignore

    def test_reset_time_window_to_historical(self):
        t0, t1, t2 = self.pot1_detector._POTDetector__time_window
        self.pot1_detector.reset_time_window(analysis_type="historical", t0_pct=0.80, t1_pct=0.15, t2_pct=0.05)
        self.assertNotEqual(t0, self.pot1_detector._POTDetector__time_window[0])
        self.assertNotEqual(t1, self.pot1_detector._POTDetector__time_window[1])
        self.assertNotEqual(t2, self.pot1_detector._POTDetector__time_window[2])

    def test_get_extremes_methods(self):
        self.pot1_detector.get_extremes(q=0.9)

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

        pd.testing.assert_series_equal(self.pot1_detector.exceedance_thresholds, expected_exceedance_threshold)
        pd.testing.assert_series_equal(self.pot1_detector.exceedances, expected_exceedance)

    def test_fit_with_genpareto_method(self):
        self.pot1_detector.get_extremes(q=0.90)
        self.pot1_detector.fit()

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

        pd.testing.assert_series_equal(self.pot1_detector._POTDetector__anomaly_score, expected_anomaly_scores)
        self.assertEqual(self.pot1_detector._POTDetector__params[0], expected_params[0])

    def test_compute_anomaly_threshold_method(self):
        expected_detected_data = [True]
        expected_anomaly_threshold = 1.8927400332325932

        self.pot2_detector.get_extremes(q=0.90)
        self.pot2_detector.fit()
        self.pot2_detector.detect(q=0.90)

        self.assertEqual(self.pot2_detector.anomaly_threshold, expected_anomaly_threshold)
        self.assertEqual(self.pot2_detector.detection_result.iloc[0], expected_detected_data)

    def test_evaluation_with_ks_1sample(self):
        self.pot2_detector.get_extremes(q=0.90)
        self.pot2_detector.fit()
        self.pot2_detector.detect(q=0.90)
        self.pot2_detector.evaluate(method="ks")

        expected_kstest_result = pd.DataFrame(
            data={
                "total_nonzero_exceedances": [50],
                "stats_distance": [0.33333333326007464],
                "p_value": [0.4234396436048128],
                "c": [-1.5741853768217173],
                "loc": [0],
                "scale": [71.62543464538814],
            }
        )

        pd.testing.assert_frame_equal(self.pot2_detector._POTDetector__eval, expected_kstest_result)

    def tearDown(self) -> None:
        return super().tearDown()
