from unittest import TestCase

import pandas as pd

import anomalytics as atics
from anomalytics.models.abstract import Detector


class TestDetector(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sample_1_ts = pd.Series(
            data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range(start="2023-01-01", periods=10)
        )
        self.ae_detector = atics.get_detector(method="AE", dataset=self.sample_1_ts)
        self.bm_detector = atics.get_detector(method="BM", dataset=self.sample_1_ts)
        self.dbscan_detector = atics.get_detector(method="DBSCAN", dataset=self.sample_1_ts)
        self.isof_detector = atics.get_detector(method="ISOF", dataset=self.sample_1_ts)
        self.mad_detector = atics.get_detector(method="MAD", dataset=self.sample_1_ts)
        self.svm_detector = atics.get_detector(method="1CSVM", dataset=self.sample_1_ts)
        self.pot_detector = atics.get_detector(method="POT", dataset=self.sample_1_ts)
        self.zs_detector = atics.get_detector(method="ZS", dataset=self.sample_1_ts)

    def test_instance_is_abstract_class(self):
        self.assertIsInstance(obj=self.ae_detector, cls=Detector)
        self.assertIsInstance(obj=self.bm_detector, cls=Detector)
        self.assertIsInstance(obj=self.dbscan_detector, cls=Detector)
        self.assertIsInstance(obj=self.isof_detector, cls=Detector)
        self.assertIsInstance(obj=self.mad_detector, cls=Detector)
        self.assertIsInstance(obj=self.svm_detector, cls=Detector)
        self.assertIsInstance(obj=self.pot_detector, cls=Detector)
        self.assertIsInstance(obj=self.zs_detector, cls=Detector)

    def test_string_method(self):
        self.assertEqual(first=str(self.ae_detector), second="AE")
        self.assertEqual(first=str(self.bm_detector), second="BM")
        self.assertEqual(first=str(self.dbscan_detector), second="DBSCAN")
        self.assertEqual(first=str(self.isof_detector), second="ISOF")
        self.assertEqual(first=str(self.mad_detector), second="MAD")
        self.assertEqual(first=str(self.svm_detector), second="1CSVM")
        self.assertEqual(first=str(self.pot_detector), second="POT")
        self.assertEqual(first=str(self.zs_detector), second="ZS")

    def test_get_extremes_pot(self):
        self.pot_detector.get_extremes(q=0.9)

        expected_exceedance_threshold = pd.Series(
            [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6.4, 7.3, 8.2, 9.1], index=self.sample_1_ts.index
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
            index=self.sample_1_ts.index,
            name="exceedances",
        )

        pd.testing.assert_series_equal(
            self.pot_detector._POTDetector__exceedance_threshold, expected_exceedance_threshold
        )
        pd.testing.assert_series_equal(self.pot_detector._POTDetector__exceedance, expected_exceedance)

    def test_genpareto_fit_from_pot_detecto(self):
        self.pot_detector.get_extremes(q=0.90)
        self.pot_detector.fit()

        expected_anomaly_scores = pd.Series(
            data=[1.922777880970598, 2.445890926224859, 3.6935717350888506, 3121651314.625431],
            index=self.sample_1_ts.index[6:],
            name="anomaly scores",
        )
        expected_params = {
            0: {
                "index": pd.Timestamp("2023-01-07 00:00:00"),
                "c": -1.6804238287454643,
                "loc": 0,
                "scale": 1.5123814458709186,
                "p_value": 0.5200808735615424,
                "anomaly_score": 1.922777880970598,
            },
        }

        pd.testing.assert_series_equal(self.pot_detector._POTDetector__anomaly_score, expected_anomaly_scores)
        self.assertEqual(self.pot_detector._POTDetector__params[0], expected_params[0])

    def tearDown(self) -> None:
        return super().tearDown()
