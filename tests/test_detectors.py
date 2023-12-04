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

    def tearDown(self) -> None:
        return super().tearDown()
