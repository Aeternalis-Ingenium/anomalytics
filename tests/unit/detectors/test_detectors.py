import unittest

import pytest

import anomalytics as atics
from anomalytics.models.abstract import Detector
from anomalytics.models.autoencoder import AutoencoderDetector
from anomalytics.models.block_maxima import BlockMaximaDetector
from anomalytics.models.dbscan import DBSCANDetector
from anomalytics.models.isoforest import IsoForestDetector
from anomalytics.models.mad import MADDetector
from anomalytics.models.one_class_svm import OneClassSVMDetector
from anomalytics.models.peaks_over_threshold import POTDetector
from anomalytics.models.zscore import ZScoreDetector


@pytest.mark.usefixtures("get_sample_1_ts")
class TestDetector(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ae_detector = atics.get_detector(method="AE", dataset=self.sample_1_ts)  # type: ignore
        self.bm_detector = atics.get_detector(method="BM", dataset=self.sample_1_ts)  # type: ignore
        self.ds_detector = atics.get_detector(method="DBSCAN", dataset=self.sample_1_ts)  # type: ignore
        self.isof_detector = atics.get_detector(method="ISOF", dataset=self.sample_1_ts)  # type: ignore
        self.mad_detector = atics.get_detector(method="MAD", dataset=self.sample_1_ts)  # type: ignore
        self.svm_detector = atics.get_detector(method="1CSVM", dataset=self.sample_1_ts)  # type: ignore
        self.pot_detector = atics.get_detector(method="POT", dataset=self.sample_1_ts)  # type: ignore
        self.zs_detector = atics.get_detector(method="ZS", dataset=self.sample_1_ts)  # type: ignore

    def test_construct_detector_classes(self):
        self.assertIsInstance(obj=self.ae_detector, cls=AutoencoderDetector)
        self.assertIsInstance(obj=self.bm_detector, cls=BlockMaximaDetector)
        self.assertIsInstance(obj=self.ds_detector, cls=DBSCANDetector)
        self.assertIsInstance(obj=self.isof_detector, cls=IsoForestDetector)
        self.assertIsInstance(obj=self.mad_detector, cls=MADDetector)
        self.assertIsInstance(obj=self.svm_detector, cls=OneClassSVMDetector)
        self.assertIsInstance(obj=self.pot_detector, cls=POTDetector)
        self.assertIsInstance(obj=self.zs_detector, cls=ZScoreDetector)

    def test_detector_string_method(self):
        self.assertEqual(first=str(self.ae_detector), second="AE")
        self.assertEqual(first=str(self.bm_detector), second="BM")
        self.assertEqual(first=str(self.ds_detector), second="DBSCAN")
        self.assertEqual(first=str(self.isof_detector), second="ISOF")
        self.assertEqual(first=str(self.mad_detector), second="MAD")
        self.assertEqual(first=str(self.svm_detector), second="1CSVM")
        self.assertEqual(first=str(self.pot_detector), second="POT")
        self.assertEqual(first=str(self.zs_detector), second="ZS")

    def test_methodological_detector_class_it_subclass_abstract_class(self):
        assert issubclass(type(self.ae_detector), Detector)
        assert issubclass(type(self.bm_detector), Detector)
        assert issubclass(type(self.ds_detector), Detector)
        assert issubclass(type(self.isof_detector), Detector)
        assert issubclass(type(self.mad_detector), Detector)
        assert issubclass(type(self.svm_detector), Detector)
        assert issubclass(type(self.pot_detector), Detector)
        assert issubclass(type(self.zs_detector), Detector)

    def tearDown(self) -> None:
        return super().tearDown()
