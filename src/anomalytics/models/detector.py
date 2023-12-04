from __future__ import annotations

import logging
import typing

import pandas as pd

logger = logging.getLogger(__name__)


class FactoryDetector:
    def __init__(
        self,
        method: typing.Literal["AE", "BM", "DBSCAN", "ISOF", "MAD", "POT", "ZS", "1CSVM"],
        dataset: typing.Union[pd.DataFrame, pd.Series],
    ):
        self.method = method
        self.dataset = dataset

    def __call__(self):
        if self.method == "AE":
            from anomalytics.models.autoencoder import AutoencoderDetector

            return AutoencoderDetector(dataset=self.dataset)

        elif self.method == "BM":
            from anomalytics.models.block_maxima import BlockMaximaDetector

            return BlockMaximaDetector(dataset=self.dataset)

        elif self.method == "DBSCAN":
            from anomalytics.models.dbscan import DBSCANDetector

            return DBSCANDetector(dataset=self.dataset)

        elif self.method == "ISOF":
            from anomalytics.models.isoforest import IsoForestDetector

            return IsoForestDetector(dataset=self.dataset)

        elif self.method == "MAD":
            from anomalytics.models.mad import MADDetector

            return MADDetector(dataset=self.dataset)

        elif self.method == "1CSVM":
            from anomalytics.models.one_class_svm import OneClassSVMDetector

            return OneClassSVMDetector(dataset=self.dataset)

        elif self.method == "POT":
            from anomalytics.models.peaks_over_threshold import POTDetector

            return POTDetector(dataset=self.dataset)

        elif self.method == "ZS":
            from anomalytics.models.zscore import ZScoreDetector

            return ZScoreDetector(dataset=self.dataset)

        raise ValueError(
            "Invalid value! Available `method` arguments: 'AE', 'BM', 'DBSCAN', 'ISOF', 'MAD', 'POT', 'ZS', '1CSVM'"
        )


def get_detector(
    method: typing.Literal["AE", "BM", "DBSCAN", "ISOF", "MAD", "POT", "ZS", "1CSVM"],
    dataset: typing.Union[pd.DataFrame, pd.Series],
):
    return FactoryDetector(method=method, dataset=dataset)()
