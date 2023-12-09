import typing

import pandas as pd

from anomalytics.models.abstract import Detector


class AutoencoderDetector(Detector):
    """
    Anomaly detector class that implements the "Autoencoder" method.
    ! TODO: Implement anomaly detection with autoencoder method!
    """

    __slots__ = [
        "__anomaly_type",
        "__dataset__",
    ]

    __anomaly_type: typing.Literal["high", "low"]
    __dataset: typing.Union[pd.DataFrame, pd.Series]

    def __init__(
        self, dataset: typing.Union[pd.DataFrame, pd.Series], anomaly_type: typing.Literal["high", "low"] = "high"
    ):
        """
        Initialize Autoencoder model for anomaly detection.

        ## Parameters
        ----------
        dataset : typing.Union[pandas.DataFrame, pandas.Series]
            DataFame or Series objects to be analyzed.
            Index must be date-time and values must be numeric.

        anomaly_type : typing.Literal["high", "low"]
            Defining which kind of anomaly are we expecting.
        """

        self.__anomaly_type = anomaly_type
        self.__dataset = dataset

    def fit(self) -> None:
        raise NotImplementedError("Not yet implemented!")

    def detect(self) -> None:
        raise NotImplementedError("Not yet implemented!")

    def evaluate(self, method: typing.Literal["ks", "qq"] = "ks") -> None:
        raise NotImplementedError("Not yet implemented!")

    @property
    def return_detected_anomalies(self) -> typing.Union[pd.DataFrame, pd.Series]:
        raise NotImplementedError("Not yet implemented!")

    @property
    def params(self) -> dict:  # type: ignore
        raise NotImplementedError("Not yet implemented!")

    def __str__(self) -> str:
        return "AE"
