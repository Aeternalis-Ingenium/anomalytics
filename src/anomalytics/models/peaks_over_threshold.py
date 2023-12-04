import logging
import typing
import warnings

import numpy as np
import pandas as pd

from anomalytics.models.abstract import Detector
from anomalytics.time_windows.time_window import set_time_window

logger = logging.getLogger(__name__)


class POTDetector(Detector):
    __slots__ = [
        "__dataset",
        "__time_window",
        "__anomaly_type",
        "__exceedance_threshold",
        "__exceedance",
        "__anomaly_score",
        "__anomaly_threshold",
        "__anomaly",
        "__eval",
        "__params",
    ]

    __anomaly_type: typing.Literal["high", "low"]
    __dataset: typing.Union[pd.DataFrame, pd.Series]
    __time_window: typing.Tuple[int, int, int]
    __exceedance_threshold: typing.Union[pd.DataFrame, pd.Series]
    __exceedance: typing.Union[pd.DataFrame, pd.Series]
    __anomaly_score: typing.Union[pd.DataFrame, pd.Series]
    __anomaly_threshold: typing.Union[pd.DataFrame, pd.Series]
    __anomaly: typing.Union[pd.DataFrame, pd.Series]
    __eval: pd.DataFrame
    __params: dict

    def __init__(
        self, dataset: typing.Union[pd.DataFrame, pd.Series], anomaly_type: typing.Literal["high", "low"] = "high"
    ):
        """
        Initialize POT model for anomaly detection.

        ## Parameters
        ----------
        dataset : typing.Union[pandas.DataFrame, pandas.Series]
            DataFame or Series objects to be analyzed.
            Index must be date-time and values must be numeric.

        anomaly_type : typing.Literal["high", "low"]
            Defining which kind of anomaly are we expecting.
        """
        logger.info("start initialization of POT detection model")

        dataset = dataset.copy(deep=True)

        if not isinstance(dataset.index, pd.DatetimeIndex):
            try:
                msg = "Invalid data type! The dataset index is not pandas.DatetimeIndex - start converting to `pandas.DatetimeIndex`"
                logger.debug(msg)
                warnings.warn(msg, category=RuntimeWarning)
                dataset.index = pd.to_datetime(dataset.index)
            except TypeError as _error:
                raise ValueError(
                    f"Invalid data type! The dataset index is not and can not be converted to `pandas.DatetimeIndex`"
                ) from _error

        if not np.issubdtype(dataset.dtype, np.number):
            try:
                msg = "Invalid data type! The dataset value is not `numpy.numeric` - start converting to `numpyp.float64`"
                logger.debug(msg)
                warnings.warn(msg, category=RuntimeWarning)
                dataset = dataset.astype(np.float64)
            except ValueError as _error:
                raise TypeError(
                    f"Invalid data type! The dataset value is and can not be converted to `numpyp.float64`"
                ) from _error

        self.__anomaly_type = anomaly_type
        self.__dataset = dataset
        self.__time_window = set_time_window(
            total_rows=self.__dataset.shape[0],
            method="POT",
            analysis_type="real-time",
            t0_pct=0.70,
            t1_pct=0.3,
            t2_pct=0.0,
        )
        self.__exceedance_threshold = None
        self.__exceedance = None
        self.__anomaly_score = None
        self.__anomaly_threshold = None
        self.__anomaly = None
        self.__eval = None
        self.__params = {}

        logger.info("successfully initialized POT detection model")

    def fit(self) -> None:
        raise NotImplementedError("Not yet implemented!")

    def detect(self) -> None:
        raise NotImplementedError("Not yet implemented!")

    def evaluate(self) -> None:
        raise NotImplementedError("Not yet implemented!")

    @property
    def params(self) -> dict:  # type: ignore
        raise NotImplementedError("Not yet implemented!")

    def set_params(self, **kwargs: str | int | float | None) -> None:
        raise NotImplementedError("Not yet implemented!")

    def __str__(self) -> str:
        return "POT"
