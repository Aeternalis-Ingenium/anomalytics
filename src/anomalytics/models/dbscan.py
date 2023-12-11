import datetime
import typing

import pandas as pd

from anomalytics.models.abstract import Detector


class DBSCANDetector(Detector):
    """
    Anomaly detector class that implements the "Density-Based Spatial Clustering of Applications with Noise" (D. B. S. C. A. N.) method.
    ! TODO: Implement anomaly detection with "DBSCAN" method!
    """

    __slots__ = [
        "__dataset",
        "__time_window",
        "__anomaly_type",
        "__fit_result",
        "__anomaly_threshold",
        "__detection",
        "__eval",
        "__params",
    ]

    __anomaly_type: typing.Literal["high", "low"]
    __dataset: typing.Union[pd.DataFrame, pd.Series]
    __time_window: typing.Tuple[int, int, int]
    __fit_result: typing.Union[pd.DataFrame, pd.Series]
    __anomaly_threshold: float
    __detection: typing.Union[pd.DataFrame, pd.Series]
    __eval: pd.DataFrame
    __params: typing.Dict

    def __init__(
        self, dataset: typing.Union[pd.DataFrame, pd.Series], anomaly_type: typing.Literal["high", "low"] = "high"
    ):
        """
        Initialize DBSCAN model for anomaly detection.

        ## Parameters
        ----------
        dataset : typing.Union[pandas.DataFrame, pandas.Series]
            DataFame or Series objects to be analyzed.
            Index must be date-time and values must be numeric.

        anomaly_type : typing.Literal["high", "low"]
            Defining which kind of anomaly are we expecting.
        """

        self.__dataset = dataset
        self.__anomaly_type = anomaly_type
        self.__time_window = None  # type: ignore
        self.__fit_result = None  # type: ignore
        self.__anomaly_threshold = None  # type: ignore
        self.__detection = None  # type: ignore
        self.__eval = None  # type: ignore
        self.__params = {}

    def fit(self) -> None:
        raise NotImplementedError("Not yet implemented!")

    def detect(self) -> None:
        raise NotImplementedError("Not yet implemented!")

    def evaluate(self, method: typing.Literal["ks", "qq"] = "ks") -> None:
        raise NotImplementedError("Not yet implemented!")

    @property
    def anomaly_threshold(self) -> float:
        """
        The anomaly threshold computed by the quantile method in `detect()`.

        ## Returns
        ----------
        anomaly_threshold : float
            The anomaly threshold.

        ## Raises
        ---------
        ValueError
            The attribute `__anomaly_threshold` is stil None.
        """
        if not isinstance(self.__anomaly_threshold, float):
            ValueError("Invalid value! `__anomaly_threshold` attribute is still None. Try calling `detect()`")
        return self.__anomaly_threshold

    @property
    def fit_result(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the dataset with all the anomaly scores.

        ## Returns
        ----------
        anomaly_scores : typing.Union[pd.DataFrame, pd.Series]
            A Pandas DataFrame or Series that contains the anomlay scores.

        ## Raises
        ---------
        ValueError
            The attribute `__anomaly_score` is neither a Pandas DataFrame, nor a Pandas Series.
        """
        if not isinstance(self.__fit_result, pd.DataFrame) and not isinstance(self.__fit_result, pd.Series):
            raise ValueError("Invalid value! `__anomaly_score` attribute is still None. Try calling `fit()`")
        return self.__fit_result

    @property
    def detection_result(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the dataset with all detected detected.

        ## Returns
        ----------
        detected_data : typing.Union[pd.DataFrame, pd.Series]
            A Pandas DataFrame or Series that contains boolean result of whether the data is anomalous or not.

        ## Raises
        ---------
        ValueError
            The attribute `__detection` is neither a Pandas DataFrame, nor a Pandas Series.
        """
        if not isinstance(self.__detection, pd.DataFrame) and not isinstance(self.__detection, pd.Series):
            raise ValueError("Invalid value! `__detection` attribute is still None. Try calling `detect()`")
        return self.__detection

    @property
    def detected_anomalies(self) -> typing.Union[pd.DataFrame, pd.Series]:
        raise NotImplementedError("Not yet implemented!")

    @property
    def evaluation_result(self) -> pd.DataFrame:
        """
        Return the evaluation result.

        ## Returns
        ----------
        evaluation_result : pandas.DataFrame
            A Pandas DataFrame that contains the Kolmogorov Smirnov test result stored in `__eval`.

        ## Raises
        ---------
        ValueError
            The attribute `__eval` is still None because `evaluate()` has not been called.
        """
        return self.__eval

    @property
    def detection_summary(self) -> pd.DataFrame:
        raise NotImplementedError("Not yet implemented!")

    @property
    def params(self) -> dict:  # type: ignore
        return self.__params

    def __str__(self) -> str:
        return "DBSCAN"
