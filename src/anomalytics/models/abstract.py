import abc
import datetime
import typing

import pandas as pd


class Detector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(
        self, dataset: typing.Union[pd.DataFrame, pd.Series], anomaly_type: typing.Literal["high", "low"] = "high"
    ):
        """
        Initialize the anomaly detection model with a specific statisticail method.

        ## Parameters
        ----------
        dataset : typing.Union[pandas.DataFrame, pandas.Series]
            DataFame or Series objects to be analyzed.
            Index must be date-time and values must be numeric.

        anomaly_type : typing.Literal["high", "low"]
            Defining which kind of anomaly are we expecting.
        """
        ...

    @abc.abstractmethod
    def fit(self) -> None:
        """
        Train the anomaly detection model using the provided data.
        """
        ...

    @abc.abstractmethod
    def detect(self) -> None:
        """
        Detect anomalies in the dataset.
        """
        ...

    @abc.abstractmethod
    def evaluate(self, method: typing.Literal["ks", "qq"] = "ks") -> None:
        """
        Evaluate the performance of the anomaly detection model based on true and predicted labels.

        ## Parameters
        -------------
        method: typing.Literal["ks", "qq"], default "ks"
            A parameter that decide what statistical method to use for testing the analysis result.
            * "ks" for Kolmogorov Smirnov
            * "qq" for QQ Plot
        """
        ...

    @property
    @abc.abstractmethod
    def fit_result(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the fitting result.

        ## Returns
        ----------
        fit_result : typing.Union[pd.DataFrame, pd.Series]
            The fitting result in a Pandas DataFrame or Series.
        """
        ...

    @property
    @abc.abstractmethod
    def detection_result(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the result of the detection method.

        ## Returns
        ----------
        detected_data : typing.Union[pd.DataFrame, pd.Series]
            The detected data in a Pandas DataFrame or Series.
        """
        ...

    @property
    @abc.abstractmethod
    def anomaly_threshold(self) -> float:
        """
        Return the anomaly threshold.

        ## Returns
        ----------
        anomaly_threshold : float
            The anomaly threshold.
        """
        ...

    @property
    @abc.abstractmethod
    def detected_anomalies(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the anomalous data in the original dataset.

        ## Returns
        ----------
        detected_anomalies : typing.Union[pd.DataFrame, pd.Series]
            All anomalous data.
        """
        ...

    @property
    @abc.abstractmethod
    def evaluation_result(self) -> pd.DataFrame:
        """
        Return the evaluation result in a Pandas DataFrame.

        ## Returns
        ----------
        evaluation_result : pandas.DataFrame
            The evaluation result from the analysis.
        """
        ...

    @property
    @abc.abstractmethod
    def detection_summary(self) -> pd.DataFrame:
        """
        Retrieve the summary of the detected data.

        ## Returns
        ----------
        detection_summary : pd.DataFrame
            A DataFrame that contains the row, column, date, anomalous data, anomaly score, and anomaly threshold from the analysis.
        """
        ...

    @property
    @abc.abstractmethod
    def params(self) -> typing.Dict:
        """
        Retrieve the parameters of the anomaly detection model.

        ## Returns
        ----------
        parameters : typing.Dict
            The fitting result from the model.
        """
        ...
