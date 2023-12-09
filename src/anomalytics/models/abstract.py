import abc
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
        method : method: typing.Literal["ks", "qq"], default "ks"
            A parameter that decide what statistical method to use for testing the analysis result.
            * "ks" for Kolmogorov Smirnov
            * "qq" for QQ Plot
        """
        ...

    @property
    @abc.abstractmethod
    def return_detected_anomalies(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Retrieve all detected anomalies.

        ## Returns
        ----------
        detected_anomalies : typing.Union[pd.DataFrame, pd.Series]
            All detected anomalies.
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
