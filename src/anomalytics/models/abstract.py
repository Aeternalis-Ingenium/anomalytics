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
    def evaluate(self) -> None:
        """
        Evaluate the performance of the anomaly detection model based on true and predicted labels.
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

    @abc.abstractmethod
    def set_params(self, **kwargs: typing.Union[str, int, float, None]) -> None:
        """
        Set the parameters for the anomaly detection model.

        ## Parameters
        -------------
        **kwargs : typing.Union[str, int, float, None]
            Store fitting parameters.
        """
        ...