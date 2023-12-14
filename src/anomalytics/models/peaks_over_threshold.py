import datetime
import logging
import typing
import warnings

import numpy as np
import pandas as pd

from anomalytics.evals.kolmogorv_smirnov import ks_1sample
from anomalytics.evals.qq_plot import visualize_qq_plot
from anomalytics.models.abstract import Detector
from anomalytics.plots.plot import visualize
from anomalytics.stats.peaks_over_threshold import (
    get_anomaly,
    get_anomaly_score,
    get_anomaly_threshold,
    get_exceedance_peaks_over_threshold,
    get_threshold_peaks_over_threshold,
)
from anomalytics.time_windows.time_window import set_time_window

logger = logging.getLogger(__name__)


class POTDetector(Detector):
    """_summary_

    ## Attributes
    -------------
    __anomaly_type : __anomaly_type: typing.Literal["high", "low"]
        The dataset that wants to be analyzed.

    __dataset : typing.Union[pandas.DataFrame, pandas.Series]
        The dataset that wants to be analyzed.

    __time_window : typing.Tuple[int, int, int]
        The dataset that wants to be analyzed.

    __exceedance_threshold : typing.Union[pandas.DataFrame, pandas.Series]
        The dataset that contains threshold for exceedances.

    __exceedance : typing.Union[pandas.DataFrame, pandas.Series]
        The dataset that contains the exceedances.

    __anomaly_score : typing.Union[pandas.DataFrame, pandas.Series]
        The dataset that contains the anomaly scores.

    __anomaly_threshold : float
        The anomaly threshold used to detect anomalies.

    __anomaly : typing.Union[pandas.DataFrame, pandas.Series]
        The dataset that contains the detected anomalies.

    __eval : pandas.DataFrame
        The result of "Kolmogorov Smirnov" test presented in a pandas.DataFrame.

    __params
        The GPD parameters resulted from GPD fitting method.

    ## Methods
    ----------
    * __init__(dataset: typing.Union[pd.DataFrame, pd.Series], anomaly_type: typing.Literal["high", "low"] = "high")
    * reset_time_window(analysis_type: typing.Literal["historical", "real-time"] = "historical", t0_pct: float = 0.65, t1_pct: float = 0.25,t2_pct: float = 0.10)
    * t1
    * t2
    * t3
    * get_extremes(q: float = 0.90)
    * fit()
    * detect(q: float = 0.90)
    * evaluate(method: typing.Literal["ks", "qq"] = "ks", is_random_param: bool = False)
    * __get_nonzero_params
    * params
    * anomaly_threshold
    * return_dataset(set_type: typing.Literal["exceedance_threshold", "exceedance", "anomaly", "anomaly_score", "eval"])
    * plot(plot_type: typing.Literal["l", "l+eth", "l+ath", "hist", "gpd", "gpd+ov"], title: str, xlabel: str, ylabel: str, bins: typing.Optional[int] = 50, plot_width: int = 13, plot_height: int = 8, plot_color: str = "black", th_color: str = "red", th_type: str = "dashed", th_line_width: int = 2, alpha: float = 0.8)

    ## Example
    ----------
    >>> import anomalytics as atics
    >>> ts = atics.read_ts("./my_dataset.csv", "csv")
    >>> pot_detector = atics.get_detector("POT", ts, "high")
    >>> print("T0 time window:", pot_detector.t0)
    >>> print("T1 time window:", pot_detector.t1)
    >>> print("T2 time window:", pot_detector.t2)
    ...
    >>> pot_detector.get_extremes(0.97) # Extract exceedances
    >>> pot_detector.fit()              # Fit the exceedances into GPD
    >>> pot_detector.detect(0.97)       # Detect anomalies
    >>> pot_detector.evaluate("ks")     # Evaluate result with Kolmogorov Smirnov
    ...
    >>> exceedance_threshold_ts = pot_detector.return_dataset("exceedance_threshold")
    >>> exceedance_threshold_ts.head() # Exceedance threshold pandas.Series
    ...
    >>> exceedance_ts = pot_detector.return_dataset("exceedance")
    >>> exceedance_ts.head()# Exceedances pandas.Series
    ...
    >>> anomaly_score_ts = pot_detector.return_dataset("anomaly_score")
    >>> anomaly_score_ts.head() # Anomaly scores pandas.Series
    ...
    >>> anomaly_threshold = pot_detector.anomaly_threshold
    >>> print("Anomaly threshold:", anomaly_score_ts) # float
    ...
    >>> anomaly_ts = pot_detector.return_dataset("anomaly")
    >>> anomaly_ts.head() # Anomalies in pandas.Series
    ...
    >>> kst_result_df = pot_detector.return_dataset("eval")
    >>> kst_result_df.head() # KS parameters in pandas.DataFrame
    ...
    """

    __slots__ = [
        "__anomaly_score",
        "__anomaly_threshold",
        "__anomaly_type",
        "__dataset",
        "__datetime",
        "__detection",
        "__eval",
        "__exceedance",
        "__exceedance_threshold",
        "__params",
        "__time_window",
    ]

    __anomaly_type: typing.Literal["high", "low"]
    __dataset: typing.Union[pd.DataFrame, pd.Series]
    __datetime: typing.Optional[pd.Series]
    __time_window: typing.Tuple[int, int, int]
    __exceedance_threshold: typing.Union[pd.DataFrame, pd.Series]
    __exceedance: typing.Union[pd.DataFrame, pd.Series]
    __anomaly_score: typing.Union[pd.DataFrame, pd.Series]
    __anomaly_threshold: float
    __detection: typing.Union[pd.DataFrame, pd.Series]
    __eval: pd.DataFrame
    __params: typing.Dict

    def __init__(
        self, dataset: typing.Union[pd.DataFrame, pd.Series], anomaly_type: typing.Literal["high", "low"] = "high"
    ):
        """
        Initialize POT model for anomaly detection.

        ## Parameters
        -------------
        dataset : typing.Union[pandas.DataFrame, pandas.Series]
            DataFame or Series objects to be analyzed.
            Index must be date-time and values must be numeric.

        anomaly_type : typing.Literal["high", "low"]
            Defining which kind of anomaly are we expecting.
        """
        logger.info("start initialization of POT detection model")

        if anomaly_type not in ["high", "low"]:
            raise ValueError(f"Invalid value! The `anomaly_type` argument must be 'high' or 'low'")
        if not isinstance(dataset, pd.DataFrame) and not isinstance(dataset, pd.Series):
            raise TypeError("Invalid value! The `dataset` argument must be a Pandas DataFrame or Series")

        dataset = dataset.copy(deep=True)

        if isinstance(dataset, pd.DataFrame):
            datetime64_columns = [
                column for column in dataset.columns if pd.api.types.is_datetime64_any_dtype(dataset[column])
            ]

            if len(datetime64_columns) == 0:
                raise ValueError(
                    "No value! One of your feature must be the Pandas datetime64 data type. Try converting with `pandas.to_datetime()`"
                )
            elif len(datetime64_columns) > 1:
                raise ValueError("Too many values! You are only allowed to have one feature with datetim64 data type.")

            self.__datetime = dataset[datetime64_columns[0]]
            self.__dataset = dataset.drop(columns=[datetime64_columns[0]], axis=1)

        elif isinstance(dataset, pd.Series):
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
            self.__datetime = None
            self.__dataset = dataset

        self.__anomaly_type = anomaly_type
        self.__time_window = set_time_window(
            total_rows=self.__dataset.shape[0],
            method="POT",
            analysis_type="real-time",
            t0_pct=0.7,
            t1_pct=0.3,
            t2_pct=0.0,
        )
        self.__exceedance_threshold = None  # type: ignore
        self.__exceedance = None  # type: ignore
        self.__anomaly_score = None  # type: ignore
        self.__anomaly_threshold = None  # type: ignore
        self.__detection = None  # type: ignore
        self.__eval = None  # type: ignore
        self.__params = {}

        logger.info("successfully initialized POT detection model")

    def reset_time_window(
        self,
        analysis_type: typing.Literal["historical", "real-time"] = "historical",
        t0_pct: float = 0.65,
        t1_pct: float = 0.25,
        t2_pct: float = 0.10,
    ) -> None:
        """
        Set a new time range for `t0`, `t1`, and `t2`.

        ## Parameters
        -------------
        analysis_type : typing.Literal["historical", "real-time"], default is "historical"
            The type of analysis defines the t2 time window.
            * t2 in "historical" will be x percent.
            * t2  in "real-time" will always be 1 row in the DataFrame or Series

        t0_pct : float, default is 0.65
            The time window used to extract the exceedances.

        t1_pct : float, default is 0.25
            The time window used to compute the anomaly threshold.

        t2_pct : float, default is 0.10
            The time window that consists of the time of interest e.g. today.
        """
        self.__time_window = set_time_window(
            total_rows=self.__dataset.shape[0],
            method="POT",
            analysis_type=analysis_type,
            t0_pct=t0_pct,
            t1_pct=t1_pct,
            t2_pct=t2_pct,
        )

    @property
    def t0(self) -> int:
        """
        The time window used to extract the exceedances.a

        ## Returns
        ----------
        t0 : int
            The `t0` time window.
        """
        return self.__time_window[0]

    @property
    def t1(self) -> int:
        """
        The time window used to compute the anomaly threshold.

        ## Returns
        ----------
        t1 : int
            The `t1` time window.
        """
        return self.__time_window[1]

    @property
    def t2(self) -> int:
        """
        The time window that contains the target period for the detection.

        ## Returns
        ----------
        t2 : int
            The `t2` time window.
        """
        return self.__time_window[2]

    @property
    def params(self) -> typing.Dict:
        """
        The generalized pareto distributions parameters.

        ## Returns
        ----------
        params : typing.Dict
            The GPD parameters.

        ## Examples
        -----------
        __dataset : pandas.DataFrame
            ```json
            {
                0: {
                    "col_1": {
                        "c": 0.0,
                        "loc": 0,
                        "scale": 0.0,
                        "p_value": 0.0,
                        "anomaly_score": 0.0,
                    },
                    "col_2": {
                        "c": -1.6148134739114448,
                        "loc": 0,
                        "scale": 9.68888084346867,
                        "p_value": 0.000012345,
                        "anomaly_score": 355.1234,
                    },
                    "total_anomaly_score": 355.1234,
                },
                1: {
                    "col_1": {
                        "c": -2.764709117887601,
                        "loc": 0,
                        "scale": 22.11767294310081,
                        "p_value": 0.0,
                        "anomaly_score": 20.1234,
                    },
                    "col_2": {
                        "c": -5.247337720538409,
                        "loc": 0,
                        "scale": 36.73136404376887,
                        "p_value": 0.000012345,
                        "anomaly_score": 876.1234,
                    },
                    "total_anomaly_score": 896.2468,
                },
            }
            ```

        __dataset : pandas.Series
            ```json
            {
                0: {
                    "index": pd.Timestamp("2023-01-07 00:00:00"),
                    "c": -2.687778724221391,
                    "loc": 0,
                    "scale": 1.3438893621106958,
                    "p_value": 0.0,
                    "anomaly_score": float("inf"),
                },
                1: {
                    "index": pd.Timestamp("2023-01-08 00:00:00"),
                    "c": -1.8725554221391,
                    "loc": 0,
                    "scale": 1.3438893621106958,
                    "p_value": 0.0,
                    "anomaly_score": 127.23451123,
                },
            }
            ```
        """
        return self.__params

    @property
    def exceedance_thresholds(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the dataset with the exceedance thresholds.

        ## Returns
        ----------
        exceedance_thresholds : typing.Union[pd.DataFrame, pd.Series]
            A Pandas DataFrame or Series that contains the thresholds for extracting the exceedances.

        ## Raises
        ---------
        TypeError
            The attribute `__exceedance_threshold` is neither a Pandas DataFrame, nor a Pandas Series.
        """
        if not isinstance(self.__exceedance_threshold, pd.DataFrame) and not isinstance(
            self.__exceedance_threshold, pd.Series
        ):
            raise TypeError(
                "Invalid type! `__exceedance_threshold` attribute is still None. Try calling `get_extremes()`"
            )
        elif isinstance(self.__exceedance_threshold, pd.DataFrame):
            exceedance_threshold = self.__exceedance_threshold.copy(deep=True)
            exceedance_threshold["datetime"] = self.__datetime.values  # type: ignore
            return exceedance_threshold
        return self.__exceedance_threshold

    @property
    def exceedances(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the dataset with the exceedances.

        ## Returns
        ----------
        exceedances: typing.Union[pd.DataFrame, pd.Series]
            A Pandas DataFrame or Series that contains exceedances.

        ## Raises
        ---------
        TypeError
            The attribute `__exceedance` is neither a Pandas DataFrame, nor a Pandas Series.
        """
        if not isinstance(self.__exceedance, pd.DataFrame) and not isinstance(self.__exceedance, pd.Series):
            raise TypeError("Invalid type! `__exceedance` attribute is still None. Try calling `get_extremes()`")
        elif isinstance(self.__exceedance, pd.DataFrame):
            exceedance = self.__exceedance.copy(deep=True)
            exceedance["datetime"] = self.__datetime.values  # type: ignore
            return exceedance
        return self.__exceedance

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
        TypeError
            The attribute `__anomaly_threshold` is stil None.
        """
        if self.__anomaly_threshold is None or not isinstance(self.__anomaly_threshold, float):
            raise TypeError("Invalid value! `__anomaly_threshold` attribute is still None. Try calling `detect()`")
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
        TypeError
            The attribute `__anomaly_score` is neither a Pandas DataFrame, nor a Pandas Series.
        """
        if not isinstance(self.__anomaly_score, pd.DataFrame) and not isinstance(self.__anomaly_score, pd.Series):
            raise TypeError("Invalid type! `__anomaly_score` attribute is still None. Try calling `fit()`")
        elif isinstance(self.__anomaly_score, pd.DataFrame):
            anomaly_score = self.__anomaly_score.copy(deep=True)
            anomaly_score["datetime"] = self.__datetime.values[self.__time_window[0] :]  # type: ignore
            return anomaly_score
        return self.__anomaly_score

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
        TypeError
            The attribute `__detection` is neither a Pandas DataFrame, nor a Pandas Series.
        """
        if not isinstance(self.__detection, pd.DataFrame) and not isinstance(self.__detection, pd.Series):
            raise TypeError("Invalid type! `__detection` attribute is still None. Try calling `detect()`")
        elif isinstance(self.__detection, pd.DataFrame):
            detection = self.__detection.copy(deep=True)
            detection["datetime"] = self.__datetime.values[self.__time_window[0] + self.__time_window[1] :]  # type: ignore
            return detection
        return self.__detection

    @property
    def detected_anomalies(self) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        Return the dataset with all detected anomalies.

        ## Returns
        ----------
        detected_anomalies : typing.Union[pd.DataFrame, pd.Series]
            A Pandas DataFrame or Series that contains all the detected anomalies.

        ## Raises
        ---------
        TypeError
            The attribute `__dataset` is neither a Pandas DataFrame, nor a Pandas Series.
        """
        if isinstance(self.__dataset, pd.DataFrame):
            t2_anomaly_scores = self.__anomaly_score.copy().iloc[self.__time_window[1] :]
            t1t2_dataset = self.__dataset.iloc[self.__time_window[0] :].copy().reset_index(drop=False, names=["row"])
            detected_anomalous_data = t1t2_dataset.iloc[t2_anomaly_scores.index].copy()
            return pd.concat(objs=[detected_anomalous_data, t2_anomaly_scores], axis=1)
        elif isinstance(self.__dataset, pd.Series):
            try:
                detected_anomalies = self.__detection[self.__detection.values == True]
            except Exception as _error:
                raise ValueError(
                    "Invalid type! The `__detection` attribute is still None. Try calling `detect()`"
                ) from _error
            return self.__dataset[detected_anomalies.index]
        raise TypeError("Invalid type! The `__dataset` attribute is neither a Pandas DataFrame, nor a Pandas Series")

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
        TypeError
            The attribute `__eval` is still None because `evaluate()` has not been called.
        """
        if isinstance(self.__eval, pd.DataFrame):
            return self.__eval
        raise TypeError("Invalid type! `__eval` attribute is still None. Try calling `evaluate()`")

    def get_extremes(self, q: float = 0.90) -> None:
        """
        Extract exceedances from the dataset.

        ## Parameters
        -------------
        q : float, default is 0.90
            The quantile used to calculate the exceedance threshold to extract exceedances.
        """
        self.__exceedance_threshold = get_threshold_peaks_over_threshold(
            dataset=self.__dataset, t0=self.__time_window[0], anomaly_type=self.__anomaly_type, q=q
        )
        self.__exceedance = get_exceedance_peaks_over_threshold(
            dataset=self.__dataset,
            threshold_dataset=self.__exceedance_threshold,
            anomaly_type=self.__anomaly_type,
        )

    def fit(self) -> None:
        """
        Fit the exceedances into GPD and use the parameters to compute the anomaly scores.
        """
        self.__anomaly_score = get_anomaly_score(
            exceedance_dataset=self.__exceedance, t0=self.__time_window[0], gpd_params=self.__params
        )

    def detect(self, q: float = 0.90) -> None:
        """
        Locate the anomalies within the dataset by comparing the anomaly scores against the anomaly threshold

        ## Parameters
        -------------
        q : float, default is 0.90
            The quantile used to calculate the anomaly threshold to detect the anomalies.
        """
        self.__anomaly_threshold = get_anomaly_threshold(
            anomaly_score_dataset=self.__anomaly_score, t1=self.__time_window[1], q=q
        )
        self.__detection = get_anomaly(
            anomaly_score_dataset=self.__anomaly_score,
            threshold=self.__anomaly_threshold,
            t1=self.__time_window[1],
        )

    def evaluate(self, method: typing.Literal["ks", "qq"] = "ks", is_random_param: bool = False) -> None:
        """
        Evalute the result of the analysis by comparing the exceeance (observed sample) with the GPD parameters (theoretical).

        ## Parameters
        -------------
        method : typing.Literal["ks", "qq"], default is "ks"
            The statistical method used to test the analysis result.
            * "ks" for Kolmogorov Smirnov test
            * "qq" for QQ Plot (visual observation)

        is_random_param : bool, default is `False`
            The parameter for "qq" evaluation to use either random or the last GPD paarameters.

        ## Returns
        ----------
        qq_plot : None
            If `method` is "qq", then the method will return a QQ plot of sample vs. theoretical quantiles.

        kstest_result : None
            If `method` is "ks", then the method assign a Pandas DataFrame with statistical distance and p-value to `__eval` attribute.
        """
        if not isinstance(self.__exceedance, pd.DataFrame) and not isinstance(self.__exceedance, pd.Series):
            raise TypeError("Invalid Type! `__exceedance` attribute must be a Pandas DataFrame or Series")

        params = self.__get_nonzero_params

        if method == "ks":
            self.__eval = pd.DataFrame(
                data=ks_1sample(dataset=self.__exceedance, stats_method="POT", fit_params=params)  # type: ignore
            )
            assert isinstance(self.__eval, pd.DataFrame)
        else:
            visualize_qq_plot(
                dataset=self.__exceedance,
                stats_method="POT",
                fit_params=params,
                is_random_param=is_random_param,
            )

    @property
    def __get_nonzero_params(
        self,
    ) -> typing.List[
        typing.Dict[
            str, typing.Union[typing.List[typing.Dict[str, typing.Union[float, int]]], typing.Union[float, int]]
        ]
    ]:
        """
        Filter and return only GPD params where there are at least 1 parameter that is greater than 0.

        ## Returns
        ----------
        nonzero_parameters : typing.List[typing.Dict[str, typing.Union[typing.List[typing.Dict[str, typing.Union[float, int]]], typing.Union[float, int]]]]
            A list of all parameters stored in __params that are greater than 0.

        ## Examples
        -----------
        __dataset : pandas.DataFrame
            ```json
            [
                {
                    "col_1": [
                        {"c": -2.020681654255883, "loc": 0, "scale": 10.103408271279417},
                        {"c": -4.216342466354261, "loc": 0, "scale": 25.29805479812557},
                        {"c": -5.247337720538409, "loc": 0, "scale": 36.73136404376887},
                        {"c": -2.764709117887601, "loc": 0, "scale": 22.11767294310081},
                    ]
                },
                {
                    "col_2": [
                        {"c": -1.6148134739114448, "loc": 0, "scale": 9.68888084346867},
                        {"c": -2.4907573384041193, "loc": 0, "scale": 58.28372171865636},
                        {"c": -1.2641494213744446, "loc": 0, "scale": 29.581096460161987},
                    ]
                },
            ]
            ```

        __dataset : pandas.Series
            ```json
            [
                {"c": -2.020681654255883, "loc": 0, "scale": 10.103408271279417},
                {"c": -4.216342466354261, "loc": 0, "scale": 25.29805479812557},
                {"c": -5.247337720538409, "loc": 0, "scale": 36.73136404376887},
                {"c": -2.764709117887601, "loc": 0, "scale": 22.11767294310081},
            ]
            ```
        """
        if self.__time_window[0] is None:
            raise ValueError("Invalid value! `t1` is not set?")

        if len(self.params) == 0:
            raise ValueError("`__params` is still empty. Need to call `fit()` first!")

        nonzero_params: typing.List = []
        t1_t2_time_window = self.__time_window[1] + self.__time_window[2]

        if isinstance(self.__dataset, pd.DataFrame):
            for index, column in enumerate(self.__dataset.columns):
                nonzero_params.append({column: []})
                for row in range(0, t1_t2_time_window):
                    if column != "total_anomaly_score":
                        if (
                            self.__params[row][column]["c"] != 0
                            or self.__params[row][column]["loc"] != 0
                            or self.__params[row][column]["scale"] != 0
                        ):
                            nonzero_params[index][column].append(
                                {
                                    "c": self.__params[row][column]["c"],
                                    "loc": self.__params[row][column]["loc"],
                                    "scale": self.__params[row][column]["scale"],
                                }
                            )

        elif isinstance(self.__dataset, pd.Series):
            for row in range(0, t1_t2_time_window):  # type: ignore
                if (
                    self.__params[row]["c"] != 0  # type: ignore
                    or self.__params[row]["loc"] != 0  # type: ignore
                    or self.__params[row]["scale"] != 0  # type: ignore
                ):
                    nonzero_params.append(self.__params[row])
        return nonzero_params

    @property
    def detection_summary(self) -> pd.DataFrame:
        try:
            detected_anomalies = self.detected_anomalies
        except Exception as _error:
            raise TypeError(
                "Invalid type! The `__detection` attribute is still None. Try calling `detect()`"
            ) from _error

        if isinstance(self.__dataset, pd.DataFrame):
            t1_datetime = self.__datetime.iloc[self.__time_window[0] :].copy().reset_index(drop=False)  # type: ignore
            anomalous_datetime = t1_datetime.iloc[detected_anomalies.index].copy()
            if (anomalous_datetime["index"].values == detected_anomalies["row"].values).all():
                confirmed_anomalous_datetime = anomalous_datetime.drop(columns=["index"], axis=1).values.flatten()
            else:
                raise ValueError("Invalid value! `index` values from datetime and detected anomalies are not the same")

            detected_anomalies["anomaly_threshold"] = [self.__anomaly_threshold] * detected_anomalies.shape[0]
            detected_anomalies.index = confirmed_anomalous_datetime
            detected_anomaly_summary = detected_anomalies[
                detected_anomalies["total_anomaly_score"] > detected_anomalies["anomaly_threshold"]
            ]
            datetime_indices = detected_anomaly_summary.index
            data = detected_anomaly_summary.to_dict("list")

        elif isinstance(self.__dataset, pd.Series):
            datetime_indices = [index for index in detected_anomalies.index]
            data = dict(
                row=[self.__dataset.index.get_loc(index) for index in detected_anomalies.index],
                anomalous_data=[data for data in detected_anomalies.values],
                anomaly_score=[score for score in self.__anomaly_score[detected_anomalies.index].values],
                anomaly_threshold=[self.__anomaly_threshold] * detected_anomalies.shape[0],
            )
        return pd.DataFrame(index=datetime_indices, data=data)

    def __filter_nonzero_df2ts(self, dataset: pd.DataFrame) -> typing.List[pd.Series]:
        return [dataset[dataset[column].values > 0][column].copy() for column in dataset.columns]

    def __filter_nonzero_ts(self, dataset: pd.Series) -> pd.Series:
        return dataset[dataset.values > 0].copy()

    def __df2ts(self, dataset: pd.DataFrame) -> typing.List[pd.Series]:
        return [dataset[column].copy() for column in dataset.columns]

    def __full_df(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = dataset.copy()
        if df.shape[0] == self.__time_window[2]:
            df.index = self.__datetime.copy()[self.__time_window[0] + self.__time_window[1] :].values.flatten()  # type: ignore
        elif df.shape[0] == self.__time_window[1] + self.__time_window[2]:
            df.index = self.__datetime.copy()[self.__time_window[0] :].values.flatten()  # type: ignore
        else:
            df.index = self.__datetime.copy().values.flatten()  # type: ignore
        return df

    def plot(
        self,
        ptype: typing.Literal[
            "hist-dataset-df",
            "hist-dataset-ts",
            "hist-gpd-df",
            "hist-gpd-ts",
            "line-anomaly-score-df",
            "line-anomaly-score-ts",
            "line-dataset-df",
            "line-dataset-ts",
            "line-exceedance-df",
            "line-exceedance-ts",
        ],
        title: str,
        xlabel: str,
        ylabel: str,
        plot_width: int = 15,
        plot_height: int = 10,
        plot_color: str = "black",
        th_color: str = "red",
        th_type: str = "dashed",
        th_line_width: int = 2,
        alpha: float = 0.8,
        bins: typing.Optional[int] = 50,
    ):
        if ptype in ["hist-gpd-df", "hist-gpd-ts"]:
            if len(self.__params) == 0:
                raise ValueError("Invalid value! `__params` attribute is still None. Try calling `fit()` first")

        elif ptype in ["line-anomaly-score-df", "line-anomaly-score-ts"]:
            if self.__anomaly_threshold is None:
                raise ValueError(
                    "Invalid value! `__anomaly_threshold` attribute is still None. Try calling `detect()` first"
                )

        if isinstance(self.__dataset, pd.DataFrame):
            if ptype == "hist-dataset-df":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    columns=self.__dataset.columns,
                    datasets=self.__df2ts(dataset=self.__dataset),
                    bins=bins,  # type: ignore
                )
            elif ptype == "hist-gpd-df":
                nonzero_params = self.__get_nonzero_params
                last_nonzero_params = []

                for index, column in enumerate(self.__exceedance.columns):
                    last_nonzero_params.append(nonzero_params[index][column][-1])  # type: ignore

                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    columns=self.__exceedance.columns,
                    datasets=self.__filter_nonzero_df2ts(dataset=self.__exceedance),
                    params=last_nonzero_params,  # type: ignore
                    bins=bins,  # type: ignore
                )
            elif ptype == "line-anomaly-score-df":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    columns=self.__anomaly_score.columns,
                    datasets=self.__filter_nonzero_df2ts(dataset=self.__full_df(dataset=self.__anomaly_score)),
                    threshold=self.__anomaly_threshold,
                    th_color=th_color,
                    th_type=th_type,
                    th_line_width=th_line_width,
                )
            elif ptype == "line-dataset-df":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    columns=self.__dataset.columns,
                    datasets=self.__df2ts(dataset=self.__full_df(dataset=self.__dataset)),
                )
            elif ptype == "line-exceedance-df":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    columns=self.__dataset.columns,
                    datasets=self.__df2ts(dataset=self.__full_df(dataset=self.__dataset)),
                    thresholds=self.__df2ts(dataset=self.exceedance_thresholds),
                    th_color=th_color,
                    th_type=th_type,
                    th_line_width=th_line_width,
                )
        elif isinstance(self.__dataset, pd.Series):
            if ptype == "hist-dataset-ts":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    dataset=self.__dataset,
                    bins=bins,  # type: ignore
                )
            elif ptype == "hist-gpd-ts":
                last_nonzero_params = self.__get_nonzero_params[-1]  # type: ignore

                return visualize(  # type: ignore
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    dataset=self.__filter_nonzero_ts(dataset=self.__exceedance),
                    params=last_nonzero_params,
                    bins=bins,
                )
            elif ptype == "line-anomaly-score-ts":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    dataset=self.__filter_nonzero_ts(dataset=self.__anomaly_score),
                    threshold=self.__anomaly_threshold,
                    th_color=th_color,
                    th_type=th_type,
                    th_line_width=th_line_width,
                )
            elif ptype == "line-dataset-ts":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    dataset=self.__dataset,
                )
            elif ptype == "line-exceedance-ts":
                return visualize(
                    plot_type=ptype,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    plot_color=plot_color,
                    alpha=alpha,
                    dataset=self.__dataset,
                    threshold=self.__exceedance_threshold,
                    th_color=th_color,
                    th_type=th_type,
                    th_line_width=th_line_width,
                )
        else:
            raise TypeError("Invalid type! `dataset` must be a Pandas DataFrame or Series")

    def __str__(self) -> str:
        return "POT"
