import datetime
import logging
import typing
import warnings

import numpy as np
import pandas as pd

from anomalytics.evals.kolmogorv_smirnov import ks_1sample
from anomalytics.evals.qq_plot import visualize_qq_plot
from anomalytics.models.abstract import Detector
from anomalytics.plots.plot import plot_gen_pareto, plot_hist, plot_line
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
        "__dataset",
        "__time_window",
        "__anomaly_type",
        "__exceedance_threshold",
        "__exceedance",
        "__anomaly_score",
        "__anomaly_threshold",
        "__detection",
        "__eval",
        "__params",
    ]

    __anomaly_type: typing.Literal["high", "low"]
    __dataset: typing.Union[pd.DataFrame, pd.Series]
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

        dataset = dataset.copy(deep=True)

        if isinstance(dataset, pd.Series):
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

        self.__anomaly_type = anomaly_type
        self.__dataset = dataset
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
        if not isinstance(self.__anomaly_threshold, float):
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
            pass

        elif isinstance(self.__dataset, pd.Series):
            try:
                detected_anomalies = self.__detection[self.__detection.values == True]
            except TypeError:
                raise TypeError("Invalid type! The `__detection` attribute is still None. Try calling `detect()`")
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
        if isinstance(self.__dataset, pd.DataFrame):
            pass

        self.__exceedance_threshold = get_threshold_peaks_over_threshold(
            ts=self.__dataset, t0=self.__time_window[0], anomaly_type=self.__anomaly_type, q=q
        )
        self.__exceedance = get_exceedance_peaks_over_threshold(
            ts=self.__dataset, t0=self.__time_window[0], anomaly_type=self.__anomaly_type, q=q
        )

    def fit(self) -> None:
        """
        Fit the exceedances into GPD and use the parameters to compute the anomaly scores.
        """
        if isinstance(self.__dataset, pd.DataFrame):
            pass

        self.__anomaly_score = get_anomaly_score(
            ts=self.__exceedance, t0=self.__time_window[0], gpd_params=self.__params
        )

    def detect(self, q: float = 0.90) -> None:
        """
        Locate the anomalies within the dataset by comparing the anomaly scores against the anomaly threshold

        ## Parameters
        -------------
        q : float, default is 0.90
            The quantile used to calculate the anomaly threshold to detect the anomalies.
        """
        if isinstance(self.__dataset, pd.DataFrame):
            pass

        self.__anomaly_threshold = get_anomaly_threshold(ts=self.__anomaly_score, t1=self.__time_window[1], q=q)
        self.__detection = get_anomaly(ts=self.__anomaly_score, t1=self.__time_window[1], q=q)

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
        params = self.__get_nonzero_params
        if method == "ks":
            self.__eval = pd.DataFrame(data=ks_1sample(ts=self.__exceedance, stats_method="POT", fit_params=params))
            assert isinstance(self.__eval, pd.DataFrame)
        else:
            visualize_qq_plot(
                ts=self.__exceedance, stats_method="POT", fit_params=params, is_random_param=is_random_param
            )

    @property
    def __get_nonzero_params(self) -> typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]]:
        """
        Filter and return only GPD params where there are at least 1 parameter that is greater than 0.

        ## Returns
        ----------
        parameters : typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]]
            A list of all parameters stored in __params that are greater than 0.
        """
        if self.__time_window[0] is None:
            raise ValueError("Invalid value! `t1` is not set?")

        if len(self.params) == 0:
            raise ValueError("`__params` is still empty. Need to call `fit()` first!")

        nonzero_params = []
        for row in range(0, self.__time_window[1] + self.__time_window[2]):  # type: ignore
            if (
                self.params[row]["c"] != 0  # type: ignore
                or self.params[row]["loc"] != 0  # type: ignore
                or self.params[row]["scale"] != 0  # type: ignore
            ):
                nonzero_params.append(self.params[row])
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
            #! TODO: Write logic to get all anomalous data, anomaly scores, index, temporal features, and anomaly threshold.
            pass
        elif isinstance(self.__dataset, pd.Series):
            data = dict(
                row=[self.__dataset.index.get_loc(index) + 1 for index in detected_anomalies.index],
                datetime=[index for index in detected_anomalies.index],
                anomalous_data=[data for data in detected_anomalies.values],
                anomaly_score=[score for score in self.__anomaly_score[detected_anomalies.index].values],
                anomaly_threshold=[self.__anomaly_threshold] * detected_anomalies.shape[0],
            )
        return pd.DataFrame(data=data)

    def plot(
        self,
        plot_type: typing.Literal["l", "l+eth", "l+ath", "hist", "gpd", "gpd+ov"],
        title: str,
        xlabel: str,
        ylabel: str,
        bins: typing.Optional[int] = 50,
        plot_width: int = 13,
        plot_height: int = 8,
        plot_color: str = "black",
        th_color: str = "red",
        th_type: str = "dashed",
        th_line_width: int = 2,
        alpha: float = 0.8,
    ):
        if isinstance(self.__exceedance, pd.Series):
            nonzero_exceedences = [exceedence for exceedence in self.__exceedance.values if exceedence > 0]
        if plot_type == "l":
            plot_line(
                dataset=self.__dataset,
                threshold=None,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                is_threshold=False,
                plot_width=plot_width,
                plot_height=plot_height,
                plot_color=plot_color,
                th_color=th_color,
                th_type=th_type,
                th_line_width=th_line_width,
                alpha=alpha,
            )
        elif plot_type == "l+ath":
            if isinstance(self.__anomaly_score, pd.Series):
                nonzero_anomaly_scores = self.__anomaly_score[self.__anomaly_score.values > 0]
            plot_line(
                dataset=nonzero_anomaly_scores,
                threshold=self.__anomaly_threshold,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                is_threshold=True,
                plot_width=plot_width,
                plot_height=plot_height,
                plot_color=plot_color,
                th_color=th_color,
                th_type=th_type,
                th_line_width=th_line_width,
                alpha=alpha,
            )
        elif plot_type == "l+eth":
            plot_line(
                dataset=self.__dataset,
                threshold=self.__exceedance_threshold,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                is_threshold=True,
                plot_width=plot_width,
                plot_height=plot_height,
                plot_color=plot_color,
                th_color=th_color,
                th_type=th_type,
                th_line_width=th_line_width,
                alpha=alpha,
            )
        elif plot_type == "hist":
            plot_hist(
                dataset=self.__dataset,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                bins=bins,
                plot_width=plot_width,
                plot_height=plot_height,
                plot_color=plot_color,
                alpha=alpha,
            )
        elif plot_type == "gpd":
            plot_gen_pareto(
                dataset=nonzero_exceedences,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                bins=bins,
                plot_width=plot_width,
                plot_height=plot_height,
                plot_color=plot_color,
                alpha=alpha,
                params=None,
            )
        elif plot_type == "gpd+ov":
            last_nonzero_params = self.__get_nonzero_params[-1]
            plot_gen_pareto(
                dataset=nonzero_exceedences,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                bins=bins,
                plot_width=plot_width,
                plot_height=plot_height,
                plot_color=plot_color,
                alpha=alpha,
                params=last_nonzero_params,
            )

    def __str__(self) -> str:
        return "POT"
