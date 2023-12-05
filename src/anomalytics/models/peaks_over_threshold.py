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
    __params: typing.Dict

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
            t0_pct=0.7,
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

    def reset_time_window(
        self,
        analysis_type: typing.Literal["historical", "real-time"] = "historical",
        t0_pct: float = 0.65,
        t1_pct: float = 0.25,
        t2_pct: float = 0.10,
    ) -> None:
        self.__time_window = set_time_window(
            total_rows=self.__dataset.shape[0],
            method="POT",
            analysis_type=analysis_type,
            t0_pct=t0_pct,
            t1_pct=t1_pct,
            t2_pct=t2_pct,
        )

    def get_extremes(self, q: float = 0.90) -> None:
        if isinstance(self.__dataset, pd.DataFrame):
            pass

        self.__exceedance_threshold = get_threshold_peaks_over_threshold(
            ts=self.__dataset, t0=self.__time_window[0], anomaly_type=self.__anomaly_type, q=q
        )
        self.__exceedance = get_exceedance_peaks_over_threshold(
            ts=self.__dataset, t0=self.__time_window[0], anomaly_type=self.__anomaly_type, q=q
        )

    def fit(self) -> None:
        if isinstance(self.__dataset, pd.DataFrame):
            pass

        self.__anomaly_score = get_anomaly_score(
            ts=self.__exceedance, t0=self.__time_window[0], gpd_params=self.__params
        )

    def detect(self, q: float = 0.90) -> None:
        if isinstance(self.__dataset, pd.DataFrame):
            pass

        self.__anomaly_threshold = get_anomaly_threshold(ts=self.__anomaly_score, t1=self.__time_window[1], q=q)
        self.__anomaly = get_anomaly(ts=self.__anomaly_score, t1=self.__time_window[1], q=q)

    def evaluate(self, method: typing.Literal["ks", "qq"] = "ks", is_random_param: bool = False) -> None:
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
    def params(self) -> dict:  # type: ignore
        return self.__params

    def return_dataset(
        self,
        set_type: typing.Literal[
            "exceedance_threshold", "exceedance", "anomaly", "anomaly_threshold", "anomaly_score", "eval"
        ],
    ) -> typing.Union[pd.DataFrame, pd.Series, float]:
        if set_type == "exceedance_threshold":
            dataset = self.__exceedance_threshold
        elif set_type == "exceedance":
            dataset = self.__exceedance
        elif set_type == "anomaly_score":
            dataset = self.__anomaly_score
        elif set_type == "anomaly_threshold":
            dataset = self.__anomaly_threshold
        elif set_type == "anomaly":
            dataset = self.__anomaly
        elif set_type == "eval":
            dataset = self.__eval
        else:
            raise ValueError(
                "Invalid value! Available `set_type` values: 'exceedance_threshold', 'exceedance', 'anomaly', 'anomaly_threshold', 'anomaly_score'"
            )
        return dataset

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
            plot_line(
                dataset=self.__exceedance,
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
                dataset=self.__exceedance,
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
            plot_gen_pareto(
                dataset=self.__exceedance,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                bins=bins,
                plot_width=plot_width,
                plot_height=plot_height,
                plot_color=plot_color,
                alpha=alpha,
                params=self.__params,
            )

    def __str__(self) -> str:
        return "POT"
