import logging
import typing

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_pot_threshold(
    ts: pd.Series, t0: int, anomaly_type: typing.Literal["high", "low"] = "high", q: float = 0.90
) -> pd.Series:
    """
    Calculate the threshold value that will be used to extract the exceedances from the original dataset.

    ## Parameters
    -------------
    ts : pandas.Series
        The dataset with 1 feature and datetime index to calculate the threshold values.

    t0 : int
        The time window used to define the dynamic expanding window to calculate the quantile score for each row.

    q : float
        The quantile to use for thresholding, default 0.90.

    ## Returns
    ----------
    pot_thresholds : pandas.Series:
        A Pandas Series where each value is a threshold to extract the exceedances from the original dataset.
    """
    logger.debug(
        f"calculating dynamic threshold for exceedance extraction using anomaly_type={anomaly_type}, t0={t0}, q={q}"
    )

    if anomaly_type not in ["high", "low"]:
        raise ValueError(f"Invalid value! The `anomaly_type` argument must be 'high' or 'low'")
    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")
    if t0 is None:
        raise ValueError("Invalid value! The `t0` argument must be an integer")
    if anomaly_type == "low":
        q = 1.0 - q

    logger.debug(f"successfully calculating threshold for {anomaly_type} anomaly type")

    return ts.expanding(min_periods=t0).quantile(q=q).bfill()


def get_peaks_over_threshold(ts: pd.Series, q: float = 0.97) -> pd.Series:
    return pd.Series()
