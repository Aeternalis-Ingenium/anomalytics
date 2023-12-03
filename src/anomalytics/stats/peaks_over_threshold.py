import logging
import typing

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_threshold_peaks_over_threshold(
    ts: pd.Series,
    t0: int,
    anomaly_type: typing.Literal["high", "low"] = "high",
    q: float = 0.90,
) -> pd.Series:
    """
    Calculate the POT threshold value that will be used to extract the exceedances from `ts` dataset.

    ## Parameters
    -------------
    ts : pandas.Series
        The dataset with 1 feature and datetime index to calculate the quantiles.

    t0 : int
        Time window to find dynamic expanding period for calculating quantile score.

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


def get_exceedance_peaks_over_threshold(
    ts: pd.Series,
    t0: int,
    anomaly_type: typing.Literal["high", "low"] = "high",
    q: float = 0.90,
) -> pd.Series:
    """
    Extract values from the `ts` dataset that exceed the POT threshold values.

    ## Parameters
    -------------
    ts : pandas.Series
        The dataset with 1 feature and datetime index to calculate the quantiles.

    t0 : int
        Time window to find dynamic expanding period for calculating quantile score.

    q : float
        The quantile to use for thresholding, default 0.90.

    ## Returns
    ----------
    exceedances : pandas.Series
        A Pandas Series with values exceeding the POT thresholds.
    """
    if anomaly_type not in ["high", "low"]:
        raise ValueError(f"Invalid value! The `anomaly_type` argument must be 'high' or 'low'")
    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")
    if t0 is None:
        raise ValueError("Invalid value! The `t0` argument must be an integer")

    pot_thresholds = get_threshold_peaks_over_threshold(ts=ts, t0=t0, anomaly_type=anomaly_type, q=q)

    if anomaly_type == "high":
        pot_exceedances = np.maximum(ts - pot_thresholds, 0.0)
    else:
        pot_exceedances = np.where(ts > pot_thresholds, 0.0, np.abs(ts - pot_thresholds))
    return pd.Series(index=ts.index, data=pot_exceedances, name="exceedances")
