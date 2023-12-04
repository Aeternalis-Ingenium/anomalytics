import datetime
import logging
import typing

import numpy as np
import pandas as pd
import scipy.stats as stats

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
    logger.debug(f"extracting exceedances from dynamic threshold using anomaly_type={anomaly_type}, t0={t0}, q={q}")
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
    logger.debug(f"successfully extracting exceedances from dynamic threshold for {anomaly_type} anomaly type")
    return pd.Series(index=ts.index, data=pot_exceedances, name="exceedances")


def fit_exceedance(ts: pd.Series, t0: int, gpd_params: typing.Dict) -> pd.Series:
    """
    Fit exceedances into generalized pareto distribution to calculate the anomaly score.

    Anomaly Score = 1 / (1 - CDF(exceedance, c, loc, scale))

    ## Parameters
    -------------
    ts : pandas.Series
        The Pandas Series that contains the exceedances.

    t0 : int
        Time window to get the first day of t1 time window for dynamic window fitting.

    gpd_params : dictionary
        A dictionary used as the storage of the GPD parameters (fitting result).

    ## Returns
    ----------
    anomaly_scores : pandas.Series
        A Pandas Series with anomaly scores (inverted p-value) as its values.
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")
    if t0 is None:
        raise ValueError("Invalid value! The `t0` argument must be an integer")

    anomaly_scores = []
    t1_t2_exceedances = ts.iloc[t0:]

    for row in range(0, t1_t2_exceedances.shape[0]):
        fit_exceedances = ts.iloc[t0 + row:]
        future_exeedance = t1_t2_exceedances.iloc[row]
        nonzero_fit_exceedances = fit_exceedances[fit_exceedances.values > 0.0]
        if future_exeedance > 0:
            if len(nonzero_fit_exceedances.values) > 0:
                (c, loc, scale) = stats.genpareto.fit(data=nonzero_fit_exceedances.values, floc=0)
                p_value = stats.genpareto.sf(x=future_exeedance, c=c, loc=loc, scale=scale)
                inverted_p_value = 1 / p_value if p_value > 0.0 else float("inf")
                gpd_params[row] = dict(
                    index=t1_t2_exceedances.index[row],
                    c=c,
                    loc=loc,
                    scale=scale,
                    p_value=p_value,
                    anomaly_score=inverted_p_value,
                )
                anomaly_scores.append(inverted_p_value)
            else:
                gpd_params[row] = dict(
                    index=t1_t2_exceedances.index[row],
                    c=0.0,
                    loc=0.0,
                    scale=0.0,
                    p_value=0.0,
                    anomaly_score=0.0,
                )
                anomaly_scores.append(0.0)
        else:
            gpd_params[row] = dict(
                index=t1_t2_exceedances.index[row],
                c=0.0,
                loc=0.0,
                scale=0.0,
                p_value=0.0,
                anomaly_score=0.0,
            )
            anomaly_scores.append(0.0)
    return pd.Series(index=ts.index[t0:], data=anomaly_scores, name="anomaly scores")
