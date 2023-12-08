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
    Calculate the Peaks Over Threshold (POT) threshold values for a given time series.

    ## Parameters
    -------------
    ts : pandas.Series
        One feature dataset and a datetime index to calculate the quantiles.

    t0 : int
        Time window to find a dynamic expanding period for calculating the quantile score.

    anomaly_type : typing.Literal["high", "low"], default is "high"
        Type of anomaly to detect - high or low.

    q : float, default is 0.90
        The quantile used for thresholding.

    ## Returns
    ----------
    pd.Series
        A Pandas Series where each value is a threshold to extract the exceedances from the original dataset.

    ## Example
    ----------
    >>> t0, t1, t2 = set_time_window(1000, "POT", "historical", t0_pct=0.7, t1_pct=0.2, t2_pct=0.1)
    >>> pot_threshold_ts = get_threshold_peaks_over_threshold(ts, t0, "high", 0.95)
    >>> pot_threshold_ts.tail()
    Date-Time
    2020-03-31 19:00:00    0.867
    2020-03-31 20:00:00    0.867
    2020-03-31 21:00:00    0.867
    2020-03-31 22:00:00    0.867
    2020-03-31 23:00:00    0.867
    Name: Example Dataset, dtype: float64

    ## Raises
    ---------
    ValueError
        If the `anomaly_type` argument is not 'high' or 'low'.
    TypeError
        If the `ts` argument is not a Pandas Series.
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
    Extract values from the time series dataset that exceed the POT threshold values.

    ## Parameters
    -------------
    ts : pandas.Series
        The dataset with one feature and a datetime index.

    t0 : int
        Time window to find a dynamic expanding period for calculating the quantile score.

    anomaly_type : typing.Literal["high", "low"], default is "high"
        Type of anomaly to detect - high or low.

    q : float, default is 0.90
        The quantile used for thresholding.

    ## Returns
    ----------
    pd.Series
        A Pandas Series with values exceeding the POT thresholds.

    ## Example
    ----------
    >>> t0, t1, t2 = set_time_window(1000, "POT", "historical", t0_pct=0.7, t1_pct=0.2, t2_pct=0.1)
    >>> exceedance_ts = get_exceedance_peaks_over_threshold(ts, t0, "high", 0.95)
    >>> exceedance_ts.tail()
    Date-Time
    2020-03-31 19:00:00    0.867
    2020-03-31 20:00:00    0.867
    2020-03-31 21:00:00    0.867
    2020-03-31 22:00:00    0.867
    2020-03-31 23:00:00    0.867
    Name: Example Dataset, dtype: float64

    ## Raises
    ---------
    ValueError
        If the `anomaly_type` argument is not 'high' or 'low'.
    TypeError
        If the `ts` argument is not a Pandas Series.
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


def get_anomaly_score(ts: pd.Series, t0: int, gpd_params: typing.Dict) -> pd.Series:
    """
    Calculate the anomaly score for each data point in a time series based on the Generalized Pareto Distribution (GPD).

    Anomaly Score = 1 / (1 - CDF(exceedance, c, loc, scale))

    ## Parameters
    -------------
    ts : pandas.Series
        The Pandas Series that contains the exceedances.

    t0 : int
        Time window to get the first day of the T1 time window for dynamic window fitting.

    gpd_params : dict
        A dictionary used as the storage of the GPD parameters (fitting result).

    ## Returns
    ----------
    pd.Series
        A Pandas Series with anomaly scores (inverted p-value) as its values.

    ## Example
    ----------
    >>> t0, t1, t2 = set_time_window(1000, "POT", "historical", t0_pct=0.7, t1_pct=0.2, t2_pct=0.1)
    >>> params = {}
    >>> anomaly_score_ts = get_anomaly_score(exceedance_ts, t0, params)
    >>> anomaly_score_ts.head()
    Date-Time
    2016-10-29 00:00:00    0.0
    2016-10-29 01:00:00    0.0
    2016-10-29 02:00:00    0.0
    2016-10-29 03:00:00    0.0
    2016-10-29 04:00:00    0.0
    Name: Example Dataset, dtype: float64
    ...
    >>> params
    {0: {'datetime': Timestamp('2016-10-29 03:00:00'),
    'c': 0.0,
    'loc': 0.0,
    'scale': 0.0,
    'p_value': 0.0,
    'anomaly_score': 0.0},
    1: {'datetime': Timestamp('2016-10-29 04:00:00'),
    ...
    'loc': 0,
    'scale': 0.19125308567629334,
    'p_value': 0.19286132173263668,
    'anomaly_score': 5.1850728337654886},
    ...}

    ## Raises
    ---------
    TypeError
        If the `ts` argument is not a Pandas Series.
    """

    logger.debug(
        f"calculating anomaly score using t0={t0}, scipy.stats.genpareto.fit(), and scipy.stats.genpareto.sf()"
    )

    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")
    if t0 is None:
        raise ValueError("Invalid value! The `t0` argument must be an integer")

    anomaly_scores = []
    t1_t2_exceedances = ts.iloc[t0:]

    for row in range(0, t1_t2_exceedances.shape[0]):
        fit_exceedances = ts.iloc[t0 + row :]
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

    logger.debug(f"successfully calculating anomaly score")

    return pd.Series(index=ts.index[t0:], data=anomaly_scores, name="anomaly scores")


def get_anomaly_threshold(ts: pd.Series, t1: int, q: float = 0.90) -> float:
    """
    Calculate a dynamic threshold based on quantiles used for comparing anomaly scores.

    ## Parameters
    -------------
    ts : pandas.Series
        The Pandas Series that contains the anomaly scores.

    t1 : int
        Time window to calculate the quantile score of all anomaly scores.

    q : float, default is 0.90
        The quantile used for thresholding.

    ## Returns
    ----------
    float
        A single float value serving as the threshold for anomalous data.

    ## Example
    ----------
    >>> t0, t1, t2 = set_time_window(1000, "POT", "historical", t0_pct=0.7, t1_pct=0.2, t2_pct=0.1)
    >>> anomaly_threshold = get_anomaly_threshold(anomaly_score_ts, t1, 0.90)
    >>> anomaly_threshold
    9.167442809714414

    ## Raises
    ---------
    TypeError
        If the `ts` argument is not a Pandas Series.
    """

    logger.debug(f"calculating anomaly threshold using t1={t1}, q={q}, and `numpy.quantile()` function")

    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")

    t1_anomaly_scores = ts[(ts.values > 0) & (ts.values != float("inf"))].iloc[:t1]

    logger.debug(f"successfully calculating anomaly threshold using {q} quantile")

    return np.quantile(
        a=t1_anomaly_scores.values,
        q=q,
    )


def get_anomaly(ts: pd.Series, t1: int, q: float = 0.90) -> pd.Series:
    """
    Detect anomalous data points by comparing anomaly scores with the anomaly threshold.

    ## Parameters
    -------------
    ts : pandas.Series
        The Pandas Series that contains the anomaly scores.

    t1 : int
        Time window to calculate the anomaly threshold and retrieve T2 anomaly scores.

    q : float, default is 0.90
        The quantile used for thresholding.

    ## Returns
    ----------
    pd.Series
        A Pandas Series indicating which values are anomalous.

    ## Example
    ----------
    >>> t0, t1, t2 = set_time_window(1000, "POT", "historical", t0_pct=0.7, t1_pct=0.2, t2_pct=0.1)
    >>> anomaly_ts = get_anomaly(anomaly_score_ts, t1, 0.90)
    >>> anomaly_ts.head()
    Date-Time
    2019-02-09 08:00:00    False
    2019-02-09 09:00:00    False
    2019-02-09 10:00:00    False
    2019-02-09 11:00:00    False
    2019-02-09 12:00:00    False
    Name: Example Dataset, dtype: bool

    ## Raises
    -------
    TypeError
        If the `ts` argument is not a Pandas Series.
    """

    logger.debug(f"detecting anomaly using t1={t1}, q={q}, and `get_anoamly_threshold()` function")

    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")

    anomaly_threshold = get_anomaly_threshold(ts=ts, t1=t1, q=q)
    t2_anomaly_scores = ts.iloc[t1:]
    anomalies = t2_anomaly_scores > anomaly_threshold

    logger.debug(
        f"successfully detecting {len(anomalies[anomalies.values == True].values)} anomalies using anomaly_threshold={anomaly_threshold}"
    )
    return pd.Series(index=t2_anomaly_scores.index, data=anomalies.values, name="anomalies")
