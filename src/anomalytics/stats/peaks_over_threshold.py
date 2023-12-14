import logging
import typing

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


def get_threshold_peaks_over_threshold(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    t0: int,
    anomaly_type: typing.Literal["high", "low"] = "high",
    q: float = 0.90,
) -> typing.Union[pd.DataFrame, pd.Series]:
    """
    Calculate the Peaks Over Threshold (POT) threshold values for a given time series.

    ## Parameters
    -------------
    dataset : typing.Union[pd.DataFrame, pd.Series]
        A dataset that must be either Pandas DataFrame or Series for threshold computation.
        * Pandas: One feature needs to be the temporal feature.
        * Series: The index needs to be Pandas DatetimeIndex.

    t0 : int
        Time window to find a dynamic expanding period for calculating the quantile score.

    anomaly_type : typing.Literal["high", "low"], default is "high"
        Type of anomaly to detect - high or low.

    q : float, default is 0.90
        The quantile used for thresholding.

    ## Returns
    ----------
    exceedance_threshold : typing.Union[pd.DataFrame, pd.Series]
        A Pandas DataFrame or Series where each value is a threshold to extract the exceedances from the original dataset.

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
        If the `ts` argument is neither a Pandas DataFrame nor Series.
    ValueError
        If `t0` is not an integer.
    """

    logger.debug(
        f"calculating dynamic threshold for exceedance extraction using anomaly_type={anomaly_type}, t0={t0}, q={q}"
    )

    if anomaly_type not in ["high", "low"]:
        raise ValueError(f"Invalid value! The `anomaly_type` argument must be 'high' or 'low'")
    if not isinstance(dataset, pd.DataFrame) and not isinstance(dataset, pd.Series):
        raise TypeError("Invalid value! The `dataset` argument must be a Pandas DataFrame or Series")
    if t0 is None or not isinstance(t0, int):
        raise TypeError("Invalid type! `t0` must be a int")
    if q is None or not isinstance(q, float):
        raise TypeError("Invalid type! `q` must be a float")
    if anomaly_type == "low":
        q = 1.0 - q

    logger.debug(f"successfully calculating threshold for {anomaly_type} anomaly type")

    return dataset.expanding(min_periods=t0).quantile(q=q).bfill()


def get_exceedance_peaks_over_threshold(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    threshold_dataset: typing.Union[pd.DataFrame, pd.Series],
    anomaly_type: typing.Literal["high", "low"] = "high",
) -> typing.Union[pd.DataFrame, pd.Series]:
    """
    Extract values from the time series dataset that exceed the POT threshold values.

    ## Parameters
    -------------
    dataset : typing.Union[pd.DataFrame, pd.Series]
        A dataset that must be either Pandas DataFrame or Series.
        * Pandas: One feature needs to be the temporal feature.
        * Series: The index needs to be Pandas DatetimeIndex.

    anomaly_type : typing.Literal["high", "low"], default is "high"
        Type of anomaly to detect - high or low.

    ## Returns
    ----------
    exceedances : typing.Union[pd.DataFrame, pd.Series]
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
    ValueError
        If `t0` is not an integer.
    """

    logger.debug(f"extracting exceedances from dynamic threshold using anomaly_type={anomaly_type}")

    if anomaly_type not in ["high", "low"]:
        raise ValueError(f"Invalid value! The `anomaly_type` argument must be 'high' or 'low'")

    if isinstance(dataset, pd.DataFrame) and isinstance(threshold_dataset, pd.DataFrame):
        if anomaly_type == "high":
            exceedances = dataset.subtract(threshold_dataset, fill_value=0.0).clip(lower=0.0)
        else:
            exceedances = pd.DataFrame(
                np.where(dataset > threshold_dataset, 0.0, np.abs(dataset.subtract(threshold_dataset, fill_value=0))),
                index=dataset.index,
                columns=dataset.columns,
            )
    elif isinstance(dataset, pd.Series) and isinstance(threshold_dataset, pd.Series):
        if anomaly_type == "high":
            exceedances = pd.Series(
                np.maximum(dataset - threshold_dataset, 0.0), index=dataset.index, name="exceedances"
            )
        else:
            exceedances = pd.Series(
                np.where(dataset > threshold_dataset, 0.0, np.abs(dataset - threshold_dataset)),
                index=dataset.index,
                name="exceedances",
            )
    else:
        raise TypeError(
            "Invalid type! both `dataset` and `threshold_dataset` arguments must be of the same type: 2x Pandas DataFrame or 2x Pandas Series"
        )

    logger.debug(f"successfully extracting exceedances from dynamic threshold for {anomaly_type} anomaly type")

    return exceedances


def __gpd_fit_dataframe(
    exceedance_dataset: pd.DataFrame, t1_t2_exceedances: pd.DataFrame, t0: int, gpd_params: typing.Dict
) -> typing.Dict:
    anomaly_scores: typing.Dict = {}

    for column in exceedance_dataset.columns:
        anomaly_scores[f"{column}_anomaly_score"] = []
    anomaly_scores["total_anomaly_score"] = []

    for row in range(0, t1_t2_exceedances.shape[0]):
        fit_exceedances = exceedance_dataset.iloc[: t0 + row]  # type: ignore
        future_exeedance = t1_t2_exceedances.iloc[[row]]
        total_anomaly_score = 0.0
        gpd_params[row] = {}

        for column in t1_t2_exceedances.columns:
            nonzero_fit_exceedances = fit_exceedances[column][fit_exceedances[column] > 0.0].to_list()
            if future_exeedance[column].iloc[0] > 0:
                if len(nonzero_fit_exceedances) > 0:
                    (c, loc, scale) = stats.genpareto.fit(data=nonzero_fit_exceedances, floc=0)
                    p_value: float = stats.genpareto.sf(x=future_exeedance[column].iloc[0], c=c, loc=loc, scale=scale)
                    inverted_p_value = 1 / p_value if p_value > 0.0 else float("inf")
                    total_anomaly_score += inverted_p_value
                    gpd_params[row][column] = dict(
                        c=c,
                        loc=loc,
                        scale=scale,
                        p_value=p_value,
                        anomaly_score=inverted_p_value,
                    )
                    anomaly_scores[f"{column}_anomaly_score"].append(inverted_p_value)
                else:
                    gpd_params[row][column] = dict(
                        c=0.0,
                        loc=0.0,
                        scale=0.0,
                        p_value=0.0,
                        anomaly_score=0.0,
                    )
                    anomaly_scores[f"{column}_anomaly_score"].append(0.0)
            else:
                gpd_params[row][column] = dict(
                    c=0.0,
                    loc=0.0,
                    scale=0.0,
                    p_value=0.0,
                    anomaly_score=0.0,
                )
                anomaly_scores[f"{column}_anomaly_score"].append(0.0)
        gpd_params[row]["total_anomaly_score"] = total_anomaly_score
        anomaly_scores["total_anomaly_score"].append(total_anomaly_score)
    return anomaly_scores


def __gpd_fit_series(
    exceedance_dataset: pd.Series, t1_t2_exceedances: pd.Series, t0: int, gpd_params: typing.Dict
) -> typing.List:
    anomaly_scores: typing.List = []
    for row in range(0, t1_t2_exceedances.shape[0]):
        fit_exceedances = exceedance_dataset.iloc[: t0 + row]
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
    return anomaly_scores


def get_anomaly_score(
    exceedance_dataset: typing.Union[pd.DataFrame, pd.Series], t0: int, gpd_params: typing.Dict
) -> typing.Union[pd.DataFrame, pd.Series]:
    """
    Calculate the anomaly score for each data point in a time series based on the Generalized Pareto Distribution (GPD).

    Anomaly Score = 1 / (1 - CDF(exceedance, c, loc, scale))

    ## Parameters
    -------------
    exceedance_dataset : typing.Union[pd.DataFrame, pd.Series]
        The Pandas Series that contains the exceedances.

    t0 : int
        Time window to get the first day of the T1 time window for dynamic window fitting.

    gpd_params : dict
        A dictionary used as the storage of the GPD parameters (fitting result).

    ## Returns
    ----------
    anomaly_scores : typing.Union[pd.DataFrame, pd.Series]
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
    if t0 is None or not isinstance(t0, int):
        raise TypeError("Invalid type! `t0` must be a int")
    if not isinstance(gpd_params, typing.Dict):
        raise TypeError("Invalid type! The `gpd_params` argument must be a dictionary")
    if not isinstance(exceedance_dataset, pd.DataFrame) and not isinstance(exceedance_dataset, pd.Series):
        raise TypeError("Invalid type! The `exceedance_dataset` argument must be a Pandas DataFrame or Series")

    t1_t2_exceedances = exceedance_dataset.iloc[t0:]

    if isinstance(exceedance_dataset, pd.DataFrame):
        anomaly_scores = pd.DataFrame(
            data=__gpd_fit_dataframe(
                exceedance_dataset=exceedance_dataset,
                t1_t2_exceedances=t1_t2_exceedances,
                t0=t0,
                gpd_params=gpd_params,
            ),
        )
    elif isinstance(exceedance_dataset, pd.Series):
        anomaly_scores = pd.Series(
            index=exceedance_dataset.index[t0:],
            data=__gpd_fit_series(
                exceedance_dataset=exceedance_dataset,
                t1_t2_exceedances=t1_t2_exceedances,
                t0=t0,
                gpd_params=gpd_params,
            ),
            name="anomaly scores",
        )

    logger.debug(f"successfully calculating anomaly score")

    return anomaly_scores


def get_anomaly_threshold(
    anomaly_score_dataset: typing.Union[pd.DataFrame, pd.Series], t1: int, q: float = 0.90
) -> float:
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

    if t1 is None or not isinstance(t1, int):
        raise TypeError("Invalid type! `t1` must be a int")
    if q is None or not isinstance(q, float):
        raise TypeError("Invalid type! `q` must be a float")

    if isinstance(anomaly_score_dataset, pd.DataFrame):
        t1_anomaly_scores = (
            anomaly_score_dataset[  # type: ignore
                (anomaly_score_dataset["total_anomaly_score"] > 0) & (anomaly_score_dataset["total_anomaly_score"] != float("inf"))  # type: ignore
            ]
            .iloc[:t1]["total_anomaly_score"]
            .to_list()
        )
    elif isinstance(anomaly_score_dataset, pd.Series):
        t1_anomaly_scores = (
            anomaly_score_dataset[(anomaly_score_dataset.values > 0) & (anomaly_score_dataset.values != float("inf"))]
            .iloc[:t1]
            .values
        )
    else:
        raise TypeError("Invalid type! The `anomaly_score_dataset` argument must be a Pandas DataFrame or Series")

    if len(t1_anomaly_scores) == 0:
        raise ValueError("There are no total anomaly scores per row > 0")

    logger.debug(f"successfully calculating anomaly threshold using {q} quantile")

    return np.quantile(
        a=t1_anomaly_scores,
        q=q,
    )


def get_anomaly(anomaly_score_dataset: typing.Union[pd.DataFrame, pd.Series], threshold: float, t1: int) -> pd.Series:
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

    logger.debug(f"detecting anomaly using t1={t1}, and `get_anoamly_threshold()` function")

    if not isinstance(anomaly_score_dataset, pd.DataFrame) and not isinstance(anomaly_score_dataset, pd.Series):
        raise TypeError("Invalid type! `anomaly_score_dataset` must be a Pandas DataFrame or Series")
    if threshold is None or not isinstance(threshold, float):
        raise TypeError("Invalid type! `threshold` must be a float")

    t2_anomaly_scores = anomaly_score_dataset.iloc[t1:].copy()

    if isinstance(anomaly_score_dataset, pd.DataFrame):
        detected_data = t2_anomaly_scores["total_anomaly_score"].apply(lambda x: x > threshold).to_list()
    elif isinstance(anomaly_score_dataset, pd.Series):
        detected_data = (t2_anomaly_scores > threshold).values

    logger.debug(f"successfully detecting anomalies using anomaly_threshold={threshold}")

    return pd.Series(index=t2_anomaly_scores.index, data=detected_data, name="detected data")
