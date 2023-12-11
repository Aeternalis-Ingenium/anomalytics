import logging
import typing

import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


def ks_1sample(
    ts: pd.Series,
    stats_method: typing.Literal[
        "AE",
        "BM",
        "DBSCAN",
        "ISOF",
        "MAD",
        "POT",
        "ZSCORE",
        "1CSVM",
    ],
    fit_params: typing.Union[typing.List, typing.Dict],
) -> typing.Dict:
    """
    Evaluate sample and the theoretical distribution using Kolmogorov Smirnov method via `scipy.stats.ks:_1samp()`.

    ## Parameters
    -------------
    ts : pandas.Series
        A Pandas Series that contains your data.

    stats_method : typing.Literal["AE", "BM", "DBSCAN", "ISOF", "MAD", "POT", "ZS","1CSVM"]
        Statistical method to define the needed fitting parameters.

    fit_params : typing.Dict
        The result of fitting e.g. c, loc, scale.

    ## Returns
    ----------
    ks_result : typing.Dict
        A storage that contains the result of Kolmogorov Smirnov and other analysis details such as fitting params, first, and last datetime.

    ## Example
    ----------
    >>> ks_result = ks_1sample(ts=exceedance_df, stats_method="POT", fit_params=params)
    >>> print(ks_result)
    ...
    {'total_nonzero_exceedances': 1079, 'start_datetime': '2023-10-10T00:00:00.000Z', 'end_datetime': '2023-10-11T01:00:00.000Z', 'stats_distance': 0.7884, 'p_value': 0.8987, 'c': -2.324, 'loc': 0, 'scale': 0.025}
    """
    logger.debug(f"performing kolmogorov smirnov test for stats_method={stats_method} with fit_params={fit_params}")

    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        try:
            datetime_index = pd.to_datetime(ts.index.values, utc=True)
            ts.index = datetime_index
        except Exception as err:
            raise SyntaxError("Syntax error! Fail to convert `ts.index` into pandas.DatetimeIndex.") from err

    if stats_method == "AE":
        raise NotImplementedError()

    if stats_method == "BM":
        raise NotImplementedError()

    if stats_method == "DBSCAN":
        raise NotImplementedError()

    if stats_method == "ISOF":
        raise NotImplementedError()

    if stats_method == "MAD":
        raise NotImplementedError()

    if stats_method == "POT":
        c = fit_params[-1]["c"]
        loc = fit_params[-1]["loc"]
        scale = fit_params[-1]["scale"]
        ks_result = stats.ks_1samp(
            x=ts[ts.values > 0],
            cdf=stats.genpareto.cdf,
            args=(c, loc, scale),
            alternative="two-sided",
            method="exact",
        )

        logger.debug(
            f"successfully performing kolmogorov smirnov test for {stats_method} stats method with result of  {ks_result.statistic} `stats_distance`"
        )

        return dict(
            total_nonzero_exceedances=[ts.shape[0]],
            stats_distance=[ks_result.statistic],
            p_value=[ks_result.pvalue],
            c=[c],
            loc=[loc],
            scale=[scale],
        )
    if stats_method == "ZS":
        raise NotImplementedError()

    if stats_method == "1CSVM":
        raise NotImplementedError()

    logger.debug(f"fail to perform kolmogorov smirnov test for {stats_method} stats method")

    raise ValueError(
        "Invalid value! Available arguments for `stats_method`: 'AE', 'BM', 'DBSCAN', 'ISOF', 'MAD', 'POT', 'ZS','1CSVM'"
    )
