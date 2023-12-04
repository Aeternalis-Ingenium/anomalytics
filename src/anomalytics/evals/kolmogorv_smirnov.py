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

    # Returns
    ---------
    ks_result : typing.Dict
        A storage that contains the result of Kolmogorov Smirnov and other analysis details such as fitting params, first, and last datetime.
    """
    logger.debug(f"performing kolmogorov smirnov test for stats_method={stats_method} with fit_params={fit_params}")

    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        try:
            datetime_index = pd.to_datetime(ts.index)
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
            x=ts.values,
            cdf=stats.genpareto.cdf,
            args=(c, loc, scale),
            alternative="two-sided",
            method="exact",
        )

        logger.debug(
            f"successfully performing kolmogorov smirnov test for {stats_method} stats method with result of  {ks_result.statistic} `stats_distance`"
        )

        return dict(
            total_nonzero_exceedances=len(ts),
            start_datetime=ts.index[0],
            end_datetime=fit_params[-1]["datetime"],
            stats_distance=ks_result.statistic,
            p_value=ks_result.pvalue,
            c=c,
            loc=loc,
            scale=scale,
        )
    if stats_method == "ZS":
        raise NotImplementedError()

    if stats_method == "1CSVM":
        raise NotImplementedError()

    logger.debug(f"fail to perform kolmogorov smirnov test for {stats_method} stats method")

    raise ValueError(
        "Invalid value! Available arguments for `stats_method`: 'AE', 'BM', 'DBSCAN', 'ISOF', 'MAD', 'POT', 'ZS','1CSVM'"
    )