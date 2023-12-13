import datetime
import logging
import typing

import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["AE"],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["BM"],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["DBSCAN",],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["ISOF",],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["MAD"],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["POT"],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["ZSCORE",],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["1CSVM",],
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
    ...


def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
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
    fit_params: typing.List[typing.Dict[str, typing.Union[datetime.datetime, float]]],
) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
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

    if not isinstance(dataset, pd.DataFrame) and not isinstance(dataset, pd.Series):
        raise TypeError("Invalid value! The `dataset` argument must be a Pandas DataFrame or Series")
    if not isinstance(fit_params, typing.List):
        raise TypeError("Invalid type! `fit_params` must be a list")

    if stats_method == "AE":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            raise NotImplementedError()

    if stats_method == "BM":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            raise NotImplementedError()

    if stats_method == "DBSCAN":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            raise NotImplementedError()

    if stats_method == "ISOF":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            raise NotImplementedError()

    if stats_method == "MAD":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            raise NotImplementedError()

    if stats_method == "POT":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            c: float = fit_params[-1]["c"]  # type: ignore
            loc: float = fit_params[-1]["loc"]  # type: ignore
            scale: float = fit_params[-1]["scale"]  # type: ignore
            ks_result = stats.ks_1samp(
                x=dataset[dataset.values > 0],
                cdf=stats.genpareto.cdf,
                args=(c, loc, scale),
                alternative="two-sided",
                method="exact",
            )

            logger.debug(
                f"successfully performing kolmogorov smirnov test for {stats_method} stats method with result of  {ks_result.statistic} `stats_distance`"
            )

            total_nonzero_exceedances: typing.List[int] = [dataset.shape[0]]
            stats_distance: typing.List[float] = [ks_result.statistic]
            p_value: typing.List[float] = [ks_result.pvalue]

            return dict(
                total_nonzero_exceedances=total_nonzero_exceedances,
                stats_distance=stats_distance,
                p_value=p_value,
                c=[c],
                loc=[loc],
                scale=[scale],
            )

    if stats_method == "ZS":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            raise NotImplementedError()

    if stats_method == "1CSVM":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError()

        if isinstance(dataset, pd.Series):
            raise NotImplementedError()

    logger.debug(f"fail to perform kolmogorov smirnov test for {stats_method} stats method")

    raise ValueError(
        "Invalid value! Available arguments for `stats_method`: 'AE', 'BM', 'DBSCAN', 'ISOF', 'MAD', 'POT', 'ZS','1CSVM'"
    )
