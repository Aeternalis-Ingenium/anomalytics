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
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["BM"],
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["DBSCAN"],
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["ISOF",],
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["MAD"],
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["POT"],
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["ZSCORE",],
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
    ...


@typing.overload
def ks_1sample(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    stats_method: typing.Literal["1CSVM",],
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
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
    fit_params: typing.List[
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], datetime.datetime, float]]
    ],
) -> typing.Dict[str, typing.Union[typing.List[str], typing.List[float], typing.List[int]]]:
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
            nonzero_exceedance_series = [dataset[dataset[column] > 0][column].copy() for column in dataset.columns]
            kstest_results: typing.Dict = {}

            for index, column in enumerate(dataset.columns):  # type: ignore
                kstest_results[column] = {}
                ks_result = stats.ks_1samp(
                    x=nonzero_exceedance_series[index],
                    cdf=stats.genpareto.cdf,
                    args=(
                        fit_params[index][column][-1]["c"],  # type: ignore
                        fit_params[index][column][-1]["loc"],  # type: ignore
                        fit_params[index][column][-1]["scale"],  # type: ignore
                    ),
                )
                kstest_results[column]["total_exceedances"] = nonzero_exceedance_series[index].shape[0]  # type: ignore
                kstest_results[column]["stat_distance"] = ks_result.statistic  # type: ignore
                kstest_results[column]["p_value"] = ks_result.pvalue  # type: ignore
                kstest_results[column]["c"] = fit_params[index][column][-1]["c"]  # type: ignore
                kstest_results[column]["loc"] = fit_params[index][column][-1]["loc"]  # type: ignore
                kstest_results[column]["scale"] = fit_params[index][column][-1]["scale"]  # type: ignore

            columns: typing.List = []
            total_nonzero_exceedances: typing.List = []
            stats_distances: typing.List = []
            p_values: typing.List = []
            cs: typing.List = []
            locs: typing.List = []
            scales: typing.List = []

            for column in kstest_results.keys():
                columns.append(column)
                total_nonzero_exceedances.append(kstest_results[column]["total_exceedances"])
                stats_distances.append(kstest_results[column]["stat_distance"])
                p_values.append(kstest_results[column]["p_value"])
                cs.append(kstest_results[column]["c"])
                locs.append(kstest_results[column]["loc"])
                scales.append(kstest_results[column]["scale"])

            return dict(
                column=columns,
                total_nonzero_exceedances=total_nonzero_exceedances,
                stats_distance=stats_distances,
                p_value=p_values,
                c=cs,
                loc=locs,
                scale=scales,
            )

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

            total_nonzero_exceedances: typing.List[int] = [dataset.shape[0]]  # type: ignore
            stats_distances: typing.List[float] = [ks_result.statistic]  # type: ignore
            p_values: typing.List[float] = [ks_result.pvalue]  # type: ignore

            return dict(
                total_nonzero_exceedances=total_nonzero_exceedances,
                stats_distance=stats_distances,
                p_value=p_values,
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
