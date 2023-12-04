import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


def calculate_theoretical_q(
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
    is_random_param: bool = False,
) -> typing.Tuple[pd.Series, np.ndarray[typing.Any, np.dtype[typing.Any]], typing.Union[typing.List, typing.Dict]]:
    logger.debug(f"performing theoretical quantile calculation for qq plot with fit_params={fit_params}")

    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")

    nonzero_ts = ts[ts.values > 0]
    sorted_nonzero_ts = np.sort(nonzero_ts)
    q = np.arange(1, len(sorted_nonzero_ts) + 1) / (len(sorted_nonzero_ts) + 1)

    if is_random_param:
        nonzero_params = fit_params[np.random.randint(a=0, b=len(fit_params) - 1)]
    else:
        nonzero_params = fit_params[-1]

    if stats_method == "AE":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "BM":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "DBSCAN":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "ISOF":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "MAD":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "POT":
        c = nonzero_params["c"]
        loc = nonzero_params["loc"]
        scale = nonzero_params["scale"]
        theoretical_q = stats.genpareto.ppf(q=q, c=c, loc=loc, scale=scale)
        logger.debug(
            f"successfully performing theoretical quantile calculation for qq plot with fit_params={fit_params}"
        )
        return (sorted_nonzero_ts, theoretical_q, nonzero_params)

    elif stats_method == "ZS":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "1CSVM":
        raise NotImplementedError("Not implemented yet!")

    logger.debug(f"fail to perform theoretical quantile calculation for qq plot with fit_params={fit_params}")

    raise ValueError(
        "Invalid value! Available arguments for `stats_method`: 'AE', 'BM', 'DBSCAN', 'ISOF', 'MAD', 'POT', 'ZS','1CSVM'"
    )


def visualize_qq_plot(
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
    is_random_param: bool = False,
    plot_width: int = 15,
    plot_height: int = 10,
):
    logger.debug(f"performing qq plot for {stats_method} analysis with total of {len(fit_params)} fir params")

    if not isinstance(ts, pd.Series):
        raise TypeError("Invalid value! The `ts` argument must be a Pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        try:
            datetime_index = pd.to_datetime(ts.index.values, utc=True)
            ts.index = datetime_index
        except Exception as err:
            raise SyntaxError("Syntax error! Fail to convert `ts.index` into pandas.DatetimeIndex.") from err

    fig = plt.figure(figsize=(plot_width, plot_height))
    (sorted_nonzero_ts, theoretical_q, params) = calculate_theoretical_q(
        ts=ts, fit_params=fit_params, stats_method=stats_method, is_random_param=is_random_param
    )

    if stats_method == "AE":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "BM":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "DBSCAN":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "ISOF":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "MAD":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "POT":
        scatter_label = f"{len(sorted_nonzero_ts)} Exceedances > 0"
        plot_label = f"\nFitted GPD Params:\n  datetime: {params['datetime']}\n  c: {round(params['c'], 3)}\n  loc: {round(params['loc'], 3)}\n  scale: {round(params['scale'], 3)}"  # type: ignore
        suptitle = "QQ Plot - GPD"

    elif stats_method == "ZS":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "1CSVM":
        raise NotImplementedError("Not implemented yet!")

    logger.debug(f"fail to plot qq for {stats_method} analysis")

    plt.scatter(theoretical_q, sorted_nonzero_ts, c="black", label=scatter_label)
    plt.plot(
        [np.min(theoretical_q), np.max(theoretical_q)],
        [np.min(theoretical_q), np.max(theoretical_q)],
        c="lime",
        lw=2,
        label=plot_label,
    )
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.title(f"Period: {ts.index[0]} - {ts.index[-1]}", fontsize=12)
    fig.legend(loc="upper left", shadow=True, fancybox=True)
    fig.suptitle(suptitle, fontsize=12)
    plt.show()