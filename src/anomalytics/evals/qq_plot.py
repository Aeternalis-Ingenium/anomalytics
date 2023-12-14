import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


def calculate_theoretical_q(
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
    fit_params: typing.List[typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], float]]],
    is_random_param: bool = False,
) -> typing.Tuple[
    typing.List[typing.Union[pd.Series, np.ndarray]],
    typing.List[typing.List[typing.Union[float, np.float64]]],
    typing.Union[
        typing.List[typing.Dict[str, typing.Union[float, int]]], typing.Dict[str, typing.Union[float, np.float64, int]]
    ],
]:
    """
    Calculate the theoretical quantiles for a given time series and statistical method from the fitting parameters.

    ## Parameters
    -------------
    ts : pandas.Series
        A Pandas Series that contains your data.

    stats_method : typing.Literal["AE", "BM", "DBSCAN", "ISOF", "MAD", "POT", "ZSCORE", "1CSVM"]
        Statistical method to be used for the calculation of theoretical quantiles.

    fit_params : typing.Union[typing.List, typing.Dict]
        Parameters for the statistical method used in the calculation. Can be a list or a dictionary.

    is_random_param : bool, default is False
        If True, randomly selects parameters from the provided list; otherwise, uses the last element of the list or the provided dictionary.

    ## Returns
    ----------
    tuple : (pd.Series, np.ndarray, typing.Union[typing.List, typing.Dict])
        A tuple containing sorted nonzero time series values, calculated theoretical quantiles, and the fit parameters used.

    ## Example
    ----------
    >>> ts = pd.Series([...])
    >>> fit_params = {"c": 0.5, "loc": 0, "scale": 1}
    >>> sorted_ts, theoretical_q, params = calculate_theoretical_q(ts, "POT", fit_params)

    ## Raises
    ----------
    TypeError
        If the input `ts` is not a Pandas Series.
    ValueError
        If the `stats_method` is not one of the predefined statistical methods.
    NotImplementedError
        If the selected statistical method's implementation is not available.
    """

    logger.debug(f"performing theoretical quantile calculation for qq plot with fit_params={fit_params}")

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
        if isinstance(dataset, pd.DataFrame):
            nonzero_datasets: typing.List = []
            theoretical_qs: typing.List = []
            nonzero_fit_params: typing.List = []

            for index, column in enumerate(dataset.columns):  # type: ignore
                exceedence_series = dataset[column].copy()  # type: ignore
                nonzero_exceedances = exceedence_series[exceedence_series.values > 0]
                sorted_nonzero_exceedances = np.sort(nonzero_exceedances)
                q = np.arange(1, len(sorted_nonzero_exceedances) + 1) / (len(sorted_nonzero_exceedances) + 1)

                if is_random_param:
                    nonzero_params = fit_params[index][column][np.random.randint(low=0, high=len(fit_params) - 1)]  # type: ignore
                else:
                    nonzero_params = fit_params[index][column][-1]  # type: ignore
                theoretical_q = stats.genpareto.ppf(
                    q=q, c=nonzero_params["c"], loc=nonzero_params["loc"], scale=nonzero_params["scale"]
                )
                nonzero_datasets.append(sorted_nonzero_exceedances)
                theoretical_qs.append(theoretical_q)
                nonzero_fit_params.append(nonzero_params)
            return (nonzero_datasets, theoretical_qs, nonzero_fit_params)

        elif isinstance(dataset, pd.Series):
            nonzero_exceedances = dataset[dataset.values > 0]
            sorted_nonzero_exceedances = np.sort(nonzero_exceedances)
            q = np.arange(1, len(sorted_nonzero_exceedances) + 1) / (len(sorted_nonzero_exceedances) + 1)

            if is_random_param:
                nonzero_params = fit_params[np.random.randint(low=0, high=len(fit_params) - 1)]
            else:
                nonzero_params = fit_params[-1]
            c = nonzero_params["c"]
            loc = nonzero_params["loc"]
            scale = nonzero_params["scale"]
            theoretical_q = stats.genpareto.ppf(q=q, c=c, loc=loc, scale=scale)
            logger.debug(
                f"successfully performing theoretical quantile calculation for qq plot with fit_params={fit_params}"
            )
            return (sorted_nonzero_exceedances, theoretical_q, nonzero_params)  # type: ignore

    elif stats_method == "ZS":
        raise NotImplementedError("Not implemented yet!")

    elif stats_method == "1CSVM":
        raise NotImplementedError("Not implemented yet!")

    logger.debug(f"fail to perform theoretical quantile calculation for qq plot with fit_params={fit_params}")

    raise ValueError(
        "Invalid value! Available arguments for `stats_method`: 'AE', 'BM', 'DBSCAN', 'ISOF', 'MAD', 'POT', 'ZS','1CSVM'"
    )


def visualize_qq_plot(
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
    fit_params: typing.List[typing.Dict[str, typing.Union[typing.List[typing.Dict[str, float]], float]]],
    is_random_param: bool = False,
    plot_width: int = 15,
    plot_height: int = 10,
):
    """
    Visualize a QQ plot for a given time series based on the specified statistical method for visual evaluation.

    ## Parameters
    -------------
    dataset : pandas.Series
        A Pandas Series that contains your data.

    stats_method : typing.Literal["AE", "BM", "DBSCAN", "ISOF", "MAD", "POT", "ZSCORE", "1CSVM"]
        The statistical method used for generating the QQ plot.

    fit_params : typing.Union[typing.List, typing.Dict]
        Parameters for the statistical method used in the plot. Can be a list or a dictionary.

    is_random_param : bool, default is False
        If True, randomly selects parameters from the provided list; otherwise, uses the last element of the list or the provided dictionary.

    plot_width : int, default is 15
        The width for figsize in `Matplotlib.plt.figure()`.

    plot_height : int, default is 10
        The height for figsize in `Matplotlib.plt.figure()`.

    ## Example
    ----------
    >>> ts = pd.Series([...])
    >>> fit_params = {"c": 0.5, "loc": 0, "scale": 1}
    >>> visualize_qq_plot(ts, "POT", fit_params)
    # This will display the QQ plot

    ## Raises
    ---------
    TypeError
        If the input `ts` is not a Pandas Series or if `ts.index` is not a DatetimeIndex.
    SyntaxError
        If there's an error converting `ts.index` to a pandas.DatetimeIndex.
    NotImplementedError
        If the selected statistical method's implementation for visualization is not available.
    """

    logger.debug(f"performing qq plot for {stats_method} analysis with total of {len(fit_params)} fir params")

    suptitle = "QQ Plot"
    x_label = "Theoretical Quantiles"
    y_label = "Sample Quantiles"

    if stats_method == "AE":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Not implemented yet!")
        elif isinstance(dataset, pd.Series):
            raise NotImplementedError("Not implemented yet!")

    elif stats_method == "BM":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Not implemented yet!")
        elif isinstance(dataset, pd.Series):
            raise NotImplementedError("Not implemented yet!")

    elif stats_method == "DBSCAN":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Not implemented yet!")
        elif isinstance(dataset, pd.Series):
            raise NotImplementedError("Not implemented yet!")

    elif stats_method == "ISOF":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Not implemented yet!")
        elif isinstance(dataset, pd.Series):
            raise NotImplementedError("Not implemented yet!")

    elif stats_method == "MAD":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Not implemented yet!")
        elif isinstance(dataset, pd.Series):
            raise NotImplementedError("Not implemented yet!")

    elif stats_method == "POT":
        if isinstance(dataset, pd.DataFrame):
            (sorted_nonzero_datasets, theoretical_qs, params) = calculate_theoretical_q(
                dataset=dataset, fit_params=fit_params, stats_method=stats_method, is_random_param=is_random_param
            )
            fig, axs = plt.subplots(figsize=(20, 15), nrows=len(sorted_nonzero_datasets))

            for index in range(0, len(sorted_nonzero_datasets)):
                ax = axs[index]
                ax.scatter(
                    theoretical_qs[index],
                    sorted_nonzero_datasets[index],
                    c="black",
                    label=f"{len(sorted_nonzero_datasets[index])} Exceedences > 0",  # type: ignore
                )
                ax.plot(
                    [np.min(theoretical_qs[index]), np.max(theoretical_qs[index])],  # type: ignore
                    [np.min(theoretical_qs[index]), np.max(theoretical_qs[index])],  # type: ignore
                    c="lime",
                    lw=2,
                    label=f"\nFitted GPD Params:\n    c: {round(params[index]['c'], 2)}\n    loc: {round(params[index]['loc'], 2)}\n    scale: {round(params[index]['scale'], 2)}",  # type: ignore
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend(loc="upper left", shadow=True, fancybox=True)
                ax.set_title(f"\nDataset Column - {index}", fontsize=12)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            fig.suptitle(suptitle, fontsize=12)

        elif isinstance(dataset, pd.Series):
            (sorted_nonzero_dataset, theoretical_q, params) = calculate_theoretical_q(
                dataset=dataset, fit_params=fit_params, stats_method=stats_method, is_random_param=is_random_param
            )
            fig = plt.figure(figsize=(plot_width, plot_height))
            scatter_label = f"{len(sorted_nonzero_dataset)} Exceedances > 0"
            plot_label = f"\nFitted GPD Params:\n  c: {round(params['c'], 2)}\n  loc: {round(params['loc'], 2)}\n  scale: {round(params['scale'], 2)}"  # type: ignore

            plt.scatter(theoretical_q, sorted_nonzero_dataset, c="black", label=scatter_label)  # type: ignore
            plt.plot(
                [np.min(theoretical_q), np.max(theoretical_q)],  # type: ignore
                [np.min(theoretical_q), np.max(theoretical_q)],  # type: ignore
                c="lime",
                lw=2,
                label=plot_label,
            )
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Series Dataset", fontsize=12)
            fig.legend(loc="upper left", shadow=True, fancybox=True)
            fig.suptitle(suptitle, fontsize=12)

    elif stats_method == "ZS":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Not implemented yet!")
        elif isinstance(dataset, pd.Series):
            raise NotImplementedError("Not implemented yet!")

    elif stats_method == "1CSVM":
        if isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Not implemented yet!")
        elif isinstance(dataset, pd.Series):
            raise NotImplementedError("Not implemented yet!")

    logger.debug(f"fail to plot qq for {stats_method} analysis")

    plt.show()
