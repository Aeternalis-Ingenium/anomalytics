import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def plot_line(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    threshold: typing.Optional[typing.Union[pd.Series, float, np.float64, np.number]],
    title: str,
    xlabel: str,
    ylabel: str,
    is_threshold: bool = True,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    th_color: str = "red",
    th_type: str = "dashed",
    th_line_width: int = 2,
    alpha: float = 0.8,
):
    fig = plt.figure(figsize=(plot_width, plot_height))
    plt.plot(dataset.index, dataset.values, color=plot_color, alpha=alpha, label=f"{dataset.shape[0]} Data Points")

    if is_threshold:
        if not isinstance(threshold, pd.Series) and (
            isinstance(threshold, float) or isinstance(threshold, np.float64) or isinstance(threshold, np.number)
        ):
            plt.axhline(
                float(threshold), c=th_color, ls=th_type, lw=th_line_width, label=f"{threshold} Anomaly Threshold"
            )
        else:
            plt.plot(
                dataset.index,
                threshold.values,
                c=th_color,
                ls=th_type,
                lw=th_line_width,
                label=f"Exceedance Threshold Mean {threshold.mean()}",
            )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    fig.legend(loc="upper left", shadow=True, fancybox=True)
    plt.show()


def plot_hist(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    bins: typing.Optional[int] = 50,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
):
    fig = plt.figure(figsize=(plot_width, plot_height))
    plt.hist(dataset.values, bins=bins, color=plot_color, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    fig.legend(loc="upper left", shadow=True, fancybox=True)
    plt.show()


def plot_gen_pareto(
    dataset: typing.Union[pd.DataFrame, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    bins: typing.Optional[int] = 50,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    params: typing.Optional[typing.Dict] = None,
):
    fig = plt.figure(figsize=(plot_width, plot_height))

    if params:
        param_label = f"\n{round(params['c'], 3)}\n{round(params['loc'], 3)}\n{round(params['scale'], 3)}\n"
        overlay = np.linspace(
            stats.genpareto.ppf(0.1, c=params["c"], loc=params["loc"], scale=params["scale"]),
            stats.genpareto.ppf(0.999, c=params["c"], loc=params["loc"], scale=params["scale"]),
            len(dataset),
        )
        plt.plot(
            overlay,
            stats.genpareto.pdf(overlay, c=params["c"], loc=params["loc"], scale=params["scale"]),
            c="lime",
            lw=2,
            label=f"\nFitted Params:{param_label}",
        )
    plt.hist(
        dataset,
        bins=bins,
        density=True,
        alpha=alpha,
        color=plot_color,
        label=f"{len(dataset)}",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.legend(loc="upper right", shadow=True, fancybox=True)
    plt.show()
