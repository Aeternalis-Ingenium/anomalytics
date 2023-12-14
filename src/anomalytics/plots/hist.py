import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def plot_hist_dataset_dataframe(
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    columns: typing.List[str],
    datasets: typing.List[pd.Series],
    bins: int = 50,
):
    fig, axs = plt.subplots(figsize=(plot_width, plot_height), nrows=len(datasets))
    for index in range(0, len(datasets)):
        ax = axs[index]
        ax.hist(
            datasets[index].values,
            bins=bins,
            density=True,
            alpha=alpha,
            color=plot_color,
            label=f"{datasets[index].shape[0]} Data >0",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"\nColumn - {columns[index]}", fontsize=10)
        ax.legend(loc="upper left", shadow=True, fancybox=True)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_hist_dataset_series(
    title: str,
    xlabel: str,
    ylabel: str,
    bins: int = 50,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    dataset: pd.Series,
):
    fig = plt.figure(figsize=(plot_width, plot_height))
    plt.hist(
        dataset.values,
        bins=bins,
        color=plot_color,
        alpha=alpha,
        label=f"{dataset.shape[0]} Data >0",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.legend(loc="upper left", shadow=True, fancybox=True)
    plt.show()


def plot_hist_gpd_dataframe(
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    columns: typing.List[str],
    datasets: typing.List[pd.Series],
    params: typing.List[typing.Dict[str, typing.List[typing.Union[float, int]]]],
    bins: int = 50,
):
    fig, axs = plt.subplots(figsize=(plot_width, plot_height), nrows=len(datasets))
    for index in range(0, len(datasets)):
        ax = axs[index]
        param_label = f"{round(params[index]['c'], 2)}\n{round(params[index]['loc'], 2)}\n{round(params[index]['scale'], 2)}\n"  # type: ignore
        overlay = np.linspace(
            start=stats.genpareto.ppf(
                q=0.001, c=params[index]["c"], loc=params[index]["loc"], scale=params[index]["scale"]
            ),
            stop=stats.genpareto.ppf(
                q=0.999, c=params[index]["c"], loc=params[index]["loc"], scale=params[index]["scale"]
            ),
            num=datasets[index].shape[0],
        )
        ax.plot(
            overlay,
            stats.genpareto.pdf(
                x=overlay, c=params[index]["c"], loc=params[index]["loc"], scale=params[index]["scale"]
            ),
            c="lime",
            lw=2,
            label=param_label,
        )
        ax.hist(
            datasets[index].values,
            bins=bins,
            density=True,
            alpha=alpha,
            color=plot_color,
            label=f"{datasets[index].shape[0]} Exceedances >0",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", shadow=True, fancybox=True)
        ax.set_title(f"\nColumn - {columns[index]}", fontsize=10)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_hist_gpd_series(
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    dataset: pd.Series,
    params: typing.Dict[str, typing.Union[float, int]],
    bins: int = 50,
):
    fig = plt.figure(figsize=(plot_width, plot_height))
    param_label = f"\n{round(params['c'], 2)}\n{round(params['loc'], 2)}\n{round(params['scale'], 2)}\n"
    overlay = np.linspace(
        start=stats.genpareto.ppf(q=0.001, c=params["c"], loc=params["loc"], scale=params["scale"]),
        stop=stats.genpareto.ppf(q=0.999, c=params["c"], loc=params["loc"], scale=params["scale"]),
        num=dataset.shape[0],
    )
    plt.plot(
        overlay,
        stats.genpareto.pdf(x=overlay, c=params["c"], loc=params["loc"], scale=params["scale"]),
        c="lime",
        lw=2,
        label=param_label,
    )
    plt.hist(
        dataset,
        bins=bins,
        density=True,
        alpha=alpha,
        color=plot_color,
        label=f"{dataset.shape[0]} Exceedances > 0",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.legend(loc="upper right", shadow=True, fancybox=True)
    plt.show()
