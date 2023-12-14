import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_line_dataset_dataframe(
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
):
    fig, axs = plt.subplots(figsize=(plot_width, plot_height), nrows=len(datasets))
    for index in range(0, len(datasets)):
        ax = axs[index]
        ax.plot(
            datasets[index].index,
            datasets[index].values,
            color=plot_color,
            alpha=alpha,
            label=f"{datasets[index].shape[0]} Data",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"\nColumn - {columns[index]}", fontsize=10)
        ax.legend(loc="upper left", shadow=True, fancybox=True)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_line_dataset_series(
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    dataset: pd.Series,
):
    fig = plt.figure(figsize=(plot_width, plot_height))
    plt.plot(dataset.index, dataset.values, color=plot_color, alpha=alpha, label=f"{dataset.shape[0]} Data")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.legend(loc="upper left", shadow=True, fancybox=True)
    plt.show()


def plot_line_exceedance_dataframe(
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
    thresholds: typing.List[pd.Series],
    th_color: str = "red",
    th_type: str = "dashed",
    th_line_width: int = 2,
):
    fig, axs = plt.subplots(figsize=(plot_width, plot_height), nrows=len(datasets))
    for index in range(0, len(datasets)):
        ax = axs[index]
        ax.plot(
            datasets[index].index,
            datasets[index].values,
            color=plot_color,
            alpha=alpha,
            label=f"{datasets[index].shape[0]} Data",
        )
        ax.plot(
            datasets[index].index,
            thresholds[index].values,
            c=th_color,
            ls=th_type,
            lw=th_line_width,
            label=f"Threshold Mean {round(thresholds[index].mean(), 2)}",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"\nColumn - {columns[index]}", fontsize=10)
        ax.legend(loc="upper left", shadow=True, fancybox=True)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_line_exceedance_series(
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    dataset: pd.Series,
    threshold: pd.Series,
    th_color: str = "red",
    th_type: str = "dashed",
    th_line_width: int = 2,
):
    fig = plt.figure(figsize=(plot_width, plot_height))
    plt.plot(dataset.index, dataset.values, color=plot_color, alpha=alpha, label=f"{dataset.shape[0]} Data")
    plt.plot(
        dataset.index,
        threshold.values,
        c=th_color,
        ls=th_type,
        lw=th_line_width,
        label=f"Threshold Mean {round(threshold.mean(), 2)}",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.legend(loc="upper left", shadow=True, fancybox=True)
    plt.show()


def plot_line_anomaly_score_dataframe(
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
    threshold: typing.Union[float, np.float64, np.number],
    th_color: str = "red",
    th_type: str = "dashed",
    th_line_width: int = 2,
):
    fig, axs = plt.subplots(figsize=(plot_width, plot_height), nrows=len(datasets))
    for index in range(0, len(datasets)):
        ax = axs[index]
        ax.plot(
            datasets[index].index,
            datasets[index].values,
            color=plot_color,
            alpha=alpha,
            label=f"{datasets[index].shape[0]} Data",
        )
        ax.axhline(
            float(threshold), c=th_color, ls=th_type, lw=th_line_width, label=f"{round(threshold, 2)} Threshold"  # type: ignore
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"\nColumn - {columns[index]}", fontsize=10)
        ax.legend(loc="upper left", shadow=True, fancybox=True)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_line_anomaly_score_series(
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    dataset: pd.Series,
    threshold: typing.Union[float, np.float64, np.number],
    th_color: str = "red",
    th_type: str = "dashed",
    th_line_width: int = 2,
):
    fig = plt.figure(figsize=(plot_width, plot_height))
    plt.plot(dataset.index, dataset.values, color=plot_color, alpha=alpha, label=f"{dataset.shape[0]} Data")
    plt.axhline(float(threshold), c=th_color, ls=th_type, lw=th_line_width, label=f"{round(threshold, 2)} Threshold")  # type: ignore
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.legend(loc="upper left", shadow=True, fancybox=True)
    plt.show()
