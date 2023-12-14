import typing

import numpy as np
import pandas as pd

from anomalytics.plots.hist import (
    plot_hist_dataset_dataframe,
    plot_hist_dataset_series,
    plot_hist_gpd_dataframe,
    plot_hist_gpd_series,
)
from anomalytics.plots.line import (
    plot_line_anomaly_score_dataframe,
    plot_line_anomaly_score_series,
    plot_line_dataset_dataframe,
    plot_line_dataset_series,
    plot_line_exceedance_dataframe,
    plot_line_exceedance_series,
)


@typing.overload
def visualize(
    plot_type: typing.Literal["hist-dataset-df",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["hist-dataset-ts",],
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    *,
    dataset: pd.Series,
    bins: int = 50,
):
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["hist-gpd-df",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["hist-gpd-ts",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["line-anomaly-score-df",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["line-anomaly-score-ts",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["line-dataset-df",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["line-dataset-ts",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["line-exceedance-df",],
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
    ...


@typing.overload
def visualize(
    plot_type: typing.Literal["line-exceedance-ts",],
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
    ...


def visualize(
    plot_type: typing.Literal[
        "hist-dataset-df",
        "hist-dataset-ts",
        "hist-gpd-df",
        "hist-gpd-ts",
        "line-anomaly-score-df",
        "line-anomaly-score-ts",
        "line-dataset-df",
        "line-dataset-ts",
        "line-exceedance-df",
        "line-exceedance-ts",
    ],
    title: str,
    xlabel: str,
    ylabel: str,
    plot_width: int = 13,
    plot_height: int = 8,
    plot_color: str = "black",
    alpha: float = 0.8,
    **kwargs,
):
    if plot_type == "hist-dataset-df":
        return plot_hist_dataset_dataframe(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "hist-dataset-ts":
        return plot_hist_dataset_series(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "hist-gpd-df":
        return plot_hist_gpd_dataframe(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "hist-gpd-ts":
        return plot_hist_gpd_series(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "line-anomaly-score-df":
        return plot_line_anomaly_score_dataframe(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "line-anomaly-score-ts":
        return plot_line_anomaly_score_series(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "line-dataset-df":
        return plot_line_dataset_dataframe(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "line-dataset-ts":
        return plot_line_dataset_series(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "line-exceedance-df":
        return plot_line_exceedance_dataframe(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
    if plot_type == "line-exceedance-ts":
        return plot_line_exceedance_series(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_color=plot_color,
            alpha=alpha,
            **kwargs,
        )
