import logging
import typing

from anomalytics.time_windows.pot_windows import compute_pot_windows

logger = logging.getLogger(__name__)


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["AE"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    encoded: typing.List[float],
) -> typing.Tuple:
    ...


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["BM"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    block_size: float = 365.2425,
) -> typing.Tuple:
    ...


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["DBSCAN"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    total_cluster: int,
) -> typing.Tuple:
    ...


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["ISOF"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    isolated: typing.List[float],
) -> typing.Tuple:
    ...


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["MAD"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    medians: typing.List[float],
) -> typing.Tuple:
    ...


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["1CSVM"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    vectors: typing.List[float],
) -> typing.Tuple:
    ...


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["POT"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    t0_pct: float = 0.65,
    t1_pct: float = 0.25,
    t2_pct: float = 0.10,
) -> typing.Tuple:
    ...


@typing.overload
def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["ZS"],
    analysis_type: typing.Literal["historical", "real-time"] = "historical",
    *,
    upper: float,
    lower: float,
) -> typing.Tuple:
    ...


def set_time_window(  # type: ignore
    total_rows: int,
    method: typing.Literal["AE", "BM", "DBSCAN", "ISOF", "MAD", "1CSVM", "POT", "ZS"],
    analysis_type: typing.Literal["historical", "real-time"],
    **kwargs,
) -> typing.Tuple:
    if method == "AE":
        raise NotImplementedError("The method 'AE' for Autoencoder specific time windows hasn't been implemented yet")
    if method == "BM":
        raise NotImplementedError("The method 'BM' for Block Maxima specific time windows hasn't been implemented yet")
    if method == "DBSCAN":
        raise NotImplementedError(
            "The method 'DBSCAN' for Density-Based Spatial Clustering Application with Noise specific time windows hasn't been implemented yet"
        )
    if method == "ISOF":
        raise NotImplementedError(
            "The method 'ISOF' for Isolation Forest specific time windows hasn't been implemented yet"
        )
    if method == "MAD":
        raise NotImplementedError(
            "The method 'MAD' for Median Absolute Deviation specific time windows hasn't been implemented yet"
        )
    if method == "1CSVM":
        raise NotImplementedError(
            "The method '1CSVM' for One Class Support Vector Method specific time windows hasn't been implemented yet"
        )
    if method == "POT":
        return compute_pot_windows(total_rows=total_rows, analysis_type=analysis_type, **kwargs)
    if method == "ZS":
        raise NotImplementedError("The method 'ZS' for Z-Score specific time windows hasn't been implemented yet")

    raise ValueError(
        f"Invalid value in '{method}' for the 'method' argument. Availabe methods are: "
        "'AUTO', 'BM', 'DBSCAN', 'ISOF', 'MAD', '1CSVM', 'POT', 'ZS'"
    )
