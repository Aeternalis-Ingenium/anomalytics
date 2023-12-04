import logging
import typing

from anomalytics.time_windows.pot_window import compute_pot_windows

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
    """
    Set the time window for various analysis methods based on the total number of rows and method-specific parameters.

    ## Overloaded Signatures
    ------------------------
    1. For method 'AE':
        * total_rows: int
        * method: "AE"
        * analysis_type: "historical" or "real-time"
        * encoded: typing.List[float]

    2. For method 'BM':
        * total_rows: int
        * method: "BM"
        * analysis_type: "historical" or "real-time"
        * block_size: float (default 365.2425)

    3. For method 'DBSCAN':
        * total_rows: int
        * method: "AE"
        * analysis_type: "historical" or "real-time"
        * total_cluster: int

    4. For method 'ISOF':
        * total_rows: int
        * method: "BM"
        * analysis_type: "historical" or "real-time"
        * isolated: typing.List[float]

    5. For method 'MAD':
        * total_rows: int
        * method: "AE"
        * analysis_type: "historical" or "real-time"
        * medians: typing.List[float],

    6. For method 'POT':
        * total_rows: int
        * method: "BM"
        * analysis_type: "historical" or "real-time"
        * t0_pct: float, default is 0.65
        * t1_pct: float, default is 0.25
        * t2_pct: float, default is 0.10

    7. For method '1CSVM':
        * total_rows: int
        * method: "AE"
        * analysis_type: "historical" or "real-time"
        * vectors: typing.List[float]

    8. For method 'ZS':
        * total_rows: int
        * method: "BM"
        * analysis_type: "historical" or "real-time"
        * upper: float
        * lower: float

    ## Parameters
    -------------
    total_rows : int
        The total number of rows in the time series data.

    method : typing.Literal["AE", "BM", "DBSCAN", "ISOF", "MAD", "1CSVM", "POT", "ZS"]
        The analysis method to be used.

    analysis_type : typing.Literal["historical", "real-time"]
        Type of analysis to be performed, either historical or real-time.

    **kwargs
        Additional keyword arguments specific to the selected method.

    ## Returns
    ----------
    typing.Tuple
        A tuple representing the time window parameters, specific to the chosen method.

    ## Example
    ----------
    >>> window = set_time_window(1000, "POT", "historical", t0_pct=0.7, t1_pct=0.2, t2_pct=0.1)
    >>> print(window)
    (700, 200, 100)

    ## Raises
    ---------
    NotImplementedError
        If the chosen method's specific time window setting isn't implemented yet.
    ValueError
        If the `method` argument is invalid or not supported.
    """

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
