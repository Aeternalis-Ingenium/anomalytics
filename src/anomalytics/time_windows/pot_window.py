import typing


def compute_pot_windows(
    total_rows: int,
    analysis_type: typing.Literal["historical", "real-time"],
    t0_pct: float = 0.65,
    t1_pct: float = 0.25,
    t2_pct: float = 0.10,
) -> typing.Tuple[int, int, int]:
    """
    Compute the time windows for Peak Over Threshold (POT) analysis based on total number of rows and specified percentages.

    ## Parameters
    -------------
    total_rows : int
        The total number of rows in the time series data.

    analysis_type : typing.Literal["historical", "real-time"]
        Type of analysis to be performed, either historical or real-time.

    t0_pct : float, default is 0.65
        Percentage of total rows allocated to the T0 time window.

    t1_pct : float, default is 0.25
        Percentage of total rows allocated to the T1 time window.

    t2_pct : float, default is 0.10
        Percentage of total rows allocated to the T2 time window (not used in real-time analysis).

    ## Returns
    ----------
    typing.Tuple[int, int, int]
        A tuple of integers representing the number of rows in each time window (T0, T1, T2).

    ## Example
    ----------
    >>> t0, t1, t2 = compute_pot_windows(1000, "historical")
    >>> print(t0, t1, t2)
    (650, 250, 100)

    ## Raises
    ---------
    ValueError
        If the percentages do not sum up correctly or if t0_pct is less than t1_pct and t2_pct.
    """

    if t0_pct - t1_pct < 0.0:
        raise ValueError("T0 time window needs to be bigger than T1 and T2, as a rule of thumb: t0 >= t1 > t2")
    if analysis_type == "real-time":
        if t0_pct + t1_pct != 1.0:
            raise ValueError(
                "In real-time analysis, the t2 time window will be the last row of the Time Series, hence `t0_pct` + `t1_pct` must equal to 1.0 (100%)"
            )
        t2 = 1
        total_rows = total_rows - t2
        t0 = int(t0_pct * total_rows)
        t1 = int(t1_pct * total_rows)
        uncounted_days = t0 + t1 - total_rows
    else:
        t0 = int(t0_pct * total_rows)
        t1 = int(t1_pct * total_rows)
        t2 = int(t2_pct * total_rows)
        uncounted_days = t0 + t1 + t2 - total_rows
    if uncounted_days < 0:
        t1 += abs(uncounted_days)
    elif uncounted_days > 0:
        t1 -= uncounted_days
    return (t0, t1, t2)
