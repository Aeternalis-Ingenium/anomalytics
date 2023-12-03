import typing


def compute_pot_windows(
    total_rows: int,
    analysis_type: typing.Literal["historical", "real-time"],
    t0_pct: float = 0.65,
    t1_pct: float = 0.25,
    t2_pct: float = 0.10,
) -> typing.Tuple[int, int, int]:
    if analysis_type == "real-time":
        if ((t0_pct + t1_pct > 1.0 or t0_pct + t1_pct == 1.0)) and (
            (t0_pct - t1_pct != 0.0) or (t1_pct - t0_pct != 0.0)
        ):
            raise ValueError(
                "In real-time analysis, the t2 time window will be the last row of the Time Series. Hence `t0_pct` + `t1_pct` must equal to 1.0 (100%)."
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
