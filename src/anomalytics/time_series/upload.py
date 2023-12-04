import typing

import pandas as pd


def create_ts_from_csv(
    path_to_file: str,
    index_col: int = 0,
    names: typing.List[str] | None = None,
    header: int | typing.Sequence[int] | typing.Literal["infer"] | None = None,
    sep: str | None = ",",
) -> pd.Series:
    return pd.read_csv(
        filepath_or_buffer=path_to_file, header=header, index_col=index_col, sep=sep, parse_dates=True, names=names
    ).squeeze()


def create_ts_from_xlsx(
    path_to_file: str,
    index_col: int = 0,
    names: typing.List[str] | None = None,
    header: int | typing.Sequence[int] | typing.Literal["infer"] | None = None,
    sheet_name: str | int | None = 0,
) -> pd.Series:
    return pd.read_excel(
        io=path_to_file, header=header, index_col=index_col, sheet_name=sheet_name, parse_dates=True, names=names
    ).squeeze()


@typing.overload
def read_ts(
    path_to_file: str,
    file_type: typing.Literal["csv"],
    index_col: int = 0,
    names: typing.List[str] | None = None,
    header: int | typing.Sequence[int] | typing.Literal["infer"] | None = None,
    *,
    sep: str | None = ",",
) -> pd.Series:
    ...


@typing.overload
def read_ts(
    path_to_file: str,
    file_type: typing.Literal["xlsx"],
    index_col: int = 0,
    names: typing.List[str] | None = None,
    header: int | typing.Sequence[int] | typing.Literal["infer"] | None = None,
    *,
    sheet_name: str | int | None = 0,
) -> pd.Series:
    ...


def read_ts(
    path_to_file: str,
    file_type: typing.Literal["csv", "xlsx"],
    index_col: int = 0,
    names: typing.List[str] | None = None,
    header: int | typing.Sequence[int] | typing.Literal["infer"] | None = None,
    **kwargs,
) -> pd.Series:
    if not path_to_file:
        raise ValueError("The argument for `path_to_file` can't be None.")

    if file_type == "csv":
        return create_ts_from_csv(
            path_to_file=path_to_file,
            index_col=index_col,
            names=names,
            header=header,
            **kwargs,
        )

    if file_type == "xlsx":
        return create_ts_from_xlsx(
            path_to_file=path_to_file,
            index_col=index_col,
            names=names,
            header=header,
            **kwargs,
        )

    raise ValueError("Invalid value for `file_type` argument, available: 'csv', 'xlsx'")
