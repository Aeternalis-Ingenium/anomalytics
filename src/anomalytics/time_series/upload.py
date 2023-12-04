import typing

import pandas as pd


def create_ts_from_csv(
    path_to_file: str,
    index_col: int = 0,
    names: typing.List[str] | None = None,
    header: int | typing.Sequence[int] | typing.Literal["infer"] | None = None,
    sep: str | None = ",",
) -> pd.Series:
    """
    Create a Pandas Series object from a CSV file.

    ## Parameters
    -------------
    path_to_file : str
        Path to the CSV file.

    index_col : int, default is 0
        Column to use as the row labels of the DataFrame.

    names : typing.List[str] | None, default is None
        List of column names to use. If file contains no header row, then you should explicitly pass `header=None`.

    header : int, typing.Sequence[int], typing.Literal["infer"], or None, default is "infer"
        Row number(s) to use as the column names, and the start of the data.

    sep : str or None, default is ","
        Delimiter to use. If None, will try to automatically determine this.

    ## Returns
    ----------
    pd.Series
        A Pandas Series created from the CSV file.

    ## Example
    ----------
    >>> ts = create_ts_from_csv("data.csv")
    >>> ts.head()
    Date-Time
    2008-11-03 08:00:00   -0.282
    2008-11-03 09:00:00   -0.368
    2008-11-03 10:00:00   -0.400
    2008-11-03 11:00:00   -0.320
    2008-11-03 12:00:00   -0.155
    Name: Example Dataset, dtype: float64
    """

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
    """
    Create a Pandas Series object from an Excel (.xlsx) file.

    ## Parameters
    -------------
    path_to_file : str
        Path to the Excel file.

    index_col : int, default is 0
        Column to use as the row labels of the DataFrame.

    names : typing.List[str] | None, default is None
        List of column names to use.

    header : int, typing.Sequence[int], typing.Literal["infer"], or None, default is "infer"
        Row number(s) to use as the column names, and the start of the data.

    sheet_name : str, int, or None, default is 0
        Name or index of the sheet.

    ## Returns
    ----------
    pd.Series
        A Pandas Series created from the Excel file.

    ## Example
    ----------
    >>> ts = create_ts_from_xlsx("data.xlsx")
    >>> ts.head()
    Date-Time
    2008-11-03 08:00:00   -0.282
    2008-11-03 09:00:00   -0.368
    2008-11-03 10:00:00   -0.400
    2008-11-03 11:00:00   -0.320
    2008-11-03 12:00:00   -0.155
    Name: Example Dataset, dtype: float64
    """

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
    """
    Read a file and create a Pandas Series object, supporting both CSV and Excel formats.

    ## Parameters
    -------------
    path_to_file : str
        Path to the file.

    file_type : typing.Literal["csv", "xlsx"]
        Type of the file ('csv' or 'xlsx').

    index_col : int, default is 0
        Column to use as the row labels of the DataFrame.

    names : typing.List[str] | None, default is None
        List of column names to use.

    header : int, typing.Sequence[int], typing.Literal["infer"], or None, default is "infer"
        Row number(s) to use as the column names, and the start of the data.

    **kwargs
        Additional keyword arguments to pass to `create_ts_from_csv` or `create_ts_from_xlsx`.

    ## Returns
    ----------
    pd.Series
        A Pandas Series created from the specified file.

    ## Example
    ----------
    >>> ts = read_ts("data.xlsx", "xlsx")
    >>> ts.head()
    Date-Time
    2008-11-03 08:00:00   -0.282
    2008-11-03 09:00:00   -0.368
    2008-11-03 10:00:00   -0.400
    2008-11-03 11:00:00   -0.320
    2008-11-03 12:00:00   -0.155
    Name: Example Dataset, dtype: float64
    >>> ts = read_ts("data.csv", "csv")
    >>> ts.head()
    Date-Time
    2008-11-03 08:00:00   -0.282
    2008-11-03 09:00:00   -0.368
    2008-11-03 10:00:00   -0.400
    2008-11-03 11:00:00   -0.320
    2008-11-03 12:00:00   -0.155
    Name: Example Dataset, dtype: float64

    ## Raises
    ---------
    ValueError
        If the `path_to_file` is None or if the `file_type` is not one of the supported formats ('csv', 'xlsx').
    """

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
