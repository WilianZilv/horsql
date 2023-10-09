from typing import List, Optional, Union
import pandas as pd
import numpy as np

Columns = Union[str, List[str]]
ColumnsRenames = Union[str, List[Union[str, tuple]]]


def is_iterable(x):
    return isinstance(x, (pd.Series, np.ndarray, list, tuple))


def sanitize_params(params: Union[list, tuple, None]):
    if params is None:
        return None

    if isinstance(params, list):
        params = tuple(params)

    if not isinstance(params, tuple):
        params = (params,)

    if not len(params):
        return None

    params = tuple([tuple(x) if is_iterable(x) else x for x in params])

    return params


def columns_to_list(value: Optional[Columns], apply_renames: bool = True):
    if value is None:
        return []

    if isinstance(value, str):
        value = [x.strip() for x in value.split(",")]

    value = list(value)

    def renames(x: Union[str, tuple]):
        if isinstance(x, tuple):
            if apply_renames:
                assert len(x) == 2, 'Input esperado: ("column_name", "renamed_column")'

                return f'{x[0]} "{x[1]}"'
            return x[0]

        return x

    return list(map(renames, value))


def columns_to_agg(agg: str, columns: Optional[Columns]):
    _columns = columns_to_list(columns, apply_renames=False)

    if _columns is None:
        return _columns

    def agg_renames(x):
        column = x
        rename = column

        if isinstance(column, tuple):
            assert len(column) == 2, 'Input esperado: ("column_name", "renamed_column")'
            column, rename = column

        return f"{agg}({column}) {rename}"

    result = list(map(agg_renames, _columns))

    return result


def format_select(
    columns: Optional[Columns] = None,
    distinct: Optional[Columns] = None,
    min: Optional[Columns] = None,
    max: Optional[Columns] = None,
    sum: Optional[Columns] = None,
    avg: Optional[Columns] = None,
    count: Optional[Columns] = None,
):
    columns = columns_to_list(columns)
    distinct = columns_to_list(distinct)

    if len(distinct) and len(columns):
        raise Exception("use 'columns' or 'distinct', not both")

    if len(distinct):
        columns = []

    _min = columns_to_agg("min", min)
    _max = columns_to_agg("max", max)
    _sum = columns_to_agg("sum", sum)
    _avg = columns_to_agg("avg", avg)
    _count = columns_to_agg("count", count)

    _columns = ", ".join([*distinct, *columns, *_min, *_max, *_sum, *_avg, *_count])

    if len(distinct):
        _columns = "DISTINCT " + _columns

    if not len(_columns):
        _columns = "*"

    return f"SELECT\n{_columns}"


def dataframe_tuples(df: pd.DataFrame, columns: Optional[Union[str, list]] = None):
    if not len(df):
        return None

    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        columns = df.columns.tolist()

    nan = {np.nan: None}
    df = df.astype(object).replace(nan).replace(nan)

    return tuple([tuple(x) for x in df[columns].to_numpy()])


def generate_udt_types_map(db, table):
    udt_types = db.information_schema.columns.get(
        ["column_name", "udt_name"],
        table_schema=table.schema.name,
        table_name=table.name,
    )
    udt_types.loc[:, "udt_name"] = "::" + udt_types["udt_name"]
    return udt_types.set_index("column_name")["udt_name"].to_dict()


def format_groupby(columns: Columns, select: str):
    groupby = ""

    if len(columns) != len(select.split(",")):
        groupby_keys = list(range(1, len(columns) + 1))
        groupby_keys = list(map(str, groupby_keys))

        if len(groupby_keys):
            groupby = "GROUP BY "
            groupby += ", ".join(groupby_keys)

    return groupby
