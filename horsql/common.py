from typing import Any, Union
import pandas as pd
import numpy as np


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


def columns_to_list(value: Union[str, list, None], mount_renames: bool = True):
    if value is None:
        return []

    if isinstance(value, str):
        value = [x.strip() for x in value.split(",")]

    value = list(value)

    def renames(x):
        if isinstance(x, tuple) and mount_renames:
            assert len(x) == 2, 'Input esperado: ("column_name", "renamed_column")'

            return f'{x[0]} "{x[1]}"'

        return x

    value = list(map(renames, value))

    return value


def columns_to_agg(agg: str, columns: Any):
    _columns = columns_to_list(columns, mount_renames=False)

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


def format_columns(
    columns: Union[str, list, None] = None,
    distinct: Union[str, list, None] = None,
    min: Union[str, list, None] = None,
    max: Union[str, list, None] = None,
    sum: Union[str, list, None] = None,
    avg: Union[str, list, None] = None,
):
    columns = columns_to_list(columns)
    distinct = columns_to_list(distinct)

    if len(distinct):
        columns = []

    _min = columns_to_agg("min", min)
    _max = columns_to_agg("max", max)
    _sum = columns_to_agg("sum", sum)
    _avg = columns_to_agg("avg", avg)

    columns = ", ".join([*distinct, *columns, *_min, *_max, *_sum, *_avg])

    if len(distinct):
        columns = "DISTINCT " + columns

    return columns


def get_correct_conditions(where: Union[list, dict, None] = None, **query):
    conditions = where
    if where is None:
        conditions = query

    return conditions
