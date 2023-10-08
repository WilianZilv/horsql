import pandas as pd
from typing import Union
from horsql.operators import Operator, Is, And, Or


def build_conditions(chain: Union[And, Or]):
    clauses = []
    params = []

    condition = chain.condition

    for item in chain.chains:
        local_clause, local_params = build_conditions(item)

        clauses.append(local_clause)
        params.extend(local_params)

    local_clauses = []

    for key, operator in chain.value.items():
        if not isinstance(operator, Operator):
            if isinstance(operator, pd.DataFrame):
                assert key in operator.columns, f"'{key}' não existe no DataFrame"
                operator = operator.get(key)

            operator = Is(operator).custom

        local_clauses.append(f"{key} {operator.build()}")
        params.extend(operator.params())

    if len(local_clauses):
        local_clause = f" {condition} ".join(local_clauses)
        clauses.append(local_clause)

    return f'({f" {condition} ".join(clauses)})', tuple(params)


def build_query(conditions):
    if isinstance(conditions, dict):
        conditions = And(**conditions)

    conditions, params = build_conditions(conditions)

    if conditions == "()":
        return "", []

    where = "WHERE " + conditions

    return where, params
