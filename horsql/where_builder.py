import pandas as pd
import typing
from horsql.operators import Operator, Is, And, Or


def build_conditions(conditions: typing.Union[list, dict], condition="and"):
    clauses = []
    params = []

    if isinstance(conditions, And):
        condition = "and"

    if isinstance(conditions, Or):
        condition = "or"

    if isinstance(conditions, list):
        for item in conditions:
            local_clause, local_params = build_conditions(item)

            clauses.append(local_clause)
            params.extend(local_params)

    if isinstance(conditions, dict):
        local_clauses = []

        if isinstance(conditions, list):
            local_clause, local_params = build_conditions(conditions)

            local_clauses.append(local_clause)
            params.extend(local_params)

        for key, operator in conditions.items():
            if key in ["_"]:
                if not isinstance(operator, (And, Or)):
                    raise Exception("num sei num sei num sei")

                local_clause, local_params = build_conditions(
                    operator, operator.condition
                )

                local_clauses.append(local_clause)
                params.extend(local_params)

                continue

            if not isinstance(operator, Operator):
                if isinstance(operator, pd.DataFrame):
                    assert key in operator.columns, f"'{key}' não existe no DataFrame"
                    operator = operator.get(key)

                operator = Is(operator).custom

            local_clauses.append(f"{key} {operator.build()}")
            params.extend(operator.params())

        local_clause = f" {condition} ".join(local_clauses)

        clauses.append(local_clause)

    return f'({f" {condition} ".join(clauses)})', tuple(params)


def build_where(conditions):
    if not isinstance(conditions, (And, Or)):
        if isinstance(conditions, dict):
            conditions = And(**conditions)

    conditions, params = build_conditions(conditions)

    if conditions == "()":
        return "", []

    where = "WHERE " + conditions

    return where, params
