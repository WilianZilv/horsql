import sqlalchemy
import pandas as pd
import numpy as np
import urllib.parse
from typing import Literal, Union, List, Optional
import sys
from horsql.common import (
    Columns,
    dataframe_tuples,
    generate_udt_types_map,
    sanitize_params,
    format_columns,
)
from horsql.query_builder import build_query
from horsql.operators import Column, And, Or


class MogrifyNotAvailable(Exception):
    ...


class Table:
    order_sql: Optional[str] = None
    limit_sql: Optional[str] = None

    def __init__(self, db: "Database", schema: "Schema", name: str):
        self.schema = schema
        self.name = name
        self.db = db

    def path(self):
        return f"{self.schema.name}.{self.name}"

    def get(
        self,
        columns: Optional[Columns] = None,
        distinct: Optional[Columns] = None,
        min: Optional[Columns] = None,
        max: Optional[Columns] = None,
        sum: Optional[Columns] = None,
        avg: Optional[Columns] = None,
        count: Optional[Columns] = None,
        where: Optional[Union[And, Or]] = None,
        **query,
    ):
        columns = format_columns(columns, distinct, min, max, sum, avg, count)

        conditions = where
        if where is None:
            conditions = query

        return self.db.select(
            columns=columns, origin=self.path(), conditions=conditions, table=self
        )

    def get_series(
        self,
        column: Optional[str] = None,
        distinct: Optional[str] = None,
        where: Optional[Union[And, Or]] = None,
        **query,
    ) -> pd.Series:
        df = self.get(
            columns=column,
            distinct=distinct,
            where=where,
            **query,
        )

        return df[df.columns[0]]

    def get_columns(self) -> List[str]:
        series = self.db.information_schema.columns.get_series(
            column="column_name", table_schema=self.schema.name, table_name=self.name
        )

        return series.tolist()

    def create(self, df: pd.DataFrame, commit: bool = True):
        self.db.insert(df, self.schema.name, self.name, commit=commit)

    def update(
        self,
        df: pd.DataFrame,
        conflict_columns: list,
        update_columns: list,
        commit: bool = True,
    ):
        self.db.update(
            df, self.schema.name, self.name, conflict_columns, update_columns, commit
        )

    def upsert(
        self,
        df: pd.DataFrame,
        conflict_columns: list,
        update_columns: list,
        commit: bool = True,
    ):
        self.db.upsert(
            df, self.schema.name, self.name, conflict_columns, update_columns, commit
        )

    def delete(self, **kwargs):
        self.db.delete(self.path(), **kwargs)

    def limit(self, limit: int):
        self.limit_sql = f"LIMIT {limit}"

        return self

    def paginate(self, page: int, page_size: int):
        page = max([0, (page - 1)])

        self.limit_sql = f"LIMIT {page_size} OFFSET {page * page_size}"

        return self

    def order_by(
        self, columns: Union[List[str], str], ascending: Union[List[bool], bool] = []
    ):
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(ascending, bool):
            ascending = [ascending]

        assert (
            len(columns) == len(ascending) or len(ascending) == 0
        ), "O número de colunas e ordens devem ser iguais"

        if len(ascending) == 0:
            ascending = [True] * len(columns)

        ascending_sql = ["ASC" if x else "DESC" for x in ascending]

        instructions_sql = [
            f"{column} {order}" for column, order in zip(columns, ascending_sql)
        ]

        if self.limit_sql is not None:
            print("Warning: 'order' before 'limit' is a best practice")

        self.order_sql = f"ORDER BY {', '.join(instructions_sql)}"

        return self


class Schema:
    def __init__(self, db, name: str):
        self.db = db
        self.name = name

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return Table(db=self.db, schema=self, name=attr)


class Database:
    def __init__(self, engine):
        self.engine = engine
        self.connect()

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return Schema(db=self, name=attr)

    def connect(self):
        self.engine.connect()
        self.con = self.engine.raw_connection()
        self.cur = self.con.cursor()

    def commit(self):
        self.con.commit()

    def mogrify(self, query: str, params=None):
        if not hasattr(self.cur, "mogrify"):
            raise MogrifyNotAvailable("mogrify not available")

        sql = self.cur.mogrify(query.strip(), params).decode("utf-8")
        sql = sql.replace("%", "%%")
        sql = Column.destroy(sql)
        return sql

    def fetch(self, sql, params: Union[tuple, list, None] = None):
        params = sanitize_params(params)

        sql = self.mogrify(sql, params)
        return pd.read_sql(sql, self.engine)

    def delete(self, origin: str, **kwargs):
        where, params = build_query(kwargs)

        SQL = f"""
            DELETE
            FROM {origin}
            {where}
        """

        params = sanitize_params(params)

        SQL = self.mogrify(SQL, params)

        self.execute(SQL)

    def select(
        self,
        columns: Union[str, list],
        origin: str,
        groupby: Union[str, list] = "",
        conditions: Optional[Union[dict, list]] = None,
        table: Optional[Table] = None,
    ):
        if isinstance(columns, list):
            columns = ", ".join(columns)

        if not len(groupby) and ("(" in columns or ")" in columns):
            e_columns = enumerate(columns.split(", "))

            groupby = [str(i + 1) for i, c in e_columns if "(" not in c or ")" not in c]

        if isinstance(groupby, list):
            groupby = ", ".join(groupby)

        where, params = build_query(conditions)

        if isinstance(groupby, str):
            if len(groupby):
                groupby = f"GROUP BY {groupby}"

        if not len(columns.strip()):
            columns = "*"

        SQL = f"SELECT\n{columns}\nFROM\n{origin}\n{where}\n{groupby}"

        if table is not None:
            if table.order_sql is not None:
                SQL += f"\n{table.order_sql}"

            if table.limit_sql is not None:
                SQL += f"\n{table.limit_sql}"

        return self.fetch(SQL, params)

    def insert(self, df, schema: str, table: str, commit=True):
        return self.execute_values(df, schema, table, commit=commit)

    def upsert(
        self,
        df,
        schema: str,
        table: str,
        on_conflict: list,
        update: list,
        commit=True,
    ):
        return self.execute_values(
            df,
            schema,
            table,
            on_conflict=(on_conflict, update),
            commit=commit,
        )

    def update(
        self,
        df: pd.DataFrame,
        schema: str,
        table: str,
        primary_key: list,
        columns: list,
        commit=True,
    ):
        SQL = """--sql
            UPDATE {0}.{1} {2}
            SET {3}
            FROM (VALUES {4}) AS df ({5})
            WHERE {6}
        """
        alias = f"{schema[0]}{table[0]}"
        values = [
            tuple(x)
            for x in df[[*primary_key, *columns]].replace({np.nan: None}).to_numpy()
        ]

        udt_types = self.information_schema.columns.get(
            ["column_name", "udt_name"], table_schema=schema, table_name=table
        )
        udt_types.loc[:, "udt_name"] = "::" + udt_types["udt_name"]
        udt_types_map = udt_types.set_index("column_name")["udt_name"].to_dict()

        SQL = SQL.format(
            schema,
            table,
            alias,
            ", ".join([f"{x} = df.{x}{udt_types_map.get(x, '')}" for x in columns]),
            ", ".join(["%s"] * len(values)),
            ", ".join([*primary_key, *columns]),
            " AND ".join([f"{alias}.{x} = df.{x}" for x in primary_key]),
        )

        SQL = self.cur.mogrify(SQL, tuple(values)).decode()
        self.cur.execute(SQL)

        if not commit:
            return

        self.commit()

    def execute_values(
        self,
        df,
        schema: str,
        table: str,
        on_conflict: Optional[tuple] = None,
        commit=True,
    ):
        rows = dataframe_tuples(df)

        if rows is None:
            return

        cols = ",".join(list(df.columns))
        params = ",".join(["%s" for _ in df.columns])

        query = f"INSERT INTO {schema}.{table}({cols}) VALUES "
        query += ",".join([self.mogrify(f"({params})", row) for row in rows])

        if on_conflict:
            primary_key, update_columns = on_conflict

            if not isinstance(primary_key, list):
                primary_key = [primary_key]

            if not isinstance(update_columns, list):
                update_columns = [update_columns]

            primary_key = ",".join(primary_key)

            update_columns = [f"{x} = excluded.{x}" for x in update_columns]
            update_columns = ",".join(update_columns)
            query += f" ON CONFLICT ({primary_key}) DO UPDATE SET {update_columns}"

        self.execute(query)

        if not commit:
            return

        self.commit()


UNKOWN_PYTHON_APP = "Unknown Python App"


def connect(
    database: str,
    host: str,
    port: int,
    user: str,
    password: str,
    dialect: Literal["postgresql", "mysql"] = "postgresql",
    pool_size: int = 5,
    echo: bool = False,
    app_name: Optional[str] = None,
) -> Database:
    dialects = {"postgresql": "postgresql+psycopg2", "mysql": "mysql"}

    assert dialect in dialects.keys(), 'Available "dialects": %s' % list(
        dialects.keys()
    )

    driver = dialects[dialect]

    application_name = ""
    if driver == "postgresql+psycopg2":
        application_name = (
            f"?application_name={app_name or sys.argv[0][-16:] or UNKOWN_PYTHON_APP}"
        )

    password = urllib.parse.quote_plus(password)

    url = f"{driver}://{user}:{password}@{host}:{port}/{database}{application_name}"

    engine = sqlalchemy.create_engine(url=url, pool_size=pool_size, echo=echo)

    return Database(engine)
