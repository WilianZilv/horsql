import sqlalchemy
import pandas as pd
import urllib.parse
from pathlib import Path
from typing import Literal, Union, List, Optional
import sys
from horsql.common import (
    Columns,
    columns_to_list,
    dataframe_tuples,
    format_groupby,
    generate_udt_types_map,
    sanitize_params,
    format_select,
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
        chain: Optional[Union[And, Or]] = None,
        **and_query,
    ):
        select = format_select(columns, distinct, min, max, sum, avg, count)

        columns = columns_to_list(columns)
        distinct = columns_to_list(distinct)

        groupby = format_groupby(columns, select)

        where, params = build_query(chain, and_query)

        SQL = f"{select}\nFROM\n{self.path()}\n{where}\n{groupby}"

        if self.order_sql is not None:
            SQL += f"\n{self.order_sql}"

        if self.limit_sql is not None:
            SQL += f"\n{self.limit_sql}"

        return self.db.fetch(SQL, params)

    def get_series(
        self,
        column: Optional[str] = None,
        distinct: Optional[str] = None,
        chain: Optional[Union[And, Or]] = None,
        **and_query,
    ) -> pd.Series:
        if column is None and distinct is None:
            raise Exception("'column' or 'distinct' must be provided")

        df = self.get(
            columns=column,
            distinct=distinct,
            chain=chain,
            **and_query,
        )

        return df[column or distinct]

    def get_columns(self) -> List[str]:
        series = self.db.information_schema.columns.get_series(
            column="column_name", table_schema=self.schema.name, table_name=self.name
        )

        return series.tolist()

    def insert(
        self,
        df: pd.DataFrame,
        on_conflict: Optional[Columns] = None,
        update: Optional[Columns] = None,
        commit: bool = True,
    ):
        self.db.insert(
            df,
            path=self.path(),
            on_conflict=on_conflict,
            update=update,
            commit=commit,
        )

    def update(
        self,
        df: pd.DataFrame,
        on_conflict: Columns,
        update: Columns,
        commit: bool = True,
    ):
        on_conflict = columns_to_list(on_conflict)
        update = columns_to_list(update)

        all_columns = on_conflict + update

        values = dataframe_tuples(df, all_columns)

        if values is None:
            return

        alias = f"{self.schema.name[0]}{self.name}"

        udt_types_map = generate_udt_types_map(self.db, self)

        set_sql = ", ".join([f"{x} = df.{x}{udt_types_map.get(x, '')}" for x in update])

        values_sql = ", ".join(["%s"] * len(values))
        values_columns_sql = ", ".join(all_columns)
        condition_sql = " AND ".join([f"{alias}.{x} = df.{x}" for x in on_conflict])

        SQL = f"UPDATE {self.schema.name}.{self.name} {alias}\n"
        SQL += f"SET {set_sql}\n"
        SQL += f"FROM (VALUES {values_sql}) AS df ({values_columns_sql})\n"
        SQL += f"WHERE {condition_sql}"

        self.db.execute(SQL, values)

        if not commit:
            return

        self.db.commit()

    def delete(
        self,
        commit=True,
        chain: Optional[Union[And, Or]] = None,
        **and_query,
    ):
        where, params = build_query(chain, and_query)

        SQL = f"DELETE FROM {self.path()} {where}"

        self.db.execute(SQL, params)

        if not commit:
            return

        self.db.commit()

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

    def execute(self, sql: str, params: Optional[tuple] = None):
        sql = self.mogrify(sql, params)
        self.cur.execute(sql)

    def commit(self):
        self.con.commit()

    def mogrify(self, query: str, params=None):
        if not hasattr(self.cur, "mogrify"):
            raise MogrifyNotAvailable("mogrify not available")

        params = sanitize_params(params)

        sql = self.cur.mogrify(query.strip(), params).decode("utf-8")
        sql = sql.replace("%", "%%")
        sql = Column.destroy(sql)
        return sql

    def fetch(self, sql: str, params: Union[tuple, list, None] = None):
        sql = self.mogrify(sql, params)
        return pd.read_sql(sql, self.engine)

    def insert(
        self,
        df,
        path: str,
        on_conflict: Optional[Columns] = None,
        update: Optional[Columns] = None,
        commit=True,
    ):
        rows = dataframe_tuples(df)

        if rows is None:
            return

        cols = ",".join(list(df.columns))
        params = ",".join(["%s" for _ in df.columns])

        query = f"INSERT INTO {path}({cols}) VALUES "
        query += ",".join([self.mogrify(f"({params})", row) for row in rows])

        if on_conflict is not None and update is not None:
            on_conflict = columns_to_list(on_conflict)
            update = columns_to_list(update)

            conflict_columns = ",".join(on_conflict)

            update_columns = [f"{x} = excluded.{x}" for x in update]
            update_columns = ",".join(update_columns)
            query += f" ON CONFLICT ({conflict_columns}) DO UPDATE SET {update_columns}"

        self.execute(query)

        if not commit:
            return

        self.commit()


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

    driver = dialects[dialect]

    application_name = ""
    if driver == "postgresql+psycopg2":
        if not app_name:
            app_path = Path(sys.argv[0])
            app_name = f"{app_path.parent.name}/{app_path.name}"

        application_name = f"?application_name={app_name}"

    password = urllib.parse.quote_plus(password)

    url = f"{driver}://{user}:{password}@{host}:{port}/{database}{application_name}"

    engine = sqlalchemy.create_engine(url=url, pool_size=pool_size, echo=echo)

    return Database(engine)
