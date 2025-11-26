import sqlite3

import pandas as pd

from ..qcutils import ReadoutMode


def get_bias_data(dbfile: str, mode: ReadoutMode | None = None) -> pd.DataFrame:
    if mode is None:
        query = "SELECT * from bias"
    else:
        query = f"SELECT * from bias WHERE {mode.query_string()}"
    with sqlite3.connect(dbfile) as conn:
        data = pd.read_sql_query(query, conn)
    # drop SQL index column
    return data.drop("index", axis=1)


def add_bias_data(dbfile: str, df: pd.DataFrame, row: dict) -> None:
    df.loc[len(df)] = row
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("bias", conn, if_exists="replace")


def get_gain_data(dbfile: str, mode: ReadoutMode | None = None) -> pd.DataFrame:
    if mode is None:
        query = "SELECT * from gain"
    else:
        query = f"SELECT * from gain WHERE {mode.query_string()}"
    with sqlite3.connect(dbfile) as conn:
        data = pd.read_sql_query(query, conn)
    # drop SQL index column
    return data.drop("index", axis=1)


def add_gain_data(dbfile: str, df: pd.DataFrame, row: dict) -> None:
    df.loc[len(df)] = row
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("gain", conn, if_exists="replace")
