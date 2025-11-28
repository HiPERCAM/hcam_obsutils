import sqlite3

import pandas as pd

from ..qcutils import ReadoutMode


def remove_duplicate_rows(df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that already exist in the database from the dataframe
    """
    df = pd.merge(df, other_df, how="left", indicator=True)
    df = df[df["_merge"] == "left_only"]
    df.drop(["_merge"], axis=1, inplace=True)
    return df


def create_zeropoint_table(dbfile: str, initial_row: dict) -> None:
    df = pd.DataFrame([initial_row])
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("zeropoint", conn, if_exists="replace")


def get_zeropoint_data(dbfile: str, band: str | None = None) -> pd.DataFrame:
    if band is None:
        query = "SELECT * from zeropoint"
    else:
        query = f"SELECT * from zeropoint WHERE band=='{band}'"
    with sqlite3.connect(dbfile) as conn:
        data = pd.read_sql_query(query, conn)
    # drop SQL index column
    return data.drop("index", axis=1)


def add_zeropoint_data(dbfile: str, df: pd.DataFrame, row: dict) -> None:
    old_df = df.copy()
    df.loc[len(df)] = row
    df = remove_duplicate_rows(df, old_df)
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("zeropoint", conn, if_exists="append")


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
    old_df = df.copy()
    df.loc[len(df)] = row
    df = remove_duplicate_rows(df, old_df)
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("bias", conn, if_exists="append")


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
    old_df = df.copy()
    df.loc[len(df)] = row
    df = remove_duplicate_rows(df, old_df)
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("gain", conn, if_exists="append")
