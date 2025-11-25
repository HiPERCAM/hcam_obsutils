import sqlite3

import pandas as pd


def get_bias_data(dbfile):
    with sqlite3.connect(dbfile) as conn:
        data = pd.read_sql_query("SELECT * from bias", conn)
    # drop SQL index column
    return data.drop("index", axis=1)


def add_bias_data(dbfile, df, row):
    df.loc[len(df)] = row
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("bias", conn, if_exists="replace")


def get_gain_data(dbfile):
    with sqlite3.connect(dbfile) as conn:
        data = pd.read_sql_query("SELECT * from gain", conn)
    # drop SQL index column
    return data.drop("index", axis=1)


def add_gain_data(dbfile, df, row):
    df.loc[len(df)] = row
    with sqlite3.connect(dbfile) as conn:
        df.to_sql("gain", conn, if_exists="replace")
        df.to_sql("gain", conn, if_exists="replace")
