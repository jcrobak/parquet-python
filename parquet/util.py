import os
import pandas as pd


class ParquetException(Exception):
    """Generic Exception related to unexpected data format when
     reading parquet file."""
    pass


def sep_from_open(opener):
    if opener in [default_open, default_openw]:
        return os.sep
    else:
        return '/'


def default_openw(f):
    return open(f, 'wb')


def default_mkdirs(f):
    os.makedirs(f, exist_ok=True)


def default_open(f):
    return open(f, 'rb')


def val_to_num(x):
    # What about ast.literal_eval?
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    try:
        return pd.to_datetime(x)
    except ValueError:
        pass
    try:
        return pd.to_timedelta(x)
    except:
        return x
