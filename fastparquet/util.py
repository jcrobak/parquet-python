import ast
import os
import shutil
import tempfile

import pandas as pd
import pytest


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


def default_open(f, mode='rb'):
    return open(f, mode)


def val_to_num(x):
    # What about ast.literal_eval?
    try:
        return ast.literal_eval(x)
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


@pytest.yield_fixture()
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


def ensure_bytes(s):
    if hasattr(s, 'encode'):
        return s.encode()
    else:
        return s
