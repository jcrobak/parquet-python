import ast
import os
import shutil
import pandas as pd
import pytest
import tempfile
import thriftpy


class ParquetException(Exception):
    """Generic Exception related to unexpected data format when
     reading parquet file."""
    pass


def sep_from_open(opener):
    if opener is default_open:
        return os.sep
    else:
        return '/'


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


def thrift_print(structure, offset=0):
    """
    Handy recursive text ouput for thrift structures
    """
    if not isinstance(structure, thriftpy.thrift.TPayload):
        return str(structure)
    s = str(structure.__class__) + '\n'
    for key in dir(structure):
        if key.startswith('_') or key in ['thrift_spec', 'read', 'write',
                                          'default_spec']:
            continue
        s = s + ' ' * offset + key + ': ' + thrift_print(getattr(structure, key)
                                                         , offset+2) + '\n'
    return s
thriftpy.thrift.TPayload.__str__ = thrift_print
thriftpy.thrift.TPayload.__repr__ = thrift_print


def thrift_copy(structure):
    """
    Recursively copy a thriftpy structure
    """
    base = structure.__class__()
    for key in dir(structure):
        if key.startswith('_') or key in ['thrift_spec', 'read', 'write',
                                          'default_spec']:
            continue
        val = getattr(structure, key)
        if isinstance(val, list):
            setattr(base, key, [thrift_copy(item)
                                if isinstance(item, thriftpy.thrift.TPayload)
                                else item for item in val])
        elif isinstance(val, thriftpy.thrift.TPayload):
            setattr(base, key, thrift_copy(val))
        else:
            setattr(base, key, val)
    return base


def index_like(index):
    """
    Does index look like a default range index?
    """
    return not (isinstance(index, pd.RangeIndex) and
                index._start == 0 and
                index._stop == len(index) and
                index._step == 1 and index.name is None)
