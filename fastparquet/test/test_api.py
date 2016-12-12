
import os

import numpy as np
import pandas as pd
import pytest

from fastparquet.util import tempdir
from fastparquet import write, ParquetFile
from fastparquet.api import statistics, sorted_partitioned_columns

TEST_DATA = "test-data"


def test_statistics(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3],
                       'y': [1.0, 2.0, 1.0],
                       'z': ['a', 'b', 'c']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])

    p = ParquetFile(fn)

    s = statistics(p)
    expected = {'distinct_count': {'x': [None, None],
                                   'y': [None, None],
                                   'z': [None, None]},
                'max': {'x': [2, 3], 'y': [2.0, 1.0], 'z': ['b', 'c']},
                'min': {'x': [1, 3], 'y': [1.0, 1.0], 'z': ['a', 'c']},
                'null_count': {'x': [0, 0], 'y': [0, 0], 'z': [0, 0]}}

    assert s == expected


def test_logical_types(tempdir):
    df = pd.util.testing.makeMixedDataFrame()

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])

    p = ParquetFile(fn)

    s = statistics(p)

    assert isinstance(s['min']['D'][0], (np.datetime64, pd.tslib.Timestamp))


def test_empty_statistics(tempdir):
    p = ParquetFile(os.path.join(TEST_DATA, "nation.impala.parquet"))

    s = statistics(p)
    assert s == {'distinct_count': {'n_comment': [None],
                                    'n_name': [None],
                                    'n_nationkey': [None],
                                    'n_regionkey': [None]},
                  'max': {'n_comment': [None],
                          'n_name': [None],
                          'n_nationkey': [None],
                          'n_regionkey': [None]},
                  'min': {'n_comment': [None],
                          'n_name': [None],
                          'n_nationkey': [None],
                          'n_regionkey': [None]},
                  'null_count': {'n_comment': [None],
                                 'n_name': [None],
                                 'n_nationkey': [None],
                                 'n_regionkey': [None]}}


def test_sorted_row_group_columns(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])

    pf = ParquetFile(fn)

    result = sorted_partitioned_columns(pf)
    expected = {'x': {'min': [1, 3], 'max': [2, 4]},
                'z': {'min': ['a', 'c'], 'max': ['b', 'd']}}

    assert result == expected


def test_iter(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    df.index.name = 'index'

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2], write_index=True)
    pf = ParquetFile(fn)
    out = iter(pf.iter_row_groups(index='index'))
    d1 = next(out)
    pd.util.testing.assert_frame_equal(d1, df[:2])
    d2 = next(out)
    pd.util.testing.assert_frame_equal(d2, df[2:])
    with pytest.raises(StopIteration):
        next(out)


def test_attributes(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])
    pf = ParquetFile(fn)
    assert pf.columns == ['x', 'y', 'z']
    assert len(pf.row_groups) == 2
    assert pf.count == 4
    assert fn == pf.info['name']
    assert fn in str(pf)
    for col in df:
        assert pf.dtypes[col] == df.dtypes[col]
