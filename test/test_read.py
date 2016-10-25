"""test_read_support.py - unit and integration tests for reading parquet data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
from itertools import product
import json
import numpy as np
import os
import pandas as pd
import sys
import tempfile
import unittest

import pandas as pd
import pytest

import parquet

TEST_DATA = "test-data"


def test_header_magic_bytes():
    """Test reading the header magic bytes."""
    f = io.BytesIO(b"PAR1_some_bogus_data")
    with pytest.raises(parquet.ParquetException):
        p = parquet.ParquetFile(f, verify=True)


def test_read_footer():
    """Test reading the footer."""
    p = parquet.ParquetFile(os.path.join(TEST_DATA, "nation.impala.parquet"))
    snames = {"schema", "n_regionkey", "n_name", "n_nationkey", "n_comment"}
    assert {s.name for s in p.schema} == snames
    assert set(p.columns) == snames - {"schema"}

files = [os.path.join(TEST_DATA, p) for p in
         ["gzip-nation.impala.parquet", "nation.dict.parquet",
          "nation.impala.parquet", "nation.plain.parquet",
          "snappy-nation.impala.parquet"]]
csvfile = os.path.join(TEST_DATA, "nation.csv")
cols = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]
expected = pd.read_csv(csvfile, delimiter="|", index_col=0, names=cols)


def test_read_s3():
    s3fs = pytest.importorskip('s3fs')
    s3 = s3fs.S3FileSystem()
    myopen = s3.open
    pf = parquet.ParquetFile('MDtemp/split/_metadata', open_with=myopen)
    df = pf.to_pandas()
    assert df.shape == (2000, 3)
    assert (df.cat.value_counts() == [1000, 1000]).all()


def test_read_dask():
    pytest.importorskip('dask')
    s3fs = pytest.importorskip('s3fs')
    s3 = s3fs.S3FileSystem()
    myopen = s3.open
    pf = parquet.ParquetFile('MDtemp/split/_metadata', open_with=myopen)
    df = pf.to_dask()
    out = df.compute()
    assert out.shape == (2000, 3)
    assert (out.cat.value_counts() == [1000, 1000]).all()


@pytest.mark.parametrize("parquet_file", files)
def test_file_csv(parquet_file):
    """Test the various file times
    """
    p = parquet.ParquetFile(parquet_file)
    data = p.to_pandas()
    if 'comment_col' in data.columns:
        mapping = {'comment_col': "n_comment", 'name': 'n_name',
                   'nation_key': 'n_nationkey', 'region_key': 'n_regionkey'}
        data.columns = [mapping[k] for k in data.columns]
    data.set_index('n_nationkey', inplace=True)

    # FIXME: in future, reader will return UTF8 strings
    for col in cols[1:]:
        if isinstance(data[col][0], bytes):
            data[col] = data[col].str.decode('utf8')
        assert (data[col] == expected[col]).all()


def test_null_int():
    """Test reading a file that contains null records."""
    p = parquet.ParquetFile(os.path.join(TEST_DATA, "test-null.parquet"))
    data = p.to_pandas()
    expected = pd.DataFrame([{"foo": 1, "bar": 2}, {"foo": 1, "bar": None}])
    for col in data:
        assert (data[col] == expected[col])[~expected[col].isnull()].all()
        assert sum(data[col].isnull()) == sum(expected[col].isnull())


def test_converted_type_null():
    """Test reading a file that contains null records for a plain column that
     is converted to utf-8."""
    p = parquet.ParquetFile(os.path.join(TEST_DATA,
                                         "test-converted-type-null.parquet"))
    data = p.to_pandas()
    expected = pd.DataFrame([{"foo": "bar"}, {"foo": None}])
    for col in data:
        if isinstance(data[col][0], bytes):
            # Remove when re-implemented converted types
            data[col] = data[col].str.decode('utf8')
        assert (data[col] == expected[col])[~expected[col].isnull()].all()
        assert sum(data[col].isnull()) == sum(expected[col].isnull())


def test_null_plain_dictionary():
    """Test reading a file that contains null records for a plain dictionary
     column."""
    p = parquet.ParquetFile(os.path.join(TEST_DATA,
                                         "test-null-dictionary.parquet"))
    data = p.to_pandas()
    expected = pd.DataFrame([{"foo": None}] + [{"foo": "bar"},
                             {"foo": "baz"}] * 3)
    for col in data:
        if isinstance(data[col][1], bytes):
            # Remove when re-implemented converted types
            data[col] = data[col].str.decode('utf8')
        assert (data[col] == expected[col])[~expected[col].isnull()].all()
        assert sum(data[col].isnull()) == sum(expected[col].isnull())


def test_dir_partition():
    """Test creation of categories from directory structure"""
    x = np.arange(2000)
    df = pd.DataFrame({
        'num': x,
        'cat': pd.Series(np.array(['fred', 'freda'])[x%2], dtype='category'),
        'catnum': pd.Series(np.array([1, 2, 3])[x%3], dtype='category')})
    pf = parquet.ParquetFile(os.path.join(TEST_DATA, "split"))
    out = pf.to_pandas()
    for cat, catnum in product(['fred', 'freda'], [1, 2, 3]):
        assert (df.num[(df.cat==cat) & (df.catnum==catnum)].tolist()) ==\
               out.num[(out.cat==cat) & (out.catnum==catnum)].tolist()
    assert out.cat.dtype == 'category'
    assert out.catnum.dtype == 'category'
    assert out.catnum.cat.categories.dtype == 'int64'
