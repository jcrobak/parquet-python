
from itertools import product
import numpy as np
import os
import pandas as pd
from fastparquet import ParquetFile
from fastparquet import write
from fastparquet import writer, encoding
import pytest
import shutil
import tempfile

from fastparquet.test.test_read import s3

TEST_DATA = "test-data"


@pytest.fixture()
def sql():
    pyspark = pytest.importorskip("pyspark")
    sc = pyspark.SparkContext.getOrCreate()
    sql = pyspark.SQLContext(sc)
    return sql


def test_uvarint():
    values = np.random.randint(0, 15000, size=100)
    o = encoding.Numpy8(np.zeros(30, dtype=np.uint8))
    for v in values:
        o.loc = 0
        writer.encode_unsigned_varint(v, o)
        o.loc = 0
        out = encoding.read_unsigned_var_int(o)
        assert v == out


def test_bitpack():
    for _ in range(10):
        values = np.random.randint(0, 15000, size=np.random.randint(10, 100),
                                   dtype=np.int32)
        width = encoding.width_from_max_int(values.max())
        o = encoding.Numpy8(np.zeros(900, dtype=np.uint8))
        writer.encode_bitpacked(values, width, o)
        o.loc = 0
        head = encoding.read_unsigned_var_int(o)
        out = encoding.Numpy32(np.zeros(300, dtype=np.int32))
        encoding.read_bitpacked(o, head, width, out)
        assert (values == out.so_far()[:len(values)]).all()
        assert out.so_far()[len(values):].sum() == 0  # zero padding
        assert out.loc - len(values) < 8


def test_length():
    lengths = np.random.randint(0, 15000, size=100)
    o = encoding.Numpy8(np.zeros(900, dtype=np.uint8))
    for l in lengths:
        o.loc = 0
        writer.write_length(l, o)
        o.loc = 0
        out = encoding.read_length(o)
        assert l == out


def test_rle_bp():
    for _ in range(10):
        values = np.random.randint(0, 15000, size=np.random.randint(10, 100),
                                   dtype=np.int32)
        out = encoding.Numpy32(np.empty(len(values) + 5, dtype=np.int32))
        o = encoding.Numpy8(np.zeros(900, dtype=np.uint8))
        width = encoding.width_from_max_int(values.max())

        # without length
        writer.encode_rle_bp(values, width, o)
        l = o.loc
        o.loc = 0

        encoding.read_rle_bit_packed_hybrid(o, width, length=l, o=out)
        assert (out.so_far()[:len(values)] == values).all()


@pytest.yield_fixture()
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("scheme,partitions,comp",
                         product(('simple', 'hive'),
                                 ([0], [0, 500]),
                                 (None, 'GZIP', 'SNAPPY')))
def test_pyspark_roundtrip(tempdir, scheme, partitions, comp, sql):
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32),
                         'i64': np.arange(1000, dtype=np.int64),
                         'f': np.arange(1000, dtype=np.float64),
                         'bhello': np.random.choice([b'hello', b'you',
                            b'people'], size=1000).astype("O")})

    data['hello'] = data.bhello.str.decode('utf8')
    data.loc[100, 'f'] = np.nan
    data['bcat'] = data.bhello.astype('category')
    data['cat'] = data.hello.astype('category')

    fname = os.path.join(tempdir, 'test.parquet')
    write(fname, data, file_scheme=scheme, partitions=partitions,
          compression=comp)

    df = sql.read.parquet(fname)
    ddf = df.toPandas()
    for col in data:
        assert (ddf[col] == data[col])[~ddf[col].isnull()].all()


def test_roundtrip_s3(s3):
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32),
                         'i64': np.arange(1000, dtype=np.int64),
                         'f': np.arange(1000, dtype=np.float64),
                         'bhello': np.random.choice([b'hello', b'you',
                            b'people'], size=1000).astype("O")})
    data['hello'] = data.bhello.str.decode('utf8')
    data['bcat'] = data.bhello.astype('category')
    data.loc[100, 'f'] = np.nan
    data['cat'] = data.hello.astype('category')
    noop = lambda x: True
    myopen = lambda f: s3.open(f, 'wb')
    write(TEST_DATA+'/temp_parq', data, file_scheme='hive', partitions=[0, 500],
          open_with=myopen, mkdirs=noop)
    myopen = s3.open
    pf = ParquetFile(TEST_DATA+'/temp_parq', open_with=myopen)
    df = pf.to_pandas(categories=['cat', 'bcat'])
    for col in data:
        assert (df[col] == data[col])[~df[col].isnull()].all()


@pytest.mark.parametrize('scheme,partitions,comp',
                         product(('simple', 'hive'),
                                 ([0], [0, 500]),
                                 (None, 'GZIP', 'SNAPPY')))
def test_roundtrip(tempdir, scheme, partitions, comp):
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32),
                         'i64': np.arange(1000, dtype=np.int64),
                         'f': np.arange(1000, dtype=np.float64),
                         'bhello': np.random.choice([b'hello', b'you',
                            b'people'], size=1000).astype("O")})
    data['a'] = np.array([b'a', b'b', b'c', b'd', b'e']*200, dtype="S1")
    data['aa'] = data['a'].map(lambda x: 2*x).astype("S2")
    data['hello'] = data.bhello.str.decode('utf8')
    data['bcat'] = data.bhello.astype('category')
    data['cat'] = data.hello.astype('category')
    fname = os.path.join(tempdir, 'test.parquet')
    write(fname, data, file_scheme=scheme, partitions=partitions,
          compression=comp)

    r = ParquetFile(fname)

    df = r.to_pandas()

    assert data.cat.dtype == 'category'

    for col in r.columns:
        assert (df[col] == data[col]).all()


@pytest.mark.parametrize('scheme', ('simple', 'hive'))
def test_roundtrip_complex(tempdir, scheme,):
    import datetime
    data = pd.DataFrame({'ui32': np.arange(1000, dtype=np.uint32),
                         'i16': np.arange(1000, dtype=np.int16),
                         'f16': np.arange(1000, dtype=np.float16),
                         'dicts': [{'oi': 'you'}] * 1000,
                         't': [datetime.datetime.now()] * 1000,
                         'td': [datetime.timedelta(seconds=1)] * 1000,
                         'bool': np.random.choice([True, False], size=1000)
                         })
    data.loc[100, 't'] = None

    fname = os.path.join(tempdir, 'test.parquet')
    write(fname, data, file_scheme=scheme)

    r = ParquetFile(fname)

    df = r.to_pandas()
    for col in r.columns:
        assert (df[col] == data[col])[~data[col].isnull()].all()


def test_write_with_dask(tempdir):
    dd = pytest.importorskip('dask.dataframe')
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32),
                         'i64': np.arange(1000, dtype=np.int64),
                         'f': np.arange(1000, dtype=np.float64),
                         'bhello': np.random.choice([b'hello', b'you',
                            b'people'], size=1000).astype("O")})
    data['a'] = np.array([b'a', b'b', b'c', b'd', b'e']*200, dtype="S1")
    data['aa'] = data['a'].map(lambda x: 2*x).astype("S2")
    data['hello'] = data.bhello.str.decode('utf8')
    data['bcat'] = data.bhello.astype('category')
    data['cat'] = data.hello.astype('category')

    df = dd.from_pandas(data, chunksize=200)
    writer.dask_dataframe_to_parquet(tempdir, df)

    r = ParquetFile(tempdir)

    df = r.to_pandas()
    for col in r.columns:
        assert (df[col] == data[col]).all()


@pytest.mark.skip()
def test_nulls_roundtrip(tempdir):
    fname = os.path.join(tempdir, 'temp.parq')
    data = pd.DataFrame({'o': np.random.choice(['hello', 'world', None],
                                               size=1000)})
    data['cat'] = data['o'].astype('category')
    writer.write(fname, data)

    r = ParquetFile(fname)
    df = r.to_pandas()
    for col in r.columns:
        assert (df[col] == data[col])[~data[col].isnull()].all()
        assert (data[col].isnull() == df[col].isnull()).all()


@pytest.mark.skip()
def test_write_delta(tempdir):
    fname = os.path.join(tempdir, 'temp.parq')
    data = pd.DataFrame({'i1': np.arange(10, dtype=np.int32) + 2,
                         'i2': np.cumsum(np.random.randint(
                                 0, 5, size=10)).astype(np.int32) + 2})
    writer.write(fname, data, encoding="DELTA_BINARY_PACKED")

    df = sql.read.parquet(fname)
    ddf = df.toPandas()
    for col in data:
        assert (ddf[col] == data[col])[~ddf[col].isnull()].all()
