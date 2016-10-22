
from itertools import product
import numpy as np
import os
import pandas as pd
from parquet import ParquetFile
from parquet import write
from parquet import writer, encoding
import pytest
import shutil
import tempfile
pyspark = pytest.importorskip("pyspark")
sc = pyspark.SparkContext.getOrCreate()
sql = pyspark.SQLContext(sc)


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
def test_pyspark_roundtrip(tempdir, scheme, partitions, comp):
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32),
                         'i64': np.arange(1000, dtype=np.int64),
                         'f': np.arange(1000, dtype=np.float64),
                         'bhello': np.random.choice([b'hello', b'you',
                            b'people'], size=1000).astype("O")})

    data['hello'] = data.bhello.str.decode('utf8')
    data['f'].iloc[100] = np.nan

    fname = os.path.join(tempdir, 'test.parquet')
    write(fname, data, file_scheme=scheme, partitions=partitions,
          compression=comp)

    df = sql.read.parquet(fname)
    ddf = df.toPandas()
    for col in data:
        assert (ddf[col] == data[col])[~ddf[col].isnull()].all()


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
    # data['cat'] = data.bhello.astype('category')
    fname = os.path.join(tempdir, 'test.parquet')
    write(fname, data, file_scheme=scheme, partitions=partitions,
          compression=comp)

    r = ParquetFile(fname)

    df = r.to_pandas()
    for col in r.columns:
        assert (df[col] == data[col]).all()
