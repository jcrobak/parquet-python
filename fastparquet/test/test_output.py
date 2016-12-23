
import numpy as np
import os
import pandas as pd
import pandas.util.testing as tm
from fastparquet import ParquetFile
from fastparquet import write, parquet_thrift
from fastparquet import writer, encoding
import pytest
import shutil
import tempfile

from fastparquet.util import tempdir
from fastparquet.test.test_read import s3
from fastparquet.compression import compressions

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


@pytest.mark.parametrize('scheme', ['simple', 'hive'])
@pytest.mark.parametrize('row_groups', [[0], [0, 500]])
@pytest.mark.parametrize('comp', [None] + list(compressions))
def test_pyspark_roundtrip(tempdir, scheme, row_groups, comp, sql):
    if comp == 'BROTLI':
        pytest.xfail("spark doesn't support BROTLI compression")
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
    write(fname, data, file_scheme=scheme, row_group_offsets=row_groups,
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
    myopen = s3.open
    write(TEST_DATA+'/temp_parq', data, file_scheme='hive',
          row_group_offsets=[0, 500], open_with=myopen, mkdirs=noop)
    myopen = s3.open
    pf = ParquetFile(TEST_DATA+'/temp_parq', open_with=myopen)
    df = pf.to_pandas(categories=['cat', 'bcat'])
    for col in data:
        assert (df[col] == data[col])[~df[col].isnull()].all()


@pytest.mark.parametrize('scheme', ['simple', 'hive'])
@pytest.mark.parametrize('row_groups', [[0], [0, 500]])
@pytest.mark.parametrize('comp', [None, 'GZIP', 'SNAPPY'])
def test_roundtrip(tempdir, scheme, row_groups, comp):
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
    write(fname, data, file_scheme=scheme, row_group_offsets=row_groups,
          compression=comp)

    r = ParquetFile(fname)

    df = r.to_pandas()

    assert data.cat.dtype == 'category'

    for col in r.columns:
        assert (df[col] == data[col]).all()


def test_bad_coltype(tempdir):
    df = pd.DataFrame({'0': [1, 2], (0, 1): [3, 4]})
    fn = os.path.join(tempdir, 'temp.parq')
    with pytest.raises(ValueError) as e:
        write(fn, df)
        assert "tuple" in str(e)


@pytest.mark.parametrize('scheme', ('simple', 'hive'))
def test_roundtrip_complex(tempdir, scheme,):
    import datetime
    data = pd.DataFrame({'ui32': np.arange(1000, dtype=np.uint32),
                         'i16': np.arange(1000, dtype=np.int16),
                         'ui8': np.array([1, 2, 3, 4]*250, dtype=np.uint8),
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


@pytest.mark.parametrize('df', [
    pd.util.testing.makeMixedDataFrame(),
    pd.DataFrame({'x': pd.date_range('3/6/2012 00:00',
                  periods=10, freq='H', tz='Europe/London')}),
    pd.DataFrame({'x': pd.date_range('3/6/2012 00:00',
                  periods=10, freq='H', tz='Europe/Berlin')}),
    pd.DataFrame({'x': pd.date_range('3/6/2012 00:00',
                  periods=10, freq='H', tz='UTC')})
    ])
def test_datetime_roundtrip(tempdir, df, capsys):
    fname = os.path.join(tempdir, 'test.parquet')
    write(fname, df)

    r = ParquetFile(fname)
    out, err = capsys.readouterr()
    if 'x' in df and str(df.x.dtype.tz) == 'Europe/London':
        # warning happens first time only
        assert "UTC" in err

    df2 = r.to_pandas()
    if 'x' in df:
        df['x'] = df.x.dt.tz_convert(None)

    pd.util.testing.assert_frame_equal(df, df2, check_categorical=False)


def test_nulls_roundtrip(tempdir):
    fname = os.path.join(tempdir, 'temp.parq')
    data = pd.DataFrame({'o': np.random.choice(['hello', 'world', None],
                                               size=1000)})
    data['cat'] = data['o'].astype('category')
    writer.write(fname, data, has_nulls=['o', 'cat'])

    r = ParquetFile(fname)
    df = r.to_pandas()
    for col in r.columns:
        assert (df[col] == data[col])[~data[col].isnull()].all()
        assert (data[col].isnull() == df[col].isnull()).all()


def test_make_definitions_with_nulls():
    for _ in range(10):
        out = np.empty(1000, dtype=np.int32)
        o = encoding.Numpy32(out)
        data = pd.Series(np.random.choice([True, None],
                                          size=np.random.randint(1, 1000)))
        out, d2 = writer.make_definitions(data, False)
        i = encoding.Numpy8(np.fromstring(out, dtype=np.uint8))
        encoding.read_rle_bit_packed_hybrid(i, 1, length=None, o=o)
        out = o.so_far()[:len(data)]
        assert (out == ~data.isnull()).sum()


def test_make_definitions_without_nulls():
    for _ in range(100):
        out = np.empty(10000, dtype=np.int32)
        o = encoding.Numpy32(out)
        data = pd.Series([True] * np.random.randint(1, 10000))
        out, d2 = writer.make_definitions(data, True)

        l = len(data) << 1
        p = 1
        while l > 127:
            l >>= 7
            p += 1
        assert len(out) == 4 + p + 1  # "length", num_count, value

        i = encoding.Numpy8(np.fromstring(out, dtype=np.uint8))
        encoding.read_rle_bit_packed_hybrid(i, 1, length=None, o=o)
        out = o.so_far()
        assert (out == ~data.isnull()).sum()

    # class mock:
    #     def is_required(self, *args):
    #         return False
    #     def max_definition_level(self, *args):
    #         return 1
    #     def __getattr__(self, item):
    #         return None
    # halper, metadata = mock(), mock()


def test_empty_row_group(tempdir):
    fname = os.path.join(tempdir, 'temp.parq')
    data = pd.DataFrame({'o': np.random.choice(['hello', 'world'],
                                               size=1000)})
    writer.write(fname, data, row_group_offsets=[0, 900, 1800])
    pf = ParquetFile(fname)
    assert len(pf.row_groups) == 2


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


def test_int_rowgroups(tempdir):
    df = pd.DataFrame({'a': [1]*100})
    fname = os.path.join(tempdir, 'test.parq')
    writer.write(fname, df, row_group_offsets=30)
    r = ParquetFile(fname)
    assert [rg.num_rows for rg in r.row_groups] == [25, 25, 25, 25]
    writer.write(fname, df, row_group_offsets=33)
    r = ParquetFile(fname)
    assert [rg.num_rows for rg in r.row_groups] == [25, 25, 25, 25]
    writer.write(fname, df, row_group_offsets=34)
    r = ParquetFile(fname)
    assert [rg.num_rows for rg in r.row_groups] == [34, 34, 32]
    writer.write(fname, df, row_group_offsets=35)
    r = ParquetFile(fname)
    assert [rg.num_rows for rg in r.row_groups] == [34, 34, 32]


def test_groups_roundtrip(tempdir):
    df = pd.DataFrame({'a': np.random.choice(['a', 'b', None], size=1000),
                       'b': np.random.randint(0, 64000, size=1000),
                       'c': np.random.choice([True, False], size=1000)})
    writer.write(tempdir, df, partition_on=['a', 'c'], file_scheme='hive')

    r = ParquetFile(tempdir)
    assert r.columns == ['b']
    out = r.to_pandas()

    for i, row in out.iterrows():
        assert row.b in list(df[(df.a==row.a)&(df.c==row.c)].b)

    writer.write(tempdir, df, row_group_offsets=[0, 50], partition_on=['a', 'c'],
                 file_scheme='hive')

    r = ParquetFile(tempdir)
    assert r.count == sum(~df.a.isnull())
    assert len(r.row_groups) == 8
    out = r.to_pandas()

    for i, row in out.iterrows():
        assert row.b in list(df[(df.a==row.a)&(df.c==row.c)].b)


def test_empty_groupby(tempdir):
    df = pd.DataFrame({'a': np.random.choice(['a', 'b', None], size=1000),
                       'b': np.random.randint(0, 64000, size=1000),
                       'c': np.random.choice([True, False], size=1000)})
    df.loc[499:, 'c'] = True  # no False in second half
    writer.write(tempdir, df, partition_on=['a', 'c'], file_scheme='hive',
                 row_group_offsets=[0, 500])
    r = ParquetFile(tempdir)
    assert r.count == sum(~df.a.isnull())
    assert len(r.row_groups) == 6
    out = r.to_pandas()

    for i, row in out.iterrows():
        assert row.b in list(df[(df.a==row.a)&(df.c==row.c)].b)


@pytest.mark.parametrize('compression', ['GZIP',
                                         'gzip',
                                         None,
                                         {'x': 'GZIP'},
                                         {'y': 'gzip', 'x': None}])
def test_write_compression_dict(tempdir, compression):
    df = pd.DataFrame({'x': [1, 2, 3],
                       'y': [1., 2., 3.]})
    fn = os.path.join(tempdir, 'tmp.parq')
    writer.write(fn, df, compression=compression)
    r = ParquetFile(fn)
    df2 = r.to_pandas()

    tm.assert_frame_equal(df, df2, check_categorical=False)


def test_write_compression_schema(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3],
                       'y': [1., 2., 3.]})
    fn = os.path.join(tempdir, 'tmp.parq')
    writer.write(fn, df, compression={'x': 'gzip'})
    r = ParquetFile(fn)

    assert all(c.meta_data.codec for row in r.row_groups
                                 for c in row.columns
                                 if c.meta_data.path_in_schema == ['x'])
    assert not any(c.meta_data.codec for row in r.row_groups
                                 for c in row.columns
                                 if c.meta_data.path_in_schema == ['y'])


def test_index(tempdir):
    fn = os.path.join(tempdir, 'tmp.parq')
    df = pd.DataFrame({'x': [1, 2, 3],
                       'y': [1., 2., 3.]},
                       index=pd.Index([10, 20, 30], name='z'))

    writer.write(fn, df)

    r = ParquetFile(fn)
    assert set(r.columns) == {'x', 'y', 'z'}


def test_naive_index(tempdir):
    fn = os.path.join(tempdir, 'tmp.parq')
    df = pd.DataFrame({'x': [1, 2, 3],
                       'y': [1., 2., 3.]})

    writer.write(fn, df)
    r = ParquetFile(fn)

    assert set(r.columns) == {'x', 'y'}

    writer.write(fn, df, write_index=True)
    r = ParquetFile(fn)

    assert set(r.columns) == {'x', 'y', 'index'}


def test_text_convert(tempdir):
    df = pd.DataFrame({'a': ['π'] * 100,
                       'b': [b'a'] * 100})
    fn = os.path.join(tempdir, 'tmp.parq')

    write(fn, df, fixed_text={'a': 2, 'b': 1})
    pf = ParquetFile(fn)
    assert pf.schema[1].type == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY
    assert pf.schema[1].type_length == 2
    assert pf.schema[2].type == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY
    assert pf.schema[2].type_length == 1
    assert pf.statistics['max']['a'] == ['π']
    df2 = pf.to_pandas()
    tm.assert_frame_equal(df, df2, check_categorical=False)

    write(fn, df)
    pf = ParquetFile(fn)
    assert pf.schema[1].type == parquet_thrift.Type.BYTE_ARRAY
    assert pf.schema[2].type == parquet_thrift.Type.BYTE_ARRAY
    assert pf.statistics['max']['a'] == ['π']
    df2 = pf.to_pandas()
    tm.assert_frame_equal(df, df2, check_categorical=False)

    write(fn, df, fixed_text={'a': 2})
    pf = ParquetFile(fn)
    assert pf.schema[1].type == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY
    assert pf.schema[2].type == parquet_thrift.Type.BYTE_ARRAY
    assert pf.statistics['max']['a'] == ['π']
    df2 = pf.to_pandas()
    tm.assert_frame_equal(df, df2, check_categorical=False)


def test_null_time(tempdir):
    """Test reading a file that contains null records."""
    tmp = str(tempdir)
    expected = pd.DataFrame({"t": [np.timedelta64(), np.timedelta64('NaT')]})
    fn = os.path.join(tmp, "test-time-null.parquet")

    # with NaT
    write(fn, expected, has_nulls=False)
    p = ParquetFile(fn)
    data = p.to_pandas()
    assert (data['t'] == expected['t'])[~expected['t'].isnull()].all()
    assert sum(data['t'].isnull()) == sum(expected['t'].isnull())

    # with NULL
    write(fn, expected, has_nulls=True)
    p = ParquetFile(fn)
    data = p.to_pandas()
    assert (data['t'] == expected['t'])[~expected['t'].isnull()].all()
    assert sum(data['t'].isnull()) == sum(expected['t'].isnull())


def test_auto_null(tempdir):
    tmp = str(tempdir)
    df = pd.DataFrame({'a': [1, 2, 3, 0],
                       'b': [1., 2., 3., np.nan],
                       'c': pd.to_timedelta([1, 2, 3, np.nan], unit='ms'),
                       'd': ['a', 'b', 'c', None]})
    df['e'] = df['d'].astype('category')
    fn = os.path.join(tmp, "test.parq")

    with pytest.raises((TypeError, AttributeError)):
        ## TODO: this should be a nicer error?
        write(fn, df, has_nulls=False)

    write(fn, df, has_nulls=True)
    pf = ParquetFile(fn)
    for col in pf.schema[2:]:
        assert col.repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    assert pf.schema[1].repetition_type == parquet_thrift.FieldRepetitionType.REQUIRED
    df2 = pf.to_pandas(categories=['e'])
    tm.assert_frame_equal(df, df2, check_categorical=False)

    write(fn, df, has_nulls=None)
    pf = ParquetFile(fn)
    for col in pf.schema[1:3]:
        assert col.repetition_type == parquet_thrift.FieldRepetitionType.REQUIRED
    assert pf.schema[4].repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    df2= pf.to_pandas(categories=['e'])
    tm.assert_frame_equal(df, df2, check_categorical=False)


@pytest.mark.parametrize('n', (10, 127, 2**8 + 1, 2**16 + 1))
def test_many_categories(tempdir, n):
    tmp = str(tempdir)
    cats = np.arange(n)
    codes = np.random.randint(0, n, size=1000000)
    df = pd.DataFrame({'x': pd.Categorical.from_codes(codes, cats), 'y': 1})
    fn = os.path.join(tmp, "test.parq")

    write(fn, df, has_nulls=False)
    pf = ParquetFile(fn)
    out = pf.to_pandas(categories={'x': n})

    tm.assert_frame_equal(df, out, check_categorical=False)

    df.set_index('x', inplace=True)
    write(fn, df, has_nulls=False, write_index=True)
    pf = ParquetFile(fn)
    out = pf.to_pandas(categories={'x': n}, index='x')

    assert (out.index == df.index).all()
    assert (out.y == df.y).all()


@pytest.mark.parametrize('row_groups', ([0], [0, 2]))
@pytest.mark.parametrize('dirs', (['', ''], ['cat=1', 'cat=2']))
def test_merge(tempdir, dirs, row_groups):
    fn = str(tempdir)

    os.makedirs(os.path.join(fn, dirs[0]), exist_ok=True)
    df0 = pd.DataFrame({'a': [1, 2, 3, 4]})
    fn0 = os.sep.join([fn, dirs[0], 'out0.parq'])
    write(fn0, df0, row_group_offsets=row_groups)

    os.makedirs(os.path.join(fn, dirs[1]), exist_ok=True)
    df1 = pd.DataFrame({'a': [5, 6, 7, 8]})
    fn1 = os.sep.join([fn, dirs[1], 'out1.parq'])
    write(fn1, df1, row_group_offsets=row_groups)

    # with file-names
    pf = writer.merge([fn0, fn1])
    assert len(pf.row_groups) == 2 * len(row_groups)
    out = pf.to_pandas().a.tolist()
    assert out == [1, 2, 3, 4, 5, 6, 7, 8]
    if "cat=1" in dirs:
        assert 'cat' in pf.cats

    # with instances
    pf = writer.merge([ParquetFile(fn0), ParquetFile(fn1)])
    assert len(pf.row_groups) == 2 * len(row_groups)
    out = pf.to_pandas().a.tolist()
    assert out == [1, 2, 3, 4, 5, 6, 7, 8]
    if "cat=1" in dirs:
        assert 'cat' in pf.cats


def test_merge_s3(tempdir, s3):
    fn = str(tempdir)

    df0 = pd.DataFrame({'a': [1, 2, 3, 4]})
    fn0 = TEST_DATA + '/out0.parq'
    write(fn0, df0, open_with=s3.open)

    df1 = pd.DataFrame({'a': [5, 6, 7, 8]})
    fn1 = TEST_DATA + '/out1.parq'
    write(fn1, df1, open_with=s3.open)

    # with file-names
    pf = writer.merge([fn0, fn1], open_with=s3.open)
    assert len(pf.row_groups) == 2
    out = pf.to_pandas().a.tolist()
    assert out == [1, 2, 3, 4, 5, 6, 7, 8]


def test_merge_fail(tempdir):
    fn = str(tempdir)

    df0 = pd.DataFrame({'a': [1, 2, 3, 4]})
    fn0 = os.sep.join([fn, 'out0.parq'])
    write(fn0, df0)

    df1 = pd.DataFrame({'a': ['a', 'b', 'c']})
    fn1 = os.sep.join([fn, 'out1.parq'])
    write(fn1, df1)

    with pytest.raises(ValueError) as e:
        writer.merge([fn0, fn1])
    assert 'schemas' in str(e)

    os.remove(fn1)
    write(fn1, df0, file_scheme='hive')
    with pytest.raises(ValueError) as e:
        writer.merge([fn0, fn1])
    assert 'multi-file' in str(e)


def test_analyse_paths():
    file_list = ['a', 'b']
    base, out = writer.analyse_paths(file_list, '/')
    assert (base, out) == ('', ['a', 'b'])

    file_list = ['c/a', 'c/b']
    base, out = writer.analyse_paths(file_list, '/')
    assert (base, out) == ('c', ['a', 'b'])

    file_list = ['c/d/a', 'c/d/b']
    base, out = writer.analyse_paths(file_list, '/')
    assert (base, out) == ('c/d', ['a', 'b'])

    file_list = ['c/cat=1/a', 'c/cat=2/b', 'c/cat=1/c']
    base, out = writer.analyse_paths(file_list, '/')
    assert (base, out) == ('c', ['cat=1/a', 'cat=2/b', 'cat=1/c'])

    file_list = ['c/cat=2/b', 'c/cat/a', 'c/cat=1/c']
    with pytest.raises(ValueError) as e:
        writer.analyse_paths(file_list, '/')
    assert 'c/cat/a' in str(e)

    file_list = ['c/cat=2/b', 'c/fred=2/a', 'c/cat=1/c']
    with pytest.raises(ValueError) as e:
        writer.analyse_paths(file_list, '/')
    assert 'directories' in str(e)

    file_list = ['c/cat=2/b', 'c/a', 'c/cat=1/c']
    with pytest.raises(ValueError) as e:
        writer.analyse_paths(file_list, '/')
    assert 'nesting' in str(e)


def test_append_simple(tempdir):
    fn = os.path.join(str(tempdir), 'test.parq')
    df = pd.DataFrame({'a': [1, 2, 3, 0],
                       'b': ['a', 'a', 'b', 'b']})
    write(fn, df, write_index=False)
    write(fn, df, append=True, write_index=False)

    pf = ParquetFile(fn)
    expected = pd.concat([df, df], ignore_index=True)
    pd.util.testing.assert_frame_equal(pf.to_pandas(), expected,
                                       check_categorical=False)


@pytest.mark.parametrize('row_groups', ([0], [0, 2]))
@pytest.mark.parametrize('partition', ([], ['b']))
def test_append(tempdir, row_groups, partition):
    fn = str(tempdir)
    df0 = pd.DataFrame({'a': [1, 2, 3, 0],
                        'b': ['a', 'b', 'a', 'b'],
                        'c': True})
    df1 = pd.DataFrame({'a': [4, 5, 6, 7],
                        'b': ['a', 'b', 'a', 'b'],
                        'c': False})
    write(fn, df0, partition_on=partition, file_scheme='hive',
          row_group_offsets=row_groups)
    write(fn, df1, partition_on=partition, file_scheme='hive',
          row_group_offsets=row_groups, append=True)

    pf = ParquetFile(fn)

    expected = pd.concat([df0, df1], ignore_index=True)

    assert len(pf.row_groups) == 2 * len(row_groups) * (len(partition) + 1)
    items_out = {tuple(row[1])
                 for row in pf.to_pandas()[['a', 'b', 'c']].iterrows()}
    items_in = {tuple(row[1])
                for row in expected.iterrows()}
    assert items_in == items_out


def test_append_fail(tempdir):
    fn = str(tempdir)
    df0 = pd.DataFrame({'a': [1, 2, 3, 0],
                        'b': ['a', 'b', 'a', 'b'],
                        'c': True})
    df1 = pd.DataFrame({'a': [4, 5, 6, 7],
                        'b': ['a', 'b', 'a', 'b'],
                        'c': False})
    write(fn, df0, file_scheme='hive')
    with pytest.raises(ValueError) as e:
        write(fn, df1, file_scheme='simple', append=True)
    assert 'existing file scheme' in str(e)

    fn2 = os.path.join(fn, 'temp.parq')
    write(fn2, df0, file_scheme='simple')
    with pytest.raises(ValueError) as e:
        write(fn2, df1, file_scheme='hive', append=True)
    assert 'existing file scheme' in str(e)


def test_bad_object_encoding(tempdir):
    df = pd.DataFrame({'x': ['a', 'ab']})
    with pytest.raises(ValueError) as e:
        write(str(tempdir), df, object_encoding='utf-8')
    assert "utf-8" in str(e)
