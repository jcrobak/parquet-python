from contextlib import contextmanager
import datetime
import os
import numpy as np
import pandas as pd
import time
from fastparquet import write, ParquetFile
from dask.utils import tmpdir


@contextmanager
def measure(name, result):
    t0 = time.time()
    yield
    t1 = time.time()
    result[name] = round((t1 - t0) * 1000, 3)


def time_column():
    with tmpdir() as tempdir:
        result = {}
        fn = os.path.join(tempdir, 'temp.parq')
        n = 10000000
        r = np.random.randint(-1e10, 1e10, n).view('timedelta64[ns]')
        df = pd.DataFrame({'x': r.copy()})

        write(fn, df)
        with measure('write random times, no nulls', result):
            write(fn, df, has_nulls=False)

        pf = ParquetFile(fn)
        out = pf.to_pandas()  # warm-up

        with measure('read random times, no nulls', result):
            out = pf.to_pandas()

        with measure('write random times, no nulls but has_null=True', result):
            write(fn, df, has_nulls=True)

        pf = ParquetFile(fn)
        out = pf.to_pandas()  # warm-up

        with measure('read random times, no nulls but has_null=True', result):
            out = pf.to_pandas()

        df.loc[n//2, 'x'] = pd.to_datetime('NaT')
        with measure('write random times, with null', result):
            write(fn, df, has_nulls=True)

        pf = ParquetFile(fn)
        out = pf.to_pandas()  # warm-up

        with measure('read random times, with null', result):
            out = pf.to_pandas()

        df.loc[n//2, 'x'] = pd.to_datetime('NaT')
        with measure('write random times, with null but has_null=False', result):
            write(fn, df, has_nulls=False)

        pf = ParquetFile(fn)
        out = pf.to_pandas()  # warm-up

        with measure('read random times, with null but has_null=False', result):
            out = pf.to_pandas()

        return result


if __name__ == '__main__':
    result = {}
    for f in [time_column]:
        result.update(f())
    for k in sorted(result):
        print(k, result[k])


def time_find_nulls(N=10000000):
    x = np.random.random(N)
    df = pd.DataFrame({'x': x})
    result = {}
    run_find_nulls(df, result)
    df.loc[N//2, 'x'] = np.nan
    run_find_nulls(df, result)
    df.loc[:, 'x'] = np.nan
    df.loc[N//2, 'x'] = np.random.random()
    run_find_nulls(df, result)
    df.loc[N//2, 'x'] = np.nan
    run_find_nulls(df, result)

    x = np.random.randint(0, 2**30, N)
    df = pd.DataFrame({'x': x})
    run_find_nulls(df, result)

    df = pd.DataFrame({'x': x.view('datetime64[s]')})
    run_find_nulls(df, result)
    v = df.loc[N//2, 'x']
    df.loc[N//2, 'x'] = pd.to_datetime('NaT')
    run_find_nulls(df, result)
    df.loc[:, 'x'] = pd.to_datetime('NaT')
    df.loc[N//2, 'x'] = v
    run_find_nulls(df, result)
    df.loc[:, 'x'] = pd.to_datetime('NaT')
    run_find_nulls(df, result)

    out = [(k + (v, )) for k, v in result.items()]
    df = pd.DataFrame(out, columns=('type', 'nvalid', 'op', 'time'))
    df.groupby(('type', 'nvalid', 'op')).sum()
    return df


def run_find_nulls(df, res):
    nvalid = (df.x == df.x).sum()
    with measure((df.x.dtype.kind, nvalid, 'notnull'), res):
        df.x.notnull()
    with measure((df.x.dtype.kind, nvalid, 'notnull,sum'), res):
        df.x.notnull().sum()
    with measure((df.x.dtype.kind, nvalid, 'notnull,any'), res):
        df.x.notnull().any()
    with measure((df.x.dtype.kind, nvalid, 'notnull,all'), res):
        df.x.notnull().all()
    with measure((df.x.dtype.kind, nvalid, 'count'), res):
        df.x.count()

