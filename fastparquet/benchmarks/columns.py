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
    result[name] = t1 - t0


def time_column():
    with tmpdir() as tempdir:
        result = {}
        fn = os.path.join(tempdir, 'temp.parq')
        n = 10000000
        offsets = np.random.randint(-1e10, 1e10, n).view('timedelta64[ns]')
        df = pd.DataFrame({'x': [datetime.datetime.now()] * n})
        df.x += offsets

        write(fn, df)
        with measure('write time column with random offsets', result):
            write(fn, df)

        pf = ParquetFile(fn)
        out = pf.to_pandas()  # warm-up

        with measure('read time columns with small offsets', result):
            out = pf.to_pandas()
            
        return result


if __name__ == '__main__':
    result = {}
    for f in [time_column]:
        result.update(f())
    print(result)
