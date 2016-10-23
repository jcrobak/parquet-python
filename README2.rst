Parquet 2.0
===========

Introduction
------------

Since the second week of October, this fork of parquet-python has been
undergoing considerable redevelopment. The aim is to have a small and simple
and performant library for reading and writing the parquet format from python.

A list of expected features and their status in this branch can be found in
(this issue)[https://github.com/martindurant/parquet-python/issues/1].
Please feel free to comment on that list as to missing items and priorities.

In the meantime, the more eyes on this code, the more example files and the
more use cases the better.

For the time being, this code should be considered extreme beta.

Requirements
------------

(all development is against recent versions in the default anaconda channels)

Required:

- numba
- numpy
- pandas

Optional (compression algorithms; gzip is always available):

- snappy
- lzo
- brotli

Installation
------------

    > pip install git+https://github.com/martindurant/parquet-python@fast_writer

or clone this repo, checkout the fast_writer branch and run

    > python setup.py develop

(NB: the final branch and repo are not yet decided, the above information is
expected to be out of date soon - at which point this README will be edited)

Usage
-----

*Reading*

.. code-block:: python

    import parquet
    pf = parquet.ParquetFile('myfile.parq')
    df = pf.to_pandas()
    df2 = pf.to_pandas(['col1', 'col2'], usecats=['col1'])

You may specify which columns to load, which of those to keep as categoricals
(if the data uses dictionary encoding). The file-path can be a single file,
a metadata file pointing to other data files, or a directory (tree) containing
data files. The latter is what is typically output by hive/spark.

*Writing*

.. code-block:: python

    import parquet
    parquet.write('outfile.parq', df)
    parquet.write('outfile2.parq', df, partitions=[0, 10000, 20000],
                  compression='GZIP', file_scheme='hive')

The default is to produce a single output file with a single row-group
(i.e., logical segment) and no compression. At the moment, only simple
data-types and plain encoding are supported, so expect performance to be
similar to `numpy.savez`.
