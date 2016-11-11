fastparquet
===========

.. image:: https://travis-ci.org/jcrobak/parquet-python.svg?branch=master
    :target: https://github.com/dask/fastparquet

fastparquet is a python implementation of the `parquet
format <https://github.com/Parquet/parquet-format>`_, aiming integrate
into python-based big data work-flows.

Not all parts of the parquet-format have been implemented yet or tested
e.g. see the Todos linked below. With that said,
fastparquet is capable of reading all the data files from the
`parquet-compatability <https://github.com/Parquet/parquet-compatibility>`_
project.

Introduction
------------

**This software is alpha, expect frequent API changes and breakages.**

A list of expected features and their status in this branch can be found in
`this issue`_, and further
Please feel free to comment on that list as to missing items and priorities.

.. _this issue: https://github.com/dask/fastparquet/issues/1

In the meantime, the more eyes on this code, the more example files and the
more use cases the better.

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

Install using conda::

   conda install -c conda-forge fastparquet

install from pypi::

   pip install fastparquet

or install latest version from github::

   pip install git+https://github.com/dask/fastparquet

For the pip methods, numba must have been previously installed (using conda).

Usage
-----

*Reading*

.. code-block:: python

    from fastparquet import ParquetFile
    pf = ParquetFile('myfile.parq')
    df = pf.to_pandas()
    df2 = pf.to_pandas(['col1', 'col2'], categories=['col1'])

You may specify which columns to load, which of those to keep as categoricals
(if the data uses dictionary encoding). The file-path can be a single file,
a metadata file pointing to other data files, or a directory (tree) containing
data files. The latter is what is typically output by hive/spark.

*Writing*

.. code-block:: python

    from fastparquet import write
    write('outfile.parq', df)
    write('outfile2.parq', df, row_group_offsets=[0, 10000, 20000],
          compression='GZIP', file_scheme='hive')

The default is to produce a single output file with a single row-group
(i.e., logical segment) and no compression. At the moment, only simple
data-types and plain encoding are supported, so expect performance to be
similar to *numpy.savez*.

History
-------

Since early October 2016, this fork of `parquet-python`_ has been
undergoing considerable redevelopment. The aim is to have a small and simple
and performant library for reading and writing the parquet format from python.

.. _parquet-python: https://github.com/jcrobak/parquet-python

