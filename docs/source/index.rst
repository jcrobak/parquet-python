fastparquet
===========

A python interface to the parquet file format.

Introduction
------------

The `Parquet format <https://github.com/Parquet/parquet-format>`_ is a common binary data store, used
particularly in the Hadoop/big-data sphere. It provides several advantages relevant to big-data
processing:

- columnar storage, only read the data of interest
- efficient binary packing
- choice of compression algorithms and encoding
- split data into files, allowing for parallel processing
- range of logical types
- statistics stored in metadata allow for skipping unneeded chunks
- data partitioning using the directory structure

Since it was developed as part of the Hadoop ecosystem, parquet's reference implementation is
written in java. This package aims to provide a performant library to read and write parquet files
from python, without any need for a python-java bridge. This will make the parquet format an
ideal storage mechanism for python-based big data workflows.

The tabular nature of Parquet is a good fit for the pandas data-frame objects, and
we exclusively deal with data-frame<->parquet.

Highlights
----------

The original outline plan for this project can be found `here <https://github.com/dask/fastparquet/issues/1>`_

Briefly, some features of interest:

- read and write parquet files, in single- or multiple-file format. The latter is common found in hive/spark usage.
- choice of compression per-column and various optimized encoding schemes; ability to choose row divisions and partitioning on write.
- acceleration of both reading and writing using `numba <http://numba.pydata.org/>`_
- ability to read and write to arbitrary file-like objects, allowing interoperability with `s3fs <http://s3fs.readthedocs.io/>`_, `hdfs3 <http://hdfs3.readthedocs.io/>`_, `adlfs <https://github.com/Azure/azure-data-lake-store-python>`_ and possibly others.
- (in development) can be called from `dask <http://dask.pydata.org>`_, to enable parallel reading and writing with parquet files, possibly distributed across a cluster.

Caveats, Known Issues
---------------------

Not all parts of the parquet-format have been implemented yet or tested.
fastparquet is, however, capable of reading all the data files from the
`parquet-compatibility <https://github.com/Parquet/parquet-compatibility>`_
project. Some encoding mechanisms in parquet are rare, and may be implemented
on request - please post an issue.

Nested data types do not fit well with the pandas tabular model, and are not
currently supported. We do aim to support 1-level nesting (lists and key-value
maps) in the future.

Not all output options will be compatible with every other parquet
framework, which each implement only a subset of the standard, see
the usage notes.

A list of current issues can be found `here <https://github.com/dask/fastparquet/>`_.

Relation to Other Projects
--------------------------

-  `parquet-cpp <https://github.com/apache/parquet-cpp/>`_ is a low-level C++
implementation of the parquet format which can be called from python using
Apache `Arrow <http://pyarrow.readthedocs.io/en/latest/>`_ bindings. We aim to
collaborate with parquet-cpp, and it is possible, in the medium term, that their low-level
routines will replace some functions in fastparquet or that high-level logic in
fastparquet will be migrated to C++.

- `parquet-python <https://github.com/jcrobak/parquet-python>`_ is the original
pure-python parquet quick-look utility which was the inspiration for fastparquet.

- `pySpark <http://spark.apache.org/docs/2.1.0/programming-guide.html>`_, a python API to the spark
engine, interfaces python commands with a java/scala execution core, and thereby
gives python programmers access to the parquet format. fastparquet has no
defined relationship to pySpark, but can provide an alternative path to providing
data to Spark or reading data produced by Spark without invoking a pySpark client
or interacting directly with the Spark scheduler.

- fastparquet lives within the `dask <http://dask.pydata.org>`_ ecosystem, and
although it is useful by itself, it is designed to work well with dask and
`distributed <http://distributed.readthedocs.io/en/latest/>`_ for parallel
execution, as well as related libraries such as s3fs for pythonic access to
Amazon S3.


Index
-----

.. toctree::

    install
    quickstart
    details
    api
    filesystems
    developer

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
