Usage Notes
===========

Some additional information to bear in mind when using fastparquet,
in no particular order.

Whilst we aim to make the package simple to use, some choices on the part
of the user may effect performance and data consistency.

Categoricals
------------

When writing a data-frame with a column of pandas type ``Category``, the
data will be encoded using Parquet "dictionary encoding". This stores all
the possible values of the column (typically strings) separately, and the
index corresponding to each value as a data set of integers. If there
is a significant performance gain to be made, such as long labels, but low
cardinality, users are suggested to turn their object columns into the
category type:

.. code-block:: python

    df[col] = df[col].astype('category')

To efficiently load a column as a categorical type, include it in the optional
keyword parameter ``categories``; however it must be encoded as dictionary
throughout the dataset (as it will, if written by fastparquet).

.. code-block:: python

    pf = ParquetFile('input.parq')
    df = pf.to_pandas(categories={'cat': 12})

Where we provide a hint that the column ``cat`` has up to 12 possible values.
``categories`` can also take a list, in which case up to 32767 (2**15 - 1)
labels are assumed.
Columns that are encoded as dictionary but not included in ``categories`` will
be de-referenced on load which is potentially expensive.

Note that before loading, it is not possible to know whether the above condition
will be met, so the ``dtypes`` attribute of a ``ParquetFile`` will show the
data type appropriate for the values of column and never ``Category``.

Byte Arrays
-----------

Often, information in a column can be encoded in a small number of characters,
perhaps a single character. Variable-length byte arrays are also slow and
inefficient, however, since the length of each value needs to be stored.

Fixed-length byte arrays provide the best of both, and will probably be the
most efficient storage where the values are 1-4 bytes long, especially if the
cardinality is relatively high for dictionary encoding. To automatically
convert string values to fixed-length when writing, use the ``fixed_text``
optional keyword, with a predetermined length.

.. code-block:: python

    write('out.parq', df, fixed_text={'char_code': 1})

Such an encoding will be the fastest to read, especially if the values are
bytes type, as opposed to UTF8 strings. The values will be converted back
to objects upon loading.

Fixed-length byte arrays are not supported by Spark, so
files written using this may not be portable.

Short-type Integers
-------------------

Types like 1-byte ints (signed or unsigned) are stored using bitpacking for
optimized space and speed. Unfortunately, Spark is known not to be
able to handle these types. If you want to generate files for reading by
Spark, be sure to transform integer columns to a minimum of 4 bytes (numpy
``int32`` or ``uint32``) before saving.

Nulls
-----

In pandas, NULL values are typically represented by the floating point ``NaN``.
This value can be stored in float and time fields, and will be read back such
that the original data is recovered. They are not, however, the same thing
as missing values, and if querying the resultant files using other frameworks,
this should be born in mind. With ``has_nulls=None`` (the default) on writing,
float and time fields will not write separate NULLs information, and
the metadata will give num_nulls=0.

Using ``has_nulls=True`` (which can
also be specified for some specific subset of columns using a list) will force
the writing of NULLs information, making the output more transferable, but
comes with a performance penalty.

Because of the ``NaN`` encoding for NULLs, pandas is unable to represent missing
data in an integer field. In practice, this means that fastparquet will never
write any NULLs in an integer field, and if reading an integer field with NULLs,
the resultant column will become a float type. This is in line with what
pandas does when reading integers.

For object and category columns, NULLs (``None``) do exist, and fastparquet can
read and write them. Including this data does come at a cost, however.
Currently, with ``has_nulls=None`` (the default), object fields will assume
the existence of NULLs; if a chunk does not in fact have any, then skipping
their decoding will be pretty efficient. In general, it is best to provide
``has_nulls`` with a list of columns known to contain NULLs - however if None
is encountered in a column not in the list, this will raise an exception.


Data Types
----------

There is fairly good correspondence between pandas data-types and Parquet
simple and logical data types. The `types documentation <https://github.com/Parquet/parquet-format/blob/master/LogicalTypes.md>`_
gives details of the implementation spec.

A couple of caveats should be noted:

- fastparquet will
  not write any Decimal columns, only float, and when reading such columns,
  the output will also be float, with potential machine-precision errors;
- only UTF8 encoding for text is automatically handled, although arbitrary
  byte strings can be written as raw bytes type;
- the time types have microsecond accuracy, whereas pandas time types normally
  are nanosecond accuracy;
- all times are stored as UTC, and timezone information will
  be lost;
- complex numbers must have their real and imaginary parts stored as two
  separate float columns.

Partitions and row-groups
-------------------------

The Parquet format allows for partitioning the data by the values of some
(low-cardinality) columns and by row sequence number. Both of these can be
in operation at the same time, and, in situations where only certain sections
of the data need to be loaded, can produce great performance benefits in
combination with load filters.

Splitting on both row-groups and partitions can potentially result in many
data-files and large metadata. It should be used sparingly, when partial
selecting of the data is anticipated.

**Row groups**

The keyword parameter ``row_group_offsets`` allows control of the row
sequence-wise splits in the data. For example, with the default value,
each row group will contain 50 million rows. The exact index of the start
of each row-group can also be specified, which may be appropriate in the
presence of a monotonic index: such as a time index might lead to the desire
to have all the row-group boundaries coincide with year boundaries in the
data.

**Partitions**

In the presence of some low-cardinality columns, it may be advantageous to
split data data on the values of those columns. This is done by writing a
directory structure with *key=value* names. Multiple partition columns can
be chosen, leading to a multi-level directory tree.

Consider the following directory tree from this `Spark example <http://Spark.apache.org/docs/latest/sql-programming-guide.html#partition-discovery>`_:

    table/
        gender=male/
           country=US/
              data.parquet
           country=CN/
              data.parquet
        gender=female/
            country=US/
               data.parquet
            country=CN/
               data.parquet

Here the two partitioned fields are *gender* and *country*, each of which have
two possible values, resulting in four datafiles. The corresponding columns
are not stored in the data-files, but inferred on load, so space is saved,
and if selecting based on these values, potentially some of the data need
not be loaded at all.

If there were two row groups and the same partitions as above, each leaf
directory would contain (up to) two files, for a total of eight. If a
row-group happens to contain no data for one of the field value combinations,
that data file is omitted.


Iteration
---------

For data-sets too big to fit conveniently into memory, it is possible to
iterate through the row-groups in a similar way to reading by chunks from
CSV with pandas.

.. code-block:: python

    pf = ParquetFile('myfile.parq')
    for df in pf.iter_row_groups():
        print(df.shape)
        # process sub-data-frame df

Thus only one row-group is in memory at a time. The same set of options
are available as in ``to_pandas`` allowing, for instance, reading only
specific columns, loading to
categoricals or to ignore some row-groups using filtering.

To get the first row-group only, one would go:

.. code-block:: python

    first = next(iter(pf.iter_row_groups()))

Connection to Dask
------------------

Dask usage is still in development. Expect the features to lag behind
those in fastparquet, and sometimes to become incompatible, if a change has
been made in the one but not the other.

`Dask <http://dask.pydata.org/>`_ provides a pandas-like dataframe interface to
larger-than-memory and distributed datasets, as part of a general parallel
computation engine. In this context, it allows the parallel loading and
processing of the component pieces of a Parquet dataset across the cored of
a CPU and/or the nodes of a distributed cluster.

Dask will provide two simple end-user functions:

- ``dask.dataframe.read_parquet`` with keyword options similar to
  ``ParquetFile.to_pandas``. The URL parameter, however, can point to
  various filesystems, such as S3 or HDFS. Loading is *lazy*, only happening
  on demand.
- ``dask.dataframe.DataFrame.to_parquet`` with keyword options similar to
  ``fastparquet.write``. One row-group/file will be generated for each division
  of the dataframe, or, if using partitioning, up to one row-group/file per
  division per partition combination.
