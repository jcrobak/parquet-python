# parquet-python

parquet-python is a pure-python implementation (currently with only read-support) of the [parquet format](https://github.com/Parquet/parquet-format). It comes with a script for reading parquet files and outputting the data to stdout as JSON or TSV (without the overhead of JVM startup). Performance has not yet been optimized, but it's useful for debugging and quick viewing of data in files.

Not all parts of the parquet-format have been implemented yet or tested e.g. nested data -- see Todos below for a full list. With that said, parquet-python is capable of reading all the data files from the [parquet-compatability](https://github.com/Parquet/parquet-compatibility) project.


# requirements

parquet-python has been tested on python 2.7. It depends on `thrift` (0.9) and `python-snappy` (for snappy compressed files).


# getting started

parquet-python is not yet uploaded to PyPi as the code has a lot of bugs. To get started, clone the project, change into the parquet-python directory, and run `python -m parquet`.

You may need to install the `thrift` and `python-snappy` projects with `easy_install` or `pip`. To install parquet-python system-wide, run `python setup.py install`.

## Example

parquet-python currently has two programatic interfaces with similar functionality to Python's csv reader. First, it supports a DictReader which returns a dictionary per row. Second, it has a reader which returns a list of values for each row. Both function require a file-like object and support an optional `columns` field to only read the specified columns.

```python

import parquet
import json

## assuming parquet file with two rows and three columns:
## foo bar baz
## 1   2   3
## 4   5   6

with open("test.parquet") as fo:
   # prints:
   # {"foo": 1, "bar": 2}
   # {"foo": 4, "bar": 5}
   for row in parquet.DictReader(fo, columns=['foo', 'bar']):
       print(json.dumps(row))


with open("test.parquet") as fo:
   # prints:
   # 1,2
   # 4,5
   for row in parquet.reader(fo, columns=['foo', 'bar]):
       print(",".join([str(r) for r in row]))
```

# Todos

* Fix handling of repetition-levels and definition-levels
* Tests for nested schemas, null data
* Support reading of data from HDFS via snakebite and/or webhdfs.
* Implement writing
* performance evaluation and optimization (i.e. how does it compare to the c++, java implementations)


# Contributing

Is done via Pull Requests. Please include tests with your changes and follow [pep8](http://www.python.org/dev/peps/pep-0008/).
