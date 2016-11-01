import os

import thriftpy


THRIFT_FILE = os.path.join(os.path.dirname(__file__), "parquet.thrift")
parquet_thrift = thriftpy.load(THRIFT_FILE, module_name="parquet_thrift")  # pylint: disable=invalid-name
