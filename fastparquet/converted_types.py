# -#- coding: utf-8 -#-
"""
Deal with parquet logical types (aka converted types), higher-order things built from primitive types.

The implementations in this class are pure python for the widest compatibility,
but they're not necessarily the most performant.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import struct
import sys
from decimal import Decimal

from .thrift_structures import parquet_thrift
logger = logging.getLogger('parquet')  # pylint: disable=invalid-name

try:
    from bson import BSON
    unbson = lambda s: BSON(s).decode()
except ImportError:
    try:
        import bson
        unbson = bson.loads
    except:
        def BSON(x):
            raise ImportError("BSON not found")

DAYS_TO_MILLIS = 86400000000000
"""Number of millis in a day. Used to convert a Date to a date"""

simple = {parquet_thrift.Type.INT32: np.int32,
          parquet_thrift.Type.INT64: np.int64,
          parquet_thrift.Type.FLOAT: np.float32,
          parquet_thrift.Type.DOUBLE: np.float64,
          parquet_thrift.Type.BOOLEAN: np.bool_,
          parquet_thrift.Type.INT96: np.dtype('S12'),
          parquet_thrift.Type.BYTE_ARRAY: np.dtype("O")}
complex = {parquet_thrift.ConvertedType.UTF8: np.dtype("O"),
           parquet_thrift.ConvertedType.DECIMAL: np.float64,
           parquet_thrift.ConvertedType.UINT_8: np.uint8,
           parquet_thrift.ConvertedType.UINT_16: np.uint16,
           parquet_thrift.ConvertedType.UINT_32: np.uint32,
           parquet_thrift.ConvertedType.UINT_64: np.uint64,
           parquet_thrift.ConvertedType.INT_8: np.int8,
           parquet_thrift.ConvertedType.INT_16: np.int16,
           parquet_thrift.ConvertedType.INT_32: np.int32,
           parquet_thrift.ConvertedType.INT_64: np.int64,
           parquet_thrift.ConvertedType.TIME_MILLIS: np.dtype('<m8[ns]'),
           parquet_thrift.ConvertedType.DATE: np.dtype('<M8[ns]'),
           parquet_thrift.ConvertedType.TIMESTAMP_MILLIS: np.dtype(
                   '<M8[ns]')
           }


def typemap(se):
    """Get the final (pandas) dtype - no actual conversion"""
    if se.converted_type is None:
        if se.type in simple:
            return simple[se.type]
        else:
            return np.dtype("S%i" % se.type_length)
    if se.converted_type in complex:
        return complex[se.converted_type]
    return np.dtype("O")


def convert(data, se):
    """Convert known types from primitive to rich.

    Parameters
    ----------
    data: pandas series of primitive type
    se: a schema element.
    """
    # TODO: if input is categorical, only map on categories
    ctype = se.converted_type
    if ctype == parquet_thrift.ConvertedType.UTF8:
        return data.astype("O").str.decode('utf8')
    if ctype == parquet_thrift.ConvertedType.DECIMAL:
        scale_factor = 10**-se.scale
        return data * scale_factor
    elif ctype == parquet_thrift.ConvertedType.DATE:
        return pd.to_datetime(data.map(datetime.date.fromordinal), box=False)
    elif ctype == parquet_thrift.ConvertedType.TIME_MILLIS:
        return pd.to_timedelta(data, unit='ms', box=False)
    elif ctype == parquet_thrift.ConvertedType.TIMESTAMP_MILLIS:
        return pd.to_datetime(data, unit='ms', box=False)
    elif ctype == parquet_thrift.ConvertedType.TIME_MICROS:
        return pd.to_timedelta(data, unit='us', box=False)
    elif ctype == parquet_thrift.ConvertedType.TIMESTAMP_MICROS:
        return pd.to_datetime(data, unit='us', box=False)
    elif ctype == parquet_thrift.ConvertedType.UINT_8:
        return data.astype(np.uint8)
    elif ctype == parquet_thrift.ConvertedType.UINT_16:
        return data.astype(np.uint16)
    elif ctype == parquet_thrift.ConvertedType.UINT_32:
        return data.astype(np.uint32)
    elif ctype == parquet_thrift.ConvertedType.UINT_64:
        return data.astype(np.uint64)
    elif ctype == parquet_thrift.ConvertedType.JSON:
        return data.astype('O').str.decode('utf8').map(json.loads)
    elif ctype == parquet_thrift.ConvertedType.BSON:
        return data.map(unbson)
    elif ctype == parquet_thrift.ConvertedType.INTERVAL:
        # for those that understand, output is month, day, ms
        # maybe should convert to timedelta
        return data.map(lambda x: np.fromstring(x, dtype='<u4'))
    else:
        logger.info("Converted type '%s'' not handled",
                    parquet_thrift.ConvertedType._VALUES_TO_NAMES[ctype])  # pylint:disable=protected-access
    return data
