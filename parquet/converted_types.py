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
except ImportError:
    def BSON(x):
        raise ImportError("BSON not found")

PY3 = sys.version_info.major > 2

# define bytes->int for non 2, 4, 8 byte ints
if PY3:
    def intbig(data):
        """Convert big ints using python 3's built-in support."""
        return int.from_bytes(data, 'big', signed=True)
else:
    def intbig(data):
        """Convert big ints using a hack of encoding bytes as hex and decoding to int."""
        return int(codecs.encode(data, 'hex'), 16)

DAYS_TO_MILLIS = 86400000000000
"""Number of millis in a day. Used to convert a Date to a date"""


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
        return data.str.decode('utf8')
    if ctype == parquet_thrift.ConvertedType.DECIMAL:
        scale_factor = 10**-se.scale
        return data * scale_factor
    elif ctype == parquet_thrift.ConvertedType.DATE:
        return data.map(datetime.date.fromordinal)
    elif ctype == parquet_thrift.ConvertedType.TIME_MILLIS:
        return pd.to_timedelta(data, unit='ms')
    elif ctype == parquet_thrift.ConvertedType.TIMESTAMP_MILLIS:
        return pd.to_datetime(data, unit='ms')
    elif ctype == parquet_thrift.ConvertedType.UINT_8:
        return data.astype(np.uint8)
    elif ctype == parquet_thrift.ConvertedType.UINT_16:
        return data.astype(np.uint16)
    elif ctype == parquet_thrift.ConvertedType.UINT_32:
        return data.astype(np.uint32)
    elif ctype == parquet_thrift.ConvertedType.UINT_64:
        return data.astype(np.uint64)
    elif ctype == parquet_thrift.ConvertedType.JSON:
        return data.map(lambda s: json.loads(s.decode()))
    elif ctype == parquet_thrift.ConvertedType.BSON and BSON:
        return data.map(lambda s: BSON(s).decode())
    else:
        logger.info("Converted type '%s'' not handled",
                    parquet_thrift.ConvertedType._VALUES_TO_NAMES[ctype])  # pylint:disable=protected-access
    return data
