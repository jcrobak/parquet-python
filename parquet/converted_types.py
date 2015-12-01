# -#- coding: utf-8 -#-
"""
Deal with parquet logical types (aka converted types), higher-order
things built from primitive types.
"""
import datetime
import pandas as pd
import numpy as np
import struct
import json
import bson
import sys
PY3 = sys.version_info.major > 2

# define bytes->int for non 2, 4, 8 byte ints
if PY3:
    intbig = lambda x: int.from_bytes(x, 'big')
    intlittle = lambda x: int.from_bytes(x, 'little')
else:
    import codecs
    intbig = lambda x: int(codecs.encode(x, 'hex'), 16)
    intlittle = lambda x: int(codecs.encode(x[::-1], 'hex'), 16)


def invert_dict(d):
    "Map key:value to value:key"
    return {v:k for k, v in d.items()}


def map_spark_timestamp(x):
    """Special conversion for 'timestamp' column as created by spark, and
    possibly hive. Such a column does not have a 'converted type' defined,
    but is instead labeled in the alternative metadata stored in the footer
    key-value area.
    
    Data should be a column/series of 12-byte values (INT96).
    
    Use with series.map(map_spark_timestamp)

    Note that times are assumed to be UTC.    
    """
    if len(x) == 12:
        sec, days = struct.unpack('<ql', x)
    else:  # For the case that numpy has stripped trailing null bytes
        sec, days = struct.unpack('<ql', x+b'\0'*(12-len(x)))
    return datetime.datetime.fromtimestamp((days - 2440588) * 86400 + sec / 1000000000)


def convert_column(data, schemae):
    """Convert known types from primitive to rich.
    Designed for pandas series."""
    ctype = types_i[schemae.converted_type]
    if  ctype == 'DECIMAL':
        scale = 10**schemae.extra[0]
        if data.dtype == object:
            out = data.map(intbig) / scale
        else:
            out = data / scale
    elif ctype == 'DATE':
        # NB: If there are both DATE and TIME_MILLIS, should combine to
        # datetime as done in map_spark_timestamp
        out = (data * 86400000000000).astype('datetime64[ns]')
    elif ctype == 'TIME_MILLIS':
        out = (data * 1e6).astype('timedelta64[ns]')
    elif ctype == 'TIMESTAMP_MILLIS':
        out = pd.to_datetime(data, unit='ms')        
    elif ctype == 'UTF8':
        ## TODO: need decoder for py2
        out = data.map(bytes.decode) if PY3 else out
    elif ctype[0] == "U":
        # unsigned integers of various widths
        arr = data.values
        arr.dtype = np.dtype('u' + arr.dtype.name)
        out = pd.Series(arr)
    elif ctype == 'JSON':
        out = data.map(bytes.decode).map(json.loads)
    elif ctype == 'BSON':
        out = data.map(bson.decode_all)
    else:
        print("Converted type %i not handled" % ctype)
        out = data
    return out


# github.com/Parquet/parquet-format/blob/master/src/thrift/parquet.thrift
# list possible converted types as follows
types = dict(
  # a BYTE_ARRAY actually contains UTF8 encoded chars    
  UTF8 = 0,

  #a map is converted as an optional field containing a repeated key/value pair
  MAP = 1,

  # a key/value pair is converted into a group of two fields 
  MAP_KEY_VALUE = 2,

  # a list is converted into an optional field containing a repeated field for its
  # values
  LIST = 3,

  # an enum is converted into a binary field
  ENUM = 4,

   # This may be used to annotate binary or fixed primitive types. The
   # underlying byte array stores the unscaled value encoded as two's
   # complement using big-endian byte order (the most significant byte is the
   # zeroth element). The value of the decimal is the value # 10^{-scale}.
   #
   # This must be accompanied by a (maximum) precision and a scale in the
   # SchemaElement. The precision specifies the number of digits in the decimal
   # and the scale stores the location of the decimal point. For example 1.23
   # would have precision 3 (3 total digits) and scale 2 (the decimal point is
   # 2 digits over).
   #
  DECIMAL = 5,

   # Stored as days since Unix epoch, encoded as the INT32 physical type.
  DATE = 6,

   # The total number of milliseconds since midnight.  The value is stored 
   # as an INT32 physical type.
   #
  TIME_MILLIS = 7,

   # Date and time recorded as milliseconds since the Unix epoch.  Recorded as
   # a physical type of INT64.
   #
  TIMESTAMP_MILLIS = 9,

   # The number describes the maximum number of meainful data bits in 
   # the stored value. 8, 16 and 32 bit values are stored using the 
   # INT32 physical type.  64 bit values are stored using the INT64
   # physical type.
  UINT_8 = 11,
  UINT_16 = 12,
  UINT_32 = 13,
  UINT_64 = 14,

   # The number describes the maximum number of meainful data bits in
   # the stored value. 8, 16 and 32 bit values are stored using the
   # INT32 physical type.  64 bit values are stored using the INT64
   # physical type.
  INT_8 = 15,
  INT_16 = 16,
  INT_32 = 17,
  INT_64 = 18,

   # A JSON document embedded within a single UTF8 column.
  JSON = 19,

   # A BSON document embedded within a single BINARY column. 
  BSON = 20,

   # This type annotates data stored as a FIXED_LEN_BYTE_ARRAY of length 12
   # This data is composed of three separate little endian unsigned
   # integers.  Each stores a component of a duration of time.  The first
   # integer identifies the number of months associated with the duration,
   # the second identifies the number of days associated with the duration
   # and the third identifies the number of milliseconds associated with 
   # the provided duration.  This duration of time is independent of any
   # particular timezone or date.
  INTERVAL = 21)

types_i = invert_dict(types)

