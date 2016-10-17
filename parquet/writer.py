
import io
import json
import numpy as np
import os
import pandas as pd
import shutil
import struct
import thriftpy

from .core import TFileTransport, TCompactProtocolFactory, parquet_thrift
from .compression import compress_data, decompress_data

MARKER = b'PAR1'
NaT = np.timedelta64(None).tobytes()  # require numpy version >= 1.7

typemap = {  # primitive type, converted type, bit width (if not standard)
    'bool': (parquet_thrift.Type.BOOLEAN, None, 1),
    'int32': (parquet_thrift.Type.INT32, None, 32),
    'int64': (parquet_thrift.Type.INT64, None, 64),
    'int8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.INT_8, 8),
    'int16': (parquet_thrift.Type.INT64, parquet_thrift.ConvertedType.INT_16, 16),
    'uint8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_8, 8),
    'uint16': (parquet_thrift.Type.INT64, parquet_thrift.ConvertedType.UINT_16, 16),
    'float32': (parquet_thrift.Type.FLOAT, None, 32),
    'float64': (parquet_thrift.Type.DOUBLE, None, 64),
    'float16': (parquet_thrift.Type.FLOAT, None, 16),
}

def find_type(data):
    """ Get appropriate typecodes for column dtype

    Data conversion does not happen here, only at write time.

    The user is expected to transform their data into the appropriate dtype
    before saving to parquet, we will not make any assumptions for them.

    If the dtype is "object" the first ten items will be examined, and is str
    or bytes, will be stored as variable length byte strings; if dict or list,
    (nested data) will be stored with BSON encoding.

    To be stored as fixed-length byte strings, the dtype must be "bytesXX"
    (pandas notation) or "|SXX" (numpy notation)

    In the case of catagoricals, the data type refers to the labels; the data
    (codes) will be stored as int. The labels are usually variable length
    strings.

    BOOLs will be bitpacked using bytearray. To instead keep the default numpy
    representation of one byte per value, change to int8 or uint8 type

    Known types that cannot be represented (must be first converted another
    type or to raw binary): float128, complex

    Parameters
    ----------
    A pandas series.

    Returns
    -------
    - a thrift schema element
    - a thrift typecode to be passed to the column chunk writer

    """
    if str(data.dtype) == 'category':
        dtype = data.cat.categories.dtype
    else:
        dtype = data.dtype
    if dtype.name in typemap:
        type, converted_type, width = typemap[dtype.name]
    elif "S" in str(dtype) or "U" in str(dtype):
        # TODO: check effect of unicode
        type, converted_type, width = (parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY,
                                       None, dtype.itemsize)
    elif dtype == "O":
        if all(isinstance(i, str) for i in data[:10]):
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.UTF8, None)
        elif all(isinstance(i, bytes) for i in data[:10]):
            type, converted_type, width = parquet_thrift.Type.BYTE_ARRAY, None, None
        elif all(isinstance(i, list) for i in data[:10]) or all(isinstance(i, dict) for i in data[:10]):
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.JSON, None)
        else:
            raise ValueError("Data type conversion unknown: %s" % dtype)
    elif str(dtype).startswith("datetime64"):
        type, converted_type, width = (parquet_thrift.Type.INT64,
                                       parquet_thrift.ConvertedType.TIMESTAMP_MICROS, None)
    elif str(dtype).startswith("timedelta64"):
        type, converted_type, width = (parquet_thrift.Type.INT64,
                                       parquet_thrift.ConvertedType.TIME_MICROS, None)
    else:
        raise ValueError("Don't know how to convert data type: %s" % dtype)
    # TODO: pandas has no explicit support for Decimal
    se = parquet_thrift.SchemaElement(name=data.name, type_length=width,
                                      converted_type=converted_type, type=type,
                                      repetition_type=parquet_thrift.FieldRepetitionType.REQUIRED)
    return se, type


def thrift_print(structure, offset=0):
    """
    Handy recursive text ouput for thrift structures
    """
    if not isinstance(structure, thriftpy.thrift.TPayload):
        return str(structure)
    s = str(structure.__class__) + '\n'
    for key in dir(structure):
        if key.startswith('_') or key in ['thrift_spec', 'read', 'write',
                                          'default_spec']:
            continue
        s = s + ' ' * offset + key + ': ' + thrift_print(getattr(structure, key)
                                                         , offset+2) + '\n'
    return s
thriftpy.thrift.TPayload.__str__ = thrift_print
thriftpy.thrift.TPayload.__repr__ = thrift_print


def write_thrift(fobj, thrift):
    """Write binary compact representation of thiftpy structured object

    Parameters
    ----------
    fobj: open file-like object (binary mode)
    thrift: thriftpy object to write

    Returns
    -------
    Number of bytes written
    """
    t0 = fobj.tell()
    tout = TFileTransport(fobj)
    pout = TCompactProtocolFactory().get_protocol(tout)
    thrift.write(pout)
    return fobj.tell() - t0


def encode_plain(data, se):
    """PLAIN encoding; returns byte representation"""
    dtype = data.dtype
    if str(dtype).startswith(('u', 'int', 'float', "S", "|S")):
        return data.values.tobytes()
    elif str(dtype).startswith('datetime'):
        return data.astype('datetime64[ms]').tobytes()
    elif str(dtype).startswith('timed'):
        return data.astype('timedelta64[ms]').tobytes()
    elif se and se.converted_type == parquet_thrift.ConvertedType.UTF8:
        data = data.str.encode('utf8')
    elif se and se.converted_type == parquet_thrift.ConvertedType.JSON:
        data = data.map(json.dumps)
    if data.dtype != "O":
        ValueError('Cannot encode data as PLAIN')
    # encode variable-length byte strings
    return b''.join(data.map(lambda x: struct.pack('<l', len(x)) + x))


def encode_rle(data, width):
    pass


def encode_dict(data):
    width = int(math.ceil(math.log(data.max() + 1, 2)))
    head = struct.pack(b"<B", width)
    return head + encode_rle(data, width)

encode = {
    'PLAIN': encode_plain,
    'RLE': encode_rle,
    'RLE_DICTIONARY': encode_dict
}


def write_column(f, data, selement, encoding='PLAIN', compression=None):
    """
    If f is a filename, opens data-only file to write to

    Returns ColumnChunk structure

    **NULL values are not yet handled**

    Parameters
    ----------
    f: open binary file
    data: pandas Series or numpy (1d) array
    selement: thrift SchemaElement
        produced by ``find_type``
    encoding: one of ``parquet_thift.Encoding``
        if the dtype is categorical, this is ignored and dictionary encoding
        automatically used
    compression: str or None
        if not None, must be one of the keys in ``compression.compress``
    """

    # no NULL handling (but NaNs are allowed)
    definition_data = b""

    # No nested field handling (encode those as J/BSON)
    repetition_data = b""

    rows = len(data)
    cats = False
    name = data.name

    if str(data.dtype) == 'category':
        dph = parquet_thrift.DictionaryPageHeader(
                num_values=len(data.cat.categories),
                encoding=parquet_thrift.Encoding.PLAIN)
        bdata = encode['PLAIN'](pd.Series(data.cat.categories), None)
        l = len(bdata)
        ph = parquet_thrift.PageHeader(
                type=parquet_thrift.PageType.DICTIONARY_PAGE,
                uncompressed_page_size=l, compressed_page_size=l,
                dictionary_page_header=dph, crc=None)

        dict_start = f.tell()
        write_thrift(f, ph)
        f.write(bdata)
        data = data.cat.codes.astype(np.int32)
        cats = True
        encoding = "PLAIN_DICTIONARY"

    start = f.tell()
    bdata = encode[encoding](data, selement)

    dph = parquet_thrift.DataPageHeader(
            num_values=rows,
            encoding=getattr(parquet_thrift.Encoding, encoding),
            definition_level_encoding=parquet_thrift.Encoding.RLE,
            repetition_level_encoding=parquet_thrift.Encoding.BIT_PACKED)
    l0 = len(bdata)

    if compression:
        bdata = compress_data(bdata, compression)
        l1 = len(bdata)
    else:
        l1 = l0

    ph = parquet_thrift.PageHeader(type=parquet_thrift.PageType.DATA_PAGE,
                                   uncompressed_page_size=l0,
                                   compressed_page_size=l1,
                                   data_page_header=dph, crc=None)

    write_thrift(f, ph)
    f.write(bdata)

    compressed_size = f.tell() - start
    uncompressed_size = compressed_size  # why doesn't this matter?

    offset = f.tell()
    try:
        # TODO: these need to be encoded the same as the data
        max, min = data.max(), data.min()
        s = parquet_thrift.Statistics(max=max.tostring(), min=min.tostring(),
                                      null_count=0)
    except:
        s = parquet_thrift.Statistics(max=None, min=None, null_count=0)

    p = [parquet_thrift.PageEncodingStats(
            page_type=parquet_thrift.PageType.DATA_PAGE,
            encoding=parquet_thrift.Encoding.PLAIN, count=1)]

    cmd = parquet_thrift.ColumnMetaData(
            type=selement.type, path_in_schema=[name],
            encodings=[parquet_thrift.Encoding.RLE,
                       parquet_thrift.Encoding.BIT_PACKED,
                       parquet_thrift.Encoding.PLAIN],
            codec=getattr(parquet_thrift.CompressionCodec, compression) if compression else 0,
            num_values=rows, statistics=s,
            data_page_offset=start, encoding_stats=p,
            key_value_metadata=[],
            total_uncompressed_size=uncompressed_size,
            total_compressed_size=compressed_size)
    if cats:
        print("Cats for column", name)
        p.append(parquet_thrift.PageEncodingStats(
                page_type=parquet_thrift.PageType.DICTIONARY_PAGE,
                encoding=parquet_thrift.Encoding.PLAIN, count=1))
        cmd.dictionary_page_offset = dict_start
    chunk = parquet_thrift.ColumnChunk(file_offset=offset,
                                       meta_data=cmd)
    write_thrift(f, chunk)
    return chunk


def make_row_group(f, data, schema, file_path=None, compression=None):
    rows = len(data)
    rg = parquet_thrift.RowGroup(num_rows=rows, total_byte_size=0,
                                 columns=[])

    for col in schema:
        if col.type is not None:
            chunk = write_column(f, data[col.name], col, compression=compression)
            rg.columns.append(chunk)
    rg.total_byte_size = sum([c.meta_data.total_uncompressed_size for c in
                              rg.columns])

    return rg


def make_part_file(partname, data, schema, compression=None):
    with open(partname, 'wb') as f:
        f.write(MARKER)
        rg = make_row_group(f, data, schema, compression=compression)
        fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                          schema=schema,
                                          version=1,
                                          created_by='parquet-python',
                                          row_groups=[rg])
        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)
    for chunk in rg.columns:
        chunk.file_path = os.path.abspath(partname)
    return rg


def write(filename, data, partitions=[0, 500], encoding=parquet_thrift.Encoding.PLAIN,
          compression=None, file_scheme='simple'):
    """ data is a 1d int array for starters

    Provisional parameters
    ----------------------
    filename: string
        File contains everything (if file_scheme='same'), else contains the
        metadata only
    data: pandas-like dataframe
        simply could be dict of numpy arrays (in which case not sure if
        partitions should be allowed)
    partitions: list of row index values to start new row groups
    encoding: single value from parquet_thrift.Encoding, if applied to all
        columns, or dict of name:parquet_thrift.Encoding for a different
        encoding per column.
    file_scheme: 'simple'|'hive'
        If simple: all goes in a single file
        If hive: each row group is in a separate file, and filename contains
        only the metadata
    """
    if file_scheme == 'simple':
        f = open(filename, 'wb')
    else:
        os.makedirs(filename, exist_ok=True)
        f = open(os.path.join(filename, '_metadata'), 'wb')
    f.write(MARKER)
    root = parquet_thrift.SchemaElement(name='schema',
                                        num_children=0)
    fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                      schema=[root],
                                      version=1,
                                      created_by='parquet-python',
                                      row_groups=[])

    for col in data.columns:
        se, type = find_type(data[col])
        fmd.schema.append(se)
        root.num_children += 1

    for i, start in enumerate(partitions):
        end = partitions[i+1] if i < (len(partitions) - 1) else None
        if file_scheme == 'simple':
            rg = make_row_group(f, data[start:end], fmd.schema,
                                compression=compression)
        else:
            partname = os.path.join(filename, 'part.%i.parquet'%i)
            rg = make_part_file(partname, data[start:end], fmd.schema,
                                compression=compression)
        fmd.row_groups.append(rg)

    foot_size = write_thrift(f, fmd)
    f.write(struct.pack(b"<i", foot_size))
    f.write(MARKER)
    f.close()
    if file_scheme != 'simple':
        f = open(os.path.join(filename, '_common_metadata'), 'wb')
        f.write(MARKER)
        fmd.row_groups = []
        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)
        f.close()


def make_unsigned_var_int(result):
    """Byte representation used for length-of-next-block"""
    bit = b''
    while result > 127:
        bit = bit + ((result & 0x7F) | 0x80).to_bytes(1, 'little')
        result >>= 7
    return bit + result.to_bytes(1, 'little')


def make_rle_string(count, value):
    """Byte representation of a single value run: count occurrances of value"""
    import struct
    header = (count << 1)
    header_bytes = make_unsigned_var_int(header)
    val_part = value.to_bytes(1, 'little')
    length = struct.pack('<l', len(header_bytes)+1)
    return length + header_bytes + val_part
