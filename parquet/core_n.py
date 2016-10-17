import io
import numpy as np
import os
import pandas as pd
import struct

from . import encoding
from .converted_types import convert
from .core import read_thrift, parquet_thrift, _read_page


def read_data(fobj, coding, count, bit_width):
    """For definition and repetition levels

    Reads with RLE/bitpacked hybrid, where length is given by first byte.
    """
    out = np.empty(count, dtype=np.int32)
    o = encoding.Numpy32(out)
    if coding == parquet_thrift.Encoding.RLE:
        while o.loc < count:
            encoding.read_rle_bit_packed_hybrid(fobj, bit_width, o=o)
    else:
        raise NotImplementedError('Encoding %s' % coding)
    return out


def read_def(io_obj, daph, helper, metadata):
    """
    Read the definition levels from this page, if any.
    """
    definition_levels = None
    num_nulls = 0
    if not helper.is_required(metadata.path_in_schema[-1]):
        max_definition_level = helper.max_definition_level(
            metadata.path_in_schema)
        bit_width = encoding.width_from_max_int(max_definition_level)
        if bit_width:
            definition_levels = read_data(
                    io_obj, daph.definition_level_encoding,
                    daph.num_values, bit_width)
        num_nulls = len(definition_levels) - (definition_levels==max_definition_level).sum()
        if num_nulls == 0:
            definition_levels = None
    return definition_levels, num_nulls


def read_rep(io_obj, daph, helper, metadata):
    """
    Read the repetition levels from this page, if any.
    """
    repetition_levels = None  # pylint: disable=unused-variable
    if len(metadata.path_in_schema) > 1:
        max_repetition_level = helper.max_repetition_level(
            metadata.path_in_schema)
        bit_width = encoding.width_from_max_int(max_repetition_level)
        repetition_levels = read_data(io_obj, daph.repetition_level_encoding,
                                      daph.num_values, bit_width)
    return repetition_levels


def read_data_page(f, helper, header, metadata):
    """Read a data page: definitions, repetitions, values (in order)

    Only values are guaranteed to exist, e.g., for a top-level, required
    field.
    """
    daph = header.data_page_header
    raw_bytes = _read_page(f, header, metadata)
    io_obj = encoding.Numpy8(np.frombuffer(memoryview(raw_bytes),
                                           dtype=np.uint8))

    definition_levels, num_nulls = read_def(io_obj, daph, helper, metadata)

    repetition_levels = read_rep(io_obj, daph, helper, metadata)
    if daph.encoding == parquet_thrift.Encoding.PLAIN:
        values = encoding.read_plain(raw_bytes[io_obj.loc:],
                                     metadata.type,
                                     int(daph.num_values - num_nulls))
    elif daph.encoding == parquet_thrift.Encoding.PLAIN_DICTIONARY:
        # bit_width is stored as single byte.
        bit_width = io_obj.read_byte()
        values = encoding.Numpy32(np.zeros(daph.num_values,
                                           dtype=np.int32))
        # length is simply "all data left in this page"
        encoding.read_rle_bit_packed_hybrid(
                    io_obj, bit_width, io_obj.len-io_obj.loc, o=values)
        values = values.data[:daph.num_values-num_nulls]
    else:
        raise NotImplementedError('Encoding %s' % daph.encoding)
    return definition_levels, repetition_levels, values


def read_dictionary_page(file_obj, schema_helper, page_header, column_metadata):
    """Read a page containing dictionary data.

    Consumes data using the plain encoding and returns an array of values.
    """
    raw_bytes = _read_page(file_obj, page_header, column_metadata)
    if column_metadata.type == parquet_thrift.Type.BYTE_ARRAY:
        # no faster way to read variable-length-strings?
        fobj = io.BytesIO(raw_bytes)
        values = [fobj.read(struct.unpack(b"<i", fobj.read(4))[0])
                  for _ in range(page_header.dictionary_page_header.num_values)]
    else:
        values = encoding.read_plain(
                raw_bytes, column_metadata.type,
                page_header.dictionary_page_header.num_values)
    return values


def read_col(column, schema_helper, infile):
    """Using the metadata in infile, read one column in one row-group.
    """
    cmd = column.meta_data
    name = ".".join(cmd.path_in_schema)
    rows = cmd.num_values
    off = min((cmd.dictionary_page_offset or cmd.data_page_offset,
               cmd.data_page_offset))
    if column.file_path:
        infile = open(os.path.join(os.path.dirname(os.path.abspath(infile)),
                                   column.file_path), 'rb')
    elif isinstance(infile, str):
        infile = open(infile, 'rb')

    infile.seek(off)
    ph = read_thrift(infile, parquet_thrift.PageHeader)

    dic = None
    if ph.type == parquet_thrift.PageType.DICTIONARY_PAGE:
        dic = np.array(read_dictionary_page(infile, schema_helper, ph, cmd))
        ph = read_thrift(infile, parquet_thrift.PageHeader)

    out = []
    num = 0
    while True:
        defi, rep, val = read_data_page(infile, schema_helper, ph, cmd)
        d = ph.data_page_header.encoding == parquet_thrift.Encoding.PLAIN_DICTIONARY
        out.append((defi, rep, val, d))
        num += len(defi) if defi is not None else len(val)
        if num >= rows:
            break
        ph = read_thrift(infile, parquet_thrift.PageHeader)

    # TODO: the code below could be separate "assemble" func
    all_dict = all(_[3] for _ in out)
    any_dict = any(_[3] for _ in out)
    any_def = any(_[0] is not None for _ in out)
    if any_def:
        # decode definitions
        dtype = encoding.DECODE_TYPEMAP.get(cmd.type, np.object_)
        try:
            # because we have nulls, and ints don't support NaN
            dtype('nan')
        except ValueError:
            dtype = np.float64
        final = np.empty(rows, dtype=(int if all_dict else dtype))
        start = 0
        for o in out:
            defi, rep, val, d = o
            if defi is not None:
                l = len(defi)
                if all_dict:
                    final[start:start+l][defi == 1] = val
                    final[start:start+l][defi != 1] = -1
                elif d:
                    final[start:start+l][defi == 1] = dic[val]
                    final[start:start+l][defi != 1] = np.nan
                else:
                    final[start:start+l][defi == 1] = val
                    final[start:start+l][defi != 1] = np.nan
            else:
                l = len(val)
                if d:
                    final[start:start+l] = dic[val]
                else:
                    final[start:start+l] = val

            start += l
    elif all_dict or not any_dict:
        final = np.concatenate([_[2] for _ in out])
    else:
        final = np.empty(rows, dtype=dic.dtype)
        start = 0
        for o in out:
            defi, rep, val, d = o
            l = len(val)
            if d:
                final[start:start+l] = dic[val]
            else:
                final[start:start+l] = val
            start += l

    if all_dict:
        out = pd.Series(pd.Categorical.from_codes(final, dic), name=name)
    else:
        out = pd.Series(final, name=name)
    se = schema_helper.schema_element(cmd.path_in_schema[-1])
    if se.converted_type is not None:
        out = convert(out, se)
    return out
