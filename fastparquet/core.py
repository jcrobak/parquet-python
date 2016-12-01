import io
import os
import re
import struct

import numpy as np
import pandas as pd
from thriftpy.protocol.compact import TCompactProtocolFactory

from . import encoding
from .compression import decompress_data
from .converted_types import convert, typemap
from .thrift_filetransport import TFileTransport
from .thrift_structures import parquet_thrift
from .util import val_to_num


def read_thrift(file_obj, ttype):
    """Read a thrift structure from the given fo."""
    tin = TFileTransport(file_obj)
    pin = TCompactProtocolFactory().get_protocol(tin)
    page_header = ttype()
    page_header.read(pin)
    return page_header


def _read_page(file_obj, page_header, column_metadata):
    """Read the data page from the given file-object and convert it to raw, uncompressed bytes (if necessary)."""
    raw_bytes = file_obj.read(page_header.compressed_page_size)
    raw_bytes = decompress_data(raw_bytes, column_metadata.codec)

    assert len(raw_bytes) == page_header.uncompressed_page_size, \
        "found {0} raw bytes (expected {1})".format(
            len(raw_bytes),
            page_header.uncompressed_page_size)
    return raw_bytes


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
                    daph.num_values, bit_width)[:daph.num_values]
        num_nulls = daph.num_values - (definition_levels ==
                                       max_definition_level).sum()
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
                                      daph.num_values,
                                      bit_width)[:daph.num_values]
    return repetition_levels


def read_data_page(f, helper, header, metadata, skip_nulls=False):
    """Read a data page: definitions, repetitions, values (in order)

    Only values are guaranteed to exist, e.g., for a top-level, required
    field.
    """
    daph = header.data_page_header
    raw_bytes = _read_page(f, header, metadata)
    io_obj = encoding.Numpy8(np.frombuffer(memoryview(raw_bytes),
                                           dtype=np.uint8))

    if skip_nulls and not helper.is_required(metadata.path_in_schema[-1]):
        num_nulls = 0
        definition_levels = None
        skip_definition_bytes(io_obj, daph.num_values)
    else:
        definition_levels, num_nulls = read_def(io_obj, daph, helper, metadata)

    repetition_levels = read_rep(io_obj, daph, helper, metadata)
    if daph.encoding == parquet_thrift.Encoding.PLAIN:
        width = helper.schema_element(metadata.path_in_schema[-1]).type_length
        values = encoding.read_plain(raw_bytes[io_obj.loc:],
                                     metadata.type,
                                     int(daph.num_values - num_nulls),
                                     width=width)
    elif daph.encoding == parquet_thrift.Encoding.PLAIN_DICTIONARY:
        # bit_width is stored as single byte.
        bit_width = io_obj.read_byte()
        if bit_width:
            values = encoding.Numpy32(np.zeros(daph.num_values,
                                               dtype=np.int32))
            # length is simply "all data left in this page"
            encoding.read_rle_bit_packed_hybrid(
                        io_obj, bit_width, io_obj.len-io_obj.loc, o=values)
            values = values.data[:daph.num_values-num_nulls]
        else:
            values = np.zeros(daph.num_values-num_nulls, dtype=np.int64)
    else:
        raise NotImplementedError('Encoding %s' % daph.encoding)
    return definition_levels, repetition_levels, values


def skip_definition_bytes(io_obj, num):
    io_obj.loc += 6
    n = num // 64
    while n:
        io_obj.loc += 1
        n //= 128


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
        width = schema_helper.schema_element(
            column_metadata.path_in_schema[-1]).type_length
        values = encoding.read_plain(
                raw_bytes, column_metadata.type,
                page_header.dictionary_page_header.num_values, width)
    return values


def read_col(column, schema_helper, infile, use_cat=False,
             grab_dict=False, selfmade=False):
    """Using the given metadata, read one column in one row-group.

    Parameters
    ----------
    column: thrift structure
        Details on the column
    schema_helper: schema.SchemaHelper
        Based on the schema for this parquet data
    infile: open file or string
        If a string, will open; if an open object, will use as-is
    use_cat: bool (False)
        If this column is encoded throughout with dict encoding, give back
        a pandas categorical column; otherwise, decode to values
    grab_dict: bool (False)
        Short-cut mode to return the dictionary values only - skips the actual
        data.
    """
    cmd = column.meta_data
    se = schema_helper.schema_element(cmd.path_in_schema[-1])
    off = min((cmd.dictionary_page_offset or cmd.data_page_offset,
               cmd.data_page_offset))

    infile.seek(off)
    ph = read_thrift(infile, parquet_thrift.PageHeader)

    dic = None
    if ph.type == parquet_thrift.PageType.DICTIONARY_PAGE:
        dic = np.array(read_dictionary_page(infile, schema_helper, ph, cmd))
        ph = read_thrift(infile, parquet_thrift.PageHeader)
        dic = convert(dic, se)
    if grab_dict:
        return dic

    rows = cmd.num_values

    out = []
    num = 0
    while True:
        # TODO: under assumption such as all_dict, could allocate once
        # and fill arrays, i.e., merge this loop and the next
        if (selfmade and hasattr(cmd, 'statistics') and
                getattr(cmd.statistics, 'null_count', 1) == 0):
            skip_nulls = True
        else:
            skip_nulls = False
        defi, rep, val = read_data_page(infile, schema_helper, ph, cmd,
                                        skip_nulls)
        d = ph.data_page_header.encoding == parquet_thrift.Encoding.PLAIN_DICTIONARY
        out.append((defi, rep, val, d))
        num += len(defi) if defi is not None else len(val)
        if num >= rows:
            break
        ph = read_thrift(infile, parquet_thrift.PageHeader)

    all_dict = use_cat and all(_[3] for _ in out)
    any_def = any(_[0] is not None for _ in out)
    do_convert = True
    if all_dict:
        dtype = np.int64
        my_nan = -1
        do_convert = False
    else:
        dtype = typemap(se)  # output dtype
        if any_def and dtype.kind == 'i':
            # integers cannot hold NULLs/NaNs
            dtype = np.dtype('float64')
            do_convert = False
        if dtype.kind == 'f':
            my_nan = np.nan
        elif dtype.kind in ["M", 'm']:
            my_nan = -9223372036854775808  # int64 version of NaT
        else:
            my_nan = None
    if len(out) == 1 and not any_def:
        defi, rep, val, d = out[0]
        if d and not all_dict:
            final = dic[val]
        elif do_convert:
            final = convert(val, se)
        else:
            final = val
    else:
        final = np.empty(cmd.num_values, dtype)
        start = 0
        for defi, rep, val, d in out:
            if d and not all_dict:
                cval = dic[val]
            elif do_convert:
                cval = convert(val, se)
            else:
                cval = val
            if defi is not None:
                part = final[start:start+len(defi)]
                part[defi != 1] = my_nan
                part[defi == 1] = cval
                start += len(defi)
            else:
                final[start:start+len(val)] = cval
                start += len(val)
    if all_dict:
        final = pd.Categorical.from_codes(final, categories=dic)
    return final


def read_row_group_file(fn, columns, *args, open=open, selfmade=False,
                        index=None):
    with open(fn, mode='rb') as f:
        return read_row_group(f, columns, *args, selfmade=selfmade,
                              index=index)


def read_row_group_arrays(file, rg, columns, categories, schema_helper, cats,
                          selfmade=False):
    """
    Read a row group and return as a dict of arrays

    Note that categorical columns (if appearing in the parameter categories)
    will be pandas Categorical objects: the codes and the category labels
    are arrays.
    """
    out = {}

    for column in rg.columns:
        name = ".".join(column.meta_data.path_in_schema)
        if name not in columns:
            continue

        use = name in categories if categories is not None else False
        s = read_col(column, schema_helper, file, use_cat=use,
                     selfmade=selfmade)
        out[name] = s
    return out


def read_row_group(file, rg, columns, categories, schema_helper, cats,
                   selfmade=False, index=None):
    """
    Access row-group in a file and read some columns into a data-frame.
    """
    out = read_row_group_arrays(file, rg, columns, categories, schema_helper,
                                cats, selfmade)

    if index is not None and index in columns:
        i = out.pop(index)
        out = pd.DataFrame(out, index=i)
        out.index.name = index
    else:
        out = pd.DataFrame(out, columns=columns)

    # apply categories
    for cat in cats:
        # *Hard assumption*: all chunks in a row group have the
        # same partition (correct for spark/hive)
        partitions = re.findall("([a-zA-Z_]+)=([^/]+)/",
                                rg.columns[0].file_path)
        val = [p[1] for p in partitions if p[0] == cat][0]
        codes = np.empty(rg.num_rows, dtype=np.int16)
        codes[:] = cats[cat].index(val)
        out[cat] = pd.Categorical.from_codes(
                codes, [val_to_num(c) for c in cats[cat]])
    return out
