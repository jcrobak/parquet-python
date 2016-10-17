from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip
import io
import json
import logging
import numpy as np
import os
import struct
import sys
from collections import OrderedDict, defaultdict

import snappy
import thriftpy
from thriftpy.protocol.compact import TCompactProtocolFactory

from . import encoding
from . import schema
from .thrift_filetransport import TFileTransport
from .thrift_structures import parquet_thrift
from .compression import decompress_data

PY3 = sys.version_info > (3,)
logger = logging.getLogger("parquet")  # pylint: disable=invalid-name


class ParquetFormatException(Exception):
    """Generic Exception related to unexpected data format when
     reading parquet file."""
    pass


def read_thrift(file_obj, ttype):
    """Read a thrift structure from the given fo."""
    tin = TFileTransport(file_obj)
    pin = TCompactProtocolFactory().get_protocol(tin)
    page_header = ttype()
    page_header.read(pin)
    return page_header


def _get_name(type_, value):
    """Return the name for the given value of the given type_.

    The value `None` returns empty string.
    """
    return type_._VALUES_TO_NAMES[value] if value is not None else "None"  # pylint: disable=protected-access


def _get_offset(cmd):
    """Return the offset into the cmd based upon if it's a dictionary page or a data page."""
    dict_offset = cmd.dictionary_page_offset
    data_offset = cmd.data_page_offset
    if dict_offset is None or data_offset < dict_offset:
        return data_offset
    return dict_offset


def _read_page(file_obj, page_header, column_metadata):
    """Read the data page from the given file-object and convert it to raw, uncompressed bytes (if necessary)."""
    raw_bytes = file_obj.read(page_header.compressed_page_size)
    raw_bytes = decompress_data(raw_bytes, column_metadata.codec)

    assert len(raw_bytes) == page_header.uncompressed_page_size, \
        "found {0} raw bytes (expected {1})".format(
            len(raw_bytes),
            page_header.uncompressed_page_size)
    return raw_bytes


def _read_data(file_obj, fo_encoding, value_count, bit_width):
    """Read data from the file-object using the given encoding.

    The data could be definition levels, repetition levels, or actual values.
    """
    vals = []
    if fo_encoding == parquet_thrift.Encoding.RLE:
        seen = 0
        while seen < value_count:
            values = encoding.read_rle_bit_packed_hybrid(file_obj, bit_width)
            if values is None:
                break  # EOF was reached.
            vals += values
            seen += len(values)
    elif fo_encoding == parquet_thrift.Encoding.BIT_PACKED:
        raise NotImplementedError("Bit packing not yet supported")

    return vals


def read_data_page(file_obj, schema_helper, page_header, column_metadata,
                   dictionary):
    """Read the data page from the given file-like object based upon the parameters.

    Metadata in the the schema_helper, page_header, column_metadata, and (optional) dictionary
    are used for parsing data.

    Returns a list of values.
    """
    daph = page_header.data_page_header
    raw_bytes = _read_page(file_obj, page_header, column_metadata)
    io_obj = io.BytesIO(raw_bytes)

    # definition levels are skipped if data is required.
    definition_levels = None
    num_nulls = 0
    if not schema_helper.is_required(column_metadata.path_in_schema[-1]):
        max_definition_level = schema_helper.max_definition_level(
            column_metadata.path_in_schema)
        bit_width = encoding.width_from_max_int(max_definition_level)
        if bit_width:
            definition_levels = _read_data(io_obj,
                                           daph.definition_level_encoding,
                                           daph.num_values,
                                           bit_width)[:daph.num_values]
        num_nulls = len(definition_levels) - definition_levels.count(max_definition_level)
        if num_nulls == 0:
            definition_levels = None

    # repetition levels are skipped if data is at the first level.
    repetition_levels = None  # pylint: disable=unused-variable
    if len(column_metadata.path_in_schema) > 1:
        max_repetition_level = schema_helper.max_repetition_level(
            column_metadata.path_in_schema)
        bit_width = encoding.width_from_max_int(max_repetition_level)
        repetition_levels = _read_data(io_obj,
                                       daph.repetition_level_encoding,
                                       daph.num_values,
                                       bit_width)
    # NOTE: The repetition levels aren't yet used.

    if daph.encoding == parquet_thrift.Encoding.PLAIN:
        read_values = encoding.read_plain(io_obj, column_metadata.type, daph.num_values - num_nulls)
        schema_element = schema_helper.schema_element(column_metadata.path_in_schema[-1])
        read_values = convert_column(read_values, schema_element) \
            if schema_element.converted_type is not None else read_values
        if definition_levels:
            itr = iter(read_values)
            vals = [next(itr) if level == max_definition_level
                    else None for level in definition_levels]
        else:
            vals = read_values

    elif daph.encoding == parquet_thrift.Encoding.PLAIN_DICTIONARY:
        # bit_width is stored as single byte.
        bit_width = struct.unpack(b"<B", io_obj.read(1))[0]
        dict_values_bytes = io_obj.read()
        dict_values_io_obj = io.BytesIO(dict_values_bytes)
        # read_values stores the bit-packed values. If there are definition levels and the data contains nulls,
        # the size of read_values will be less than daph.num_values
        read_values = []
        while dict_values_io_obj.tell() < len(dict_values_bytes):
            read_values.extend(encoding.read_rle_bit_packed_hybrid(
                dict_values_io_obj, bit_width, len(dict_values_bytes)))

        if definition_levels:
            itr = iter(read_values)
            # add the nulls into a new array, values, but using the definition_levels data.
            values = [dictionary[next(itr)] if level == max_definition_level
                      else None for level in definition_levels]
        else:
            values = [dictionary[v] for v in read_values]

        # there can be extra values on the end of the array because the last bit-packed chunk may be zero-filled.
        if len(values) > daph.num_values:
            values = values[0: daph.num_values]
        vals = values

    else:
        raise ParquetFormatException("Unsupported encoding: %s",
                                     _get_name(parquet_thrift.Encoding, daph.encoding))
    return vals


def _read_dictionary_page(file_obj, schema_helper, page_header, column_metadata):
    """Read a page containing dictionary data.

    Consumes data using the plain encoding and returns an array of values.
    """
    raw_bytes = _read_page(file_obj, page_header, column_metadata)
    io_obj = io.BytesIO(raw_bytes)
    values = encoding.read_plain(
        io_obj,
        column_metadata.type,
        page_header.dictionary_page_header.num_values
    )
    # convert the values once, if the dictionary is associated with a converted_type.
    schema_element = schema_helper.schema_element(column_metadata.path_in_schema[-1])
    return convert_column(values, schema_element) if schema_element.converted_type is not None else values


def reader(file_obj, footer, columns=None):
    """
    Reader for a parquet file object.

    This function is a generator returning a list of values for each row
    of data in the parquet file.

    :param file_obj: the file containing parquet data
    :param columns: the columns to include. If None (default), all columns
                    are included. Nested values are referenced with "." notation
    """
    if hasattr(file_obj, 'mode') and 'b' not in file_obj.mode:
        logger.error("parquet.reader requires the fileobj to be opened in binary mode!")
    schema_helper = schema.SchemaHelper(footer.schema)
    keys = columns if columns else [s.name for s in
                                    footer.schema if s.type]
    res = defaultdict(list)
    for row_group in footer.row_groups:
        row_group_rows = row_group.num_rows
        for col_group in row_group.columns:
            dict_items = []
            cmd = col_group.meta_data
            # skip if the list of columns is specified and this isn't in it
            if columns and not ".".join(cmd.path_in_schema) in columns:
                continue

            offset = _get_offset(cmd)
            file_obj.seek(offset, 0)
            values_seen = 0
            while values_seen < row_group_rows:
                page_header = read_thrift(file_obj, parquet_thrift.PageHeader)
                if page_header.type == parquet_thrift.PageType.DATA_PAGE:
                    values = read_data_page(file_obj, schema_helper, page_header, cmd,
                                            dict_items)
                    res[".".join(cmd.path_in_schema)] += values
                    values_seen += page_header.data_page_header.num_values
                elif page_header.type == parquet_thrift.PageType.DICTIONARY_PAGE:
                    assert dict_items == []
                    dict_items = _read_dictionary_page(file_obj, schema_helper, page_header, cmd)
                else:
                    logger.info("Skipping unknown page type=%s",
                                _get_name(parquet_thrift.PageType, page_header.type))

    return [res[k] for k in keys if res[k]]
