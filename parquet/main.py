from __future__ import absolute_import, division, print_function
import gzip
import struct
import io
from parquet.ttypes import (FileMetaData, CompressionCodec, Encoding,
                    FieldRepetitionType, PageHeader, PageType, Type)
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from parquet import encoding

try:
    import snappy
except ImportError:
    logger.warn(
        "Couldn't import snappy. Support for snappy compression disabled.")


class ParquetFormatException(Exception):
    pass


def _check_header_magic_bytes(fo):
    "Returns true if the file-like obj has the PAR1 magic bytes at the header"
    fo.seek(0, 0)
    magic = fo.read(4)
    return magic == 'PAR1'


def _check_footer_magic_bytes(fo):
    "Returns true if the file-like obj has the PAR1 magic bytes at the footer"
    fo.seek(-4, 2)  # seek to four bytes from the end of the file
    magic = fo.read(4)
    return magic == 'PAR1'


def _get_footer_size(fo):
    "Readers the footer size in bytes, which is serialized as little endian"
    fo.seek(-8, 2)
    tup = struct.unpack("<i", fo.read(4))
    return tup[0]


def _read_footer(fo):
    """Reads the footer from the given file object, returning a FileMetaData
    object. This method assumes that the fo references a valid parquet file"""
    footer_size = _get_footer_size(fo)
    fo.seek(-(8 + footer_size), 2)  # seek to beginning of footer
    tin = TTransport.TFileObjectTransport(fo)
    pin = TCompactProtocol.TCompactProtocol(tin)
    fmd = FileMetaData()
    fmd.read(pin)
    return fmd


def _read_page_header(fo):
    """Reads the page_header from the given fo"""
    tin = TTransport.TFileObjectTransport(fo)
    pin = TCompactProtocol.TCompactProtocol(tin)
    ph = PageHeader()
    ph.read(pin)
    return ph


def read_footer(filename):
    """Reads and returns the FileMetaData object for the given file."""
    with open(filename, 'rb') as fo:
        if not _check_header_magic_bytes(fo) or \
           not _check_footer_magic_bytes(fo):
            raise ParquetFormatException("{0} is not a valid parquet file "
                                         "(missing magic bytes)"
                                         .format(filename))
        return _read_footer(fo)


def _get_name(type_, value):
    """Returns the name for the given value of the given type_ unless value is
    None, in which case it returns empty string"""
    return type_._VALUES_TO_NAMES[value] if value is not None else "None"


def _get_offset(cmd):
    """Returns the offset into the cmd based upon if it's a dictionary page or
    a data page"""
    dict_offset = cmd.dictionary_page_offset
    data_offset = cmd.data_page_offset
    if dict_offset is None or data_offset < dict_offset:
        return data_offset
    return dict_offset


def _read_page(fo, page_header, column_metadata):
    """Internal function to read the data page from the given file-object
    and convert it to raw, uncompressed bytes (if necessary)."""
    bytes_from_file = fo.read(page_header.compressed_page_size)
    codec = column_metadata.codec
    if codec is not None and codec != CompressionCodec.UNCOMPRESSED:
        if column_metadata.codec == CompressionCodec.SNAPPY:
            raw_bytes = snappy.decompress(bytes_from_file)
        elif column_metadata.codec == CompressionCodec.GZIP:
            io_obj = io.BytesIO(bytes_from_file)
            with gzip.GzipFile(fileobj=io_obj, mode='rb') as f:
                raw_bytes = f.read()
        else:
            raise ParquetFormatException(
                "Unsupported Codec: {0}".format(codec))
    else:
        raw_bytes = bytes_from_file
    assert len(raw_bytes) == page_header.uncompressed_page_size, \
        "found {0} raw bytes (expected {1})".format(
            len(raw_bytes),
            page_header.uncompressed_page_size)
    return raw_bytes


def _read_data(fo, fo_encoding, value_count, bit_width):
    """Internal method to read data from the file-object using the given
    encoding. The data could be definition levels, repetition levels, or
    actual values.
    """
    vals = []
    if fo_encoding == Encoding.RLE:
        seen = 0
        while seen < value_count:
            values = encoding.read_rle_bit_packed_hybrid(fo, bit_width)
            if values is None:
                break  # EOF was reached.
            vals += values
            seen += len(values)
    elif fo_encoding == Encoding.BIT_PACKED:
        raise NotImplementedError("Bit packing not yet supported")

    return vals


def read_data_page(fo, schema_helper, page_header, column_metadata,
                   dictionary, arr, values_seen): 
    """Reads the datapage from the given file-like object based upon the
    metadata in the schema_helper, page_header, column_metadata, and
    (optional) dictionary. Returns a list of values.
    """
    daph = page_header.data_page_header
    raw_bytes = _read_page(fo, page_header, column_metadata)
    io_obj = io.BytesIO(raw_bytes)
    V0 = values_seen
    # definition levels are skipped if data is required.
    if not schema_helper.is_required(column_metadata.path_in_schema[-1]):
        max_definition_level = schema_helper.max_definition_level(
            column_metadata.path_in_schema)
        bit_width = encoding.width_from_max_int(max_definition_level)
        if bit_width == 0:
            definition_levels = [0] * daph.num_values
        else:
            definition_levels = _read_data(io_obj,
                                           daph.definition_level_encoding,
                                           daph.num_values,
                                           bit_width)
            

    # repetition levels are skipped if data is at the first level.
    if len(column_metadata.path_in_schema) > 1:
        max_repetition_level = schema_helper.max_repetition_level(
            column_metadata.path_in_schema)
        bit_width = encoding.width_from_max_int(max_repetition_level)
        repetition_levels = _read_data(io_obj,
                                       daph.repetition_level_encoding,
                                       daph.num_values, bit_width)

    # TODO Actually use the definition and repetition levels.

    if daph.encoding == Encoding.PLAIN:
        raw_bytes = io_obj.read()
        width = getattr(column_metadata, 'width')
        arr[values_seen:values_seen+daph.num_values] = encoding.read_plain(
                        raw_bytes, column_metadata.type, width, daph.num_values)
        values_seen += daph.num_values
    elif daph.encoding == Encoding.PLAIN_DICTIONARY:
        # bit_width is stored as single byte.
        bit_width = struct.unpack("<B", io_obj.read(1))[0]
        total_seen = 0
        dict_values_bytes = io_obj.read()
        dict_values_io_obj = io.BytesIO(dict_values_bytes)
        # TODO jcrobak -- not sure that this loop is needed?
        while total_seen < daph.num_values:
            values = encoding.read_rle_bit_packed_hybrid(
                dict_values_io_obj, bit_width, len(dict_values_bytes))
            if len(values) + total_seen > daph.num_values:
                values = values[0: daph.num_values - total_seen]
            arr[values_seen:values_seen+len(values)] =  [dictionary[v] for v in values]
            values_seen += len(values)
            total_seen += len(values)
    else:
        raise ParquetFormatException("Unsupported encoding: %s",
                                     _get_name(Encoding, daph.encoding))
    if values_seen == V0:
        import pdb
        pdb.set_trace()
    return values_seen


def read_dictionary_page(fo, page_header, column_metadata, width=None):
    raw_bytes = _read_page(fo, page_header, column_metadata)
    dict_items = encoding.read_plain(raw_bytes, column_metadata.type, width, -1)
    return dict_items
