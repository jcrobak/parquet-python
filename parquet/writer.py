# -*- coding: utf-8 -*-
"""
Write parquet files.
"""

from parquet.ttypes import (FileMetaData, CompressionCodec, Encoding,
                    FieldRepetitionType, PageHeader, PageType, Type,
                    SchemaElement, RowGroup, ColumnChunk, ColumnMetaData,
                    DataPageHeader, PageHeader, IndexPageHeader, DictionaryPageHeader,
                    KeyValue)
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
import io
import struct
"""  BOOLEAN = 0
  INT32 = 1
  INT64 = 2
  INT96 = 3
  FLOAT = 4
  DOUBLE = 5
  BYTE_ARRAY = 6
  FIXED_LEN_BYTE_ARRAY = 7"""

def df_to_parquet(df, filename, index=False):
    with open(filename, 'wb') as fo:
        nrows, ncols = df.shape
        fo.write(b'PAR1')
        fmd = FileMetaData(num_rows=nrows, created_by=b'python-parquet',
                           schema=[], row_groups=[], version=1, key_value_metadata=[])
        rg = RowGroup(num_rows=nrows, columns=[])
        fmd.schema.append(SchemaElement(type=0, name=b'Root', num_children=ncols))
        for col in df:
            binary = df[col].values.tostring()
            typ = df[col].dtype
            if typ == 'int64':
                typcode = 2
            if typ == 'int32':
                typcode = 1
            if typ == 'float64':
                typcode = 5
            start = fo.tell()
            rle_string = make_rle_string(nrows, 1)
            fmd.schema.append(SchemaElement(type=typcode, name=col.encode(), repetition_type=1))
            cmd = ColumnMetaData(type=typcode, encodings=[0], path_in_schema=[col.encode()],
                                 codec=0, num_values=nrows, data_page_offset=start, key_value_metadata=[])
            dph = DataPageHeader(num_values=nrows, encoding=0, definition_level_encoding=3,
                                 repetition_level_encoding=0)
            ph = PageHeader(type=0, uncompressed_page_size=len(binary)+len(rle_string),
                            compressed_page_size=len(binary)+len(rle_string),
                            data_page_header=dph, crc=None)
            tin = TTransport.TFileObjectTransport(fo)
            now = TCompactProtocol.TCompactProtocol(tin)
            ph.write(now)
            fo.write(rle_string)
            fo.write(binary)
            cmd.total_uncompressed_size = fo.tell() - start
            cmd.total_compressed_size = cmd.total_uncompressed_size
            chunk = ColumnChunk(file_offset=start, meta_data=cmd)
            rg.columns.append(chunk)
        fmd.row_groups.append(rg)
        rg.total_byte_size = sum(x.meta_data.total_uncompressed_size for x in
                        rg.columns)
        
        footer = io.BytesIO()
        tin = TTransport.TFileObjectTransport(footer)
        pin = TCompactProtocol.TCompactProtocol(tin)
        fmd.write(pin)
        binary = footer.getvalue()
        footer_size = len(binary)
        fo.write(binary)
        fo.write(struct.pack('<i', footer_size))
        fo.write(b'PAR1')
    from parquet.rparquet import ParquetFile
    return ParquetFile(filename)


def make_unsigned_var_int(result):
    bit = b''
    while result > 127:
        bit = bit + ((result & 0x7F) | 0x80).to_bytes(1, 'little')
        result >>= 7
    return bit + result.to_bytes(1, 'little')

def make_rle_string(count, value):
    import struct
    header = (count << 1) 
    header_bytes = make_unsigned_var_int(header)
    val_part = value.to_bytes(1, 'little')
    length = struct.pack('<l', len(header_bytes)+1)
    return length + header_bytes + val_part
