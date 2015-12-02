# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:45:54 2015

@author: mdurant
"""
from __future__ import absolute_import, division, print_function
from parquet import main as mparquet
from parquet.converted_types import convert_column
from collections import defaultdict
import pandas as pd
import numpy as np
import io
import struct
from parquet.encoding import np_dtypes
import parquet.schema


def schema_full_names(schema):
    """Rationalize schema names as given in column chunk metadata.
    Probably inverse of how the "children" were assigned in the first place."""
    level = 0
    prior = []
    children = []
    schema[0].fullname = 'Root'
    for s in schema[1:]:  # ignore root node
        s.fullname = '.'.join(prior + [s.name.decode()])
        if s.num_children is not None:
            level += 1
            prior.append(s.name.decode())
            children.append(s.num_children)
        elif level > 0:
            children[-1] -= 1
            if children[-1] == 0:
                prior.pop(-1)
                children.pop(-1)
                level -= 1

class ParquetFile(object):
    "Represents parquet file. Schema is read on init."
    def __init__(self, filename):
        "Access and analyze parquet file."
        self.fo = open(filename, 'rb')
        self.footer = mparquet._read_footer(self.fo)
        self.schema_helper = parquet.schema.SchemaHelper(self.footer.schema)
        self.rg = self.footer.row_groups
        self.rows = [row.num_rows for row in self.rg]
        self.cg = self.rg[0].columns
        self.schema = [s for s in self.footer.schema if s.num_children is None]
        schema_full_names(self.footer.schema)
        self.cols = [".".join(x.decode() for x in c.meta_data.path_in_schema) for c in
                         self.cg]
        self.rows = self.footer.num_rows

    def get_columns(self, columns=None):
        """
        Load given columns as a dataframe.
        
        Columns is either a list (a subset of self.cols), or if None,
        gets all columns.
        
        Will attempt to transform 'Converted' types.
        """
        columns = columns or self.cols
        res = defaultdict(list)
        # Alternative to appending values to a list is to make arrays
        # beforehand using the schema, and assign
        for rg in self.rg:
            # Alternative to reading whole file: iterate over row-groups
            # or be able to limit max number of rows returned
            cg = rg.columns
            for col in cg:
                name = ".".join(x.decode() for x in col.meta_data.path_in_schema)
                ind = [s for s in self.schema if s.fullname==name]
                width = ind[0].type_length
                cmd = col.meta_data
                cmd.width = width
                if name not in columns:
                    continue
                if cmd.type == 7:
                    arr = np.empty(rg.num_rows, dtype=np.dtype('S%i'%width))
                else:
                    arr = np.empty(rg.num_rows, dtype=np_dtypes[cmd.type])
                offset = mparquet._get_offset(cmd)
                self.fo.seek(offset, 0)
                values_seen = 0
                dict_items = []
                while values_seen < rg.num_rows:
                    ph = mparquet._read_page_header(self.fo)
                    if ph.type == mparquet.PageType.DATA_PAGE:
                        mparquet.read_data_page(self.fo,
                                self.schema_helper, ph, cmd, dict_items,
                                arr, values_seen)
                        values_seen += ph.data_page_header.num_values
                    else:
                        dict_items = mparquet.read_dictionary_page(
                                self.fo, ph, cmd, width)
                res[name].append(arr)
        res = {key:np.concatenate(d) for key, d in res.items()}
        out = pd.DataFrame(res)
        for col in columns:
            schemae = [s for s in self.schema if col==s.name.decode()][0]
            if schemae.converted_type:
                out[col] = convert_column(out[col], schemae)
        return out

from parquet.ttypes import (FileMetaData, CompressionCodec, Encoding,
                    FieldRepetitionType, PageHeader, PageType, Type,
                    SchemaElement, RowGroup, ColumnChunk, ColumnMetaData,
                    DataPageHeader, PageHeader)
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
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
        fo.write(b'PAR1')
        footer = io.BytesIO()
        tin = TTransport.TFileObjectTransport(footer)
        pin = TCompactProtocol.TCompactProtocol(tin)
        fmd = FileMetaData(num_rows=len(df), created_by=b'python-parquet',
                           schema=[], row_groups=[])
        rg = RowGroup(num_rows=len(df), columns=[])
        fmd.schema.append(SchemaElement(type=0, name=b'Root', num_children=len(df.columns)))
        for col in df:
            binary = df[col].values.tostring()
            typ = df[col].dtype
            if typ == 'int64':
                typcode = 2
            if typ == 'int32':
                typcode = 1
            if typ == 'float64':
                typcode = 5
            fmd.schema.append(SchemaElement(type=typcode, name=col.encode()))
            cmd = ColumnMetaData(type=typcode, encodings=[0], path_in_schema=[col.encode()],
                                 codec=0, num_values=len(df), total_uncompressed_size=len(binary),
                                 total_compressed_size=len(binary), data_page_offset=fo.tell())
            chunk = ColumnChunk(file_offset=fo.tell()+len(binary), meta_data=cmd)
            rg.columns.append(chunk)
            dph = DataPageHeader( num_values=len(df), encoding=0)
            ph = PageHeader(type=0, uncompressed_page_size=len(binary), compressed_page_size=len(binary),
                            data_page_header=dph)
            tin = TTransport.TFileObjectTransport(fo)
            now = TCompactProtocol.TCompactProtocol(tin)
            ph.write(now)
            fo.write(binary)
        fmd.row_groups.append(rg)
        rg.total_byte_size = fo.tell() - 4
        fmd.write(pin)
        binary = footer.getvalue()
        footer_size = len(binary)
        fo.write(binary)
        fo.write(struct.pack('<i', footer_size))
        fo.write(b'PAR1')
    return ParquetFile(filename)

if __name__ == '__main__':
    import os, time
    df = pd.DataFrame({'x': np.arange(10, dtype='int64'),
                       'y': np.arange(10, dtype='int32'),
                       'z': np.arange(10, dtype='float64')})
    df_to_parquet(df, 'temp.parquet')
    f = ParquetFile('temp.parquet')
    f.get_columns()
    t0 = time.time()
    f = ParquetFile(os.sep.join([os.path.expanduser('~'), 'try.parquet']))
    out = f.get_columns()
    t1 = time.time()
    f2 = ParquetFile('/Users/mdurant/Downloads/parquet-data/impala/1.1.1-GZIP/customer.impala.parquet')
    out2 = f2.get_columns()
    t2 = time.time()
    print(t1-t0, t2-t1)