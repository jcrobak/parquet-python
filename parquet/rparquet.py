# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:45:54 2015

@author: mdurant
"""
from __future__ import absolute_import, division, print_function
from parquet import main as parquet
from parquet.converted_types import convert_column
from collections import defaultdict
import pandas as pd

class ParquetFile(object):
    def __init__(self, filename):
        self.fo = open(filename, 'rb')
        self.footer = parquet._read_footer(self.fo)
        self.schema_helper = parquet.schema.SchemaHelper(self.footer.schema)
        self.rg = self.footer.row_groups
        self.rows = [row.num_rows for row in self.rg]
        self.cg = self.rg[0].columns
        self.schema = self.footer.schema
        self.cols = [".".join(x.decode() for x in c.meta_data.path_in_schema) for c in
                         self.cg]

    def get_columns(self, columns=None):
        columns = columns or self.cols
        res = defaultdict(list)
        for rg in self.rg:
            cg = rg.columns
            for col in cg:
                name = ".".join(x.decode() for x in col.meta_data.path_in_schema)
                ind = [s for s in self.schema if s.name.decode()==name]
                width = ind[0].type_length
                if name not in columns:
                    continue
                offset = parquet._get_offset(col.meta_data)
                self.fo.seek(offset, 0)
                values_seen = 0
                cmd = col.meta_data
                cmd.width = width
                dict_items = []
                while values_seen < rg.num_rows:
                    ph = parquet._read_page_header(self.fo)
                    if ph.type == parquet.PageType.DATA_PAGE:
                        values = parquet.read_data_page(self.fo,
                                self.schema_helper, ph, cmd, dict_items)
                        res[name] += values
                        values_seen += ph.data_page_header.num_values
                    else:
                        dict_items = parquet.read_dictionary_page(
                                self.fo, ph, cmd, width)
        out = pd.DataFrame(res)
        for col in columns:
            schemae = [s for s in self.schema if col==s.name.decode()][0]
            if schemae.converted_type:
                out[col] = convert_column(out[col], schemae)
        return out

if __name__ == '__main__':
    import os
    f = ParquetFile(os.sep.join([os.path.expanduser('~'), 'try.parquet']))
    out = f.get_columns()