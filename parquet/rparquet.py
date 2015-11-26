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
import numpy as np
from parquet.encoding import np_dtypes

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
        self.footer = parquet._read_footer(self.fo)
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
                offset = parquet._get_offset(cmd)
                self.fo.seek(offset, 0)
                values_seen = 0
                dict_items = []
                while values_seen < rg.num_rows:
                    ph = parquet._read_page_header(self.fo)
                    if ph.type == parquet.PageType.DATA_PAGE:
                        parquet.read_data_page(self.fo,
                                self.schema_helper, ph, cmd, dict_items,
                                arr, values_seen)
                        values_seen += ph.data_page_header.num_values
                    else:
                        dict_items = parquet.read_dictionary_page(
                                self.fo, ph, cmd, width)
                res[name].append(arr)
        res = {key:np.concatenate(d) for key, d in res.items()}
        out = pd.DataFrame(res)
        for col in columns:
            schemae = [s for s in self.schema if col==s.name.decode()][0]
            if schemae.converted_type:
                out[col] = convert_column(out[col], schemae)
        return out

if __name__ == '__main__':
    import os, time
    t0 = time.time()
    f = ParquetFile(os.sep.join([os.path.expanduser('~'), 'try.parquet']))
    out = f.get_columns()
    t1 = time.time()
    f2 = ParquetFile('/Users/mdurant/Downloads/parquet-data/impala/1.1.1-GZIP/customer.impala.parquet')
    out2 = f2.get_columns()
    t2 = time.time()
    print(t1-t0, t2-t1)