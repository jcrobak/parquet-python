"""parquet - read parquet files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import pandas as pd
import re
import struct
import thriftpy

from .core import ParquetFormatException, read_thrift
from .thrift_structures import parquet_thrift
from .writer import write
from . import core, schema, converted_types


def def_open(f):
    return open(f, 'rb')


class ParquetFile(object):
    """For now: metadata representation

    Parameters
    ----------
    fname: path/URL string
    verify: test file start/end bytes
    open_with: function returning an open file
    """

    def __init__(self, fname, verify=False, open_with=def_open):
        if isinstance(fname, str):
            try:
                # backend may not have something equivalent to `isdir()`
                f = open_with(fname)
            except (IOError, OSError):
                fname = os.path.join(fname, '_metadata')
                f = open_with(fname)
        else:
            f = fname
            self.fname = str(fname)
        self.open = open_with
        self.fname = fname
        self.f = f
        self._parse_header(f, verify)
        self._read_partitions()

    def _parse_header(self, f, verify=True):
        try:
            f.seek(0)
            if verify:
                assert f.read(4) == b'PAR1'
            f.seek(-8, 2)
            head_size = struct.unpack('<i', f.read(4))[0]
            if verify:
                assert f.read() == b'PAR1'
        except (AssertionError, struct.error):
            raise ParquetFormatException('File parse failed: %s' % self.fname)

        f.seek(-(head_size+8), 2)
        try:
            fmd = read_thrift(f, parquet_thrift.FileMetaData)
        except thriftpy.transport.TTransportException:
            raise ParquetFormatException('Metadata parse failed: %s' %
                                         self.fname)
        self.fmd = fmd
        self.head_size = head_size
        self.version = fmd.version
        self.schema = fmd.schema
        self.row_groups = fmd.row_groups or []
        self.key_value_metadata = fmd.key_value_metadata
        self.created_by = fmd.created_by
        self.group_files = {}
        for i, rg in enumerate(self.row_groups):
            for chunk in rg.columns:
                self.group_files.setdefault(i, set()).add(chunk.file_path)

    @property
    def columns(self):
        return [f.name for f in self.schema if f.num_children is None]

    def _read_partitions(self):
        cats = {}
        for rg in self.row_groups:
            for col in rg.columns:
                partitions = re.findall("([a-zA-Z_]+)=([^/]+)/",
                                        col.file_path or "")
                for key, val in partitions:
                    cats.setdefault(key, set()).add(val)
        self.cats = {key: list(v) for key, v in cats.items()}

    def to_dask(self, columns=None, usecats=None, **kwargs):
        import dask.dataframe as dd
        cols = columns or (self.columns + list(self.cats))
        tot = [self.read_row_group_delayed(rg, cols, usecats, **kwargs)
               for rg in self.row_groups]
        dtypes = {k: v for k, v in self.dtypes.items() if k in cols}

        # TODO: if categories vary from one rg to next, need to cope
        return dd.from_delayed(tot, metadata=dtypes, divisions=self.divisions)

    def read_row_group_delayed(self, rg, cols, usecats):
        from dask import delayed
        return delayed(self.read_row_group)(rg, cols, usecats)

    def read_row_group(self, rg, cols, usecats):
        out = {}
        fname = self.fname

        for col in rg.columns:
            name = ".".join(col.meta_data.path_in_schema)
            if name not in cols:
                continue

            if col.file_path is None:
                # continue reading from the same base file
                infile = self.f
            else:
                # relative file
                ofname = os.path.join(os.path.dirname(self.fname),
                                      col.file_path)
                if ofname != fname:
                    # open relative file, if not the current one
                    infile = self.open(ofname)
                    fname = ofname

            use = name in usecats if usecats is not None else False
            s = core.read_col(col, schema.SchemaHelper(self.schema),
                              infile, use_cat=use)
            out[name] = s
        out = pd.DataFrame(out)

        # apply categories
        for cat in self.cats:
            # *Hard assumption*: all chunks in a row group have the
            # same partition (correct for spark/hive)
            partitions = re.findall("([a-zA-Z_]+)=([^/]+)/",
                                    rg.columns[0].file_path)
            val = [p[1] for p in partitions if p[0] == cat][0]
            codes = np.empty(rg.num_rows, dtype=np.int16)
            codes[:] = self.cats[cat].index(val)
            out[cat] = pd.Categorical.from_codes(
                    codes, [val_to_num(c) for c in self.cats[cat]])
        return out

    def to_pandas(self, columns=None, usecats=None):
        cols = columns or self.columns
        tot = [self.read_row_group(rg, cols, usecats) for rg in
               self.row_groups]

        # TODO: if categories vary from one rg to next, need
        # pandas.types.concat.union_categoricals
        return pd.concat(tot, ignore_index=True)

    @property
    def count(self):
        return sum(rg.num_rows for rg in self.row_groups)

    @property
    def info(self):
        return {'name': self.fname, 'columns': self.columns,
                'categories': list(self.cats), 'rows': self.count}

    @property
    def dtypes(self):
        dtype = {f.name: converted_types.typemap(f)
                 for f in self.schema if f.num_children is None}
        for cat in self.cats:
            dtype[cat] = pd.Series(self.cats[cat]).map(val_to_num).dtype
        return dtype

    @property
    def divisions(self):
        return np.cumsum([0] + [rg.num_rows for rg in self.row_groups])

    def __str__(self):
        return "<Parquet File '%s'>" % self.fname

    __repr__ = __str__


def val_to_num(x):
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    try:
        return pd.to_datetime(x)
    except ValueError:
        pass
    try:
        return pd.to_timedelta(x)
    except:
        return x
