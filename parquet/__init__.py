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

from .core import (parquet_thrift, reader, TFileTransport, TCompactProtocolFactory,
                   ParquetFormatException, read_thrift)
from .writer import write
from . import core_n


class ParquetFile(object):
    """For now: metadata representation"""

    def __init__(self, fname, verify=True):
        self.verify = verify
        if isinstance(fname, str):
            if os.path.isdir(fname):
                fname = os.path.join(fname, '_metadata')
            f = open(fname, 'rb')
        else:
            f = fname
        self.fname = fname
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
        self.row_groups = fmd.row_groups
        self.key_value_metadata = fmd.key_value_metadata
        self.created_by = fmd.created_by

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

    def to_pandas(self, columns=None):
        cols = columns or self.columns
        import pandas as pd
        tot = []
        for rg in self.row_groups:
            # read values
            out = pd.DataFrame()
            for col in rg.columns:
                name = ".".join(col.meta_data.path_in_schema)
                if name not in cols:
                    continue
                s = core_n.read_col(col, schema.SchemaHelper(self.schema),
                                    self.fname)
                out[name] = s

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
            tot.append(out)

        # TODO: if categories vary from one rg to next, need
        # pandas.types.concat.union_categoricals
        return pd.concat(tot, ignore_index=True)

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
