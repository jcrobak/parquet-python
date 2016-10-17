"""parquet - read parquet files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import struct

import thriftpy

from .core import (parquet_thrift, reader, TFileTransport, TCompactProtocolFactory,
                   ParquetFormatException)
from .writer import write
from . import core_n


class ParquetFile(object):
    """For now: metadata representation"""

    def __init__(self, fname, verify=True):
        self.fname = fname
        self.verify = verify
        if isinstance(fname, str):
            if os.path.isdir(fname):
                fname = os.path.join(fname, '_metadata')
            f = open(fname, 'rb')
        else:
            f = fname
        self._parse_header(f, verify)

    def _parse_header(self, f, verify=True):
        try:
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
            fmd = parquet_thrift.FileMetaData()
            tin = TFileTransport(f)
            pin = TCompactProtocolFactory().get_protocol(tin)
            fmd.read(pin)
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

    def to_pandas(self, columns=None):
        cols = columns or self.columns
        import pandas as pd
        tot = []
        for rg in self.row_groups:
            out = pd.DataFrame()
            for col in rg.columns:
                name = ".".join(col.meta_data.path_in_schema)
                if name not in cols:
                    continue
                s = core_n.read_col(col, schema.SchemaHelper(self.schema),
                                    self.fname)
                out[name] = s
            tot.append(out)
        return pd.concat(tot, ignore_index=True)

    def __str__(self):
        return "<Parquet File '%s'>" % self.fname

    __repr__ = __str__
