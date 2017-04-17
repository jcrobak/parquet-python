"""thrift_filetransport.py - read thrift encoded data from a file object."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from thriftpy.transport import TTransportBase, TTransportException


class TFileTransport(TTransportBase):  # pylint: disable=too-few-public-methods
    """TTransportBase implementation for decoding data from a file object."""

    def __init__(self, fo):
        """Initialize with `fo`, the file object to read from."""
        self._fo = fo
        self._pos = fo.tell()
        self._read = fo.read
        self.read = fo.read
        self._write = fo.write
        self.write = fo.write
