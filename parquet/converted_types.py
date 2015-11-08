# -*- coding: utf-8 -*-
"""
Deal with parquet logical types (aka converted types), higher-order
things built from primitive types.
"""

# define bytes->int for non 2, 4, 8 byte ints
if hasattr(int, 'from_bytes'):
    b2int = lambda x: int.from_bytes(x, 'big')
else:
    import codecs
    b2int = lambda x: int(codecs.encode(x, 'hex'), 16)


def convert_column(data, schemae):
    """Convert known types from primitive to rich.
    Designed for pandas series."""
    if schemae.converted_type == 5:   # DECIMAL
        scale = 10**schemae.extra[0]
        pecision = schemae.extra[1]   # not used - defined by byte width
        if data.dtype == object:
            out = data.map(b2int) / scale
        else:
            out = data / scale
    else:
        print("Converted type %i not known" % schemae.converted_type)
        out = data
    return out
