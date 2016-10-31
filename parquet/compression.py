
import gzip
from .thrift_structures import parquet_thrift
# TODO: use stream/direct-to-buffer conversions instead of memcopy

# TODO: enable ability to pass kwargs to compressor

compress = {'GZIP': gzip.compress, 'UNCOMPRESSED': lambda x: x}
decompress = {'GZIP': gzip.decompress, 'UNCOMPRESSED': lambda x: x}
try:
    import snappy
    compress['SNAPPY'] = snappy.compress
    decompress['SNAPPY'] = snappy.decompress
except ImportError:
    pass
try:
    import lzo
    compress['LZO'] = lzo.compress
    decompress['LZO'] =lzo. decompress
except ImportError:
    pass
try:
    import brotli
    compress['BROTLI'] = brotli.compress
    decompress['BROTLI'] = brotli.decompress
except ImportError:
    pass


rev_map = {getattr(parquet_thrift.CompressionCodec, key): key for key in
           dir(parquet_thrift.CompressionCodec) if key in compress}


def compress_data(data, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm not in compress:
        raise RuntimeError("Compression '%s' not available" % compress)
    return compress[algorithm](data)


def decompress_data(data, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm not in compress:
        raise RuntimeError("Decompression '%s' not available" % compress)
    return decompress[algorithm](data)

