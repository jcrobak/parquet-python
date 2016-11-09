
import gzip
import zlib
from .thrift_structures import parquet_thrift
# TODO: use stream/direct-to-buffer conversions instead of memcopy

# TODO: enable ability to pass kwargs to compressor

compress = {'GZIP': gzip.compress,
            'zlib': zlib.compress,
            'UNCOMPRESSED': lambda x: x}

decompress = {'GZIP': gzip.decompress,
              'zlib': zlib.decompress,
              'UNCOMPRESSED': lambda x: x}
try:
    import snappy
    compress['SNAPPY'] = snappy.compress
    decompress['SNAPPY'] = snappy.decompress
except ImportError:
    pass
try:
    import lz4
    compress['lz4'] = lz4.compress
    decompress['lz4'] =lz4.decompress
except ImportError:
    pass
try:
    import lzo
    compress['LZO'] = lzo.compress
    decompress['LZO'] =lzo.decompress
except ImportError:
    pass
try:
    import brotli
    compress['BROTLI'] = brotli.compress
    decompress['BROTLI'] = brotli.decompress
except ImportError:
    pass

compress = {k.lower(): v for k, v in compress.items()}
decompress = {k.lower(): v for k, v in decompress.items()}

rev_map = {getattr(parquet_thrift.CompressionCodec, key): key for key in
           dir(parquet_thrift.CompressionCodec) if key in compress}


def compress_data(data, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm.lower() not in compress:
        raise RuntimeError("Compression '%s' not available.  Options: %s" %
                (algorithm, sorted(compress)))
    return compress[algorithm.lower()](data)


def decompress_data(data, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm.lower() not in decompress:
        raise RuntimeError("Decompression '%s' not available.  Options: %s" %
                (algorithm.lower(), sorted(decompress)))
    return decompress[algorithm.lower()](data)
