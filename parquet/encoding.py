import math
import struct
import io
import numpy as np

np_dtypes = {
    0: np.dtype('bool'),
    1: np.dtype('int32'),
    2: np.dtype('int64'),
    3: np.dtype('S12'),
    4: np.dtype('float32'),
    5: np.dtype('float64'),
    6: np.dtype("O"),
    }


def read_plain(bit, type_, type_length, num=-1):
    if type_ == 7:  # Fixed-length byte-strings
        d = np.dtype('S%i'%type_length)
    else:
        d = np_dtypes[type_]
    if type_ == 6 and num < 0:  # unknown number of var-length strings
        pos = 0
        out = []
        while pos < len(bit):
            length = struct.unpack('<l', bit[pos:pos+4])[0]
            pos += 4
            out.append(bit[pos:pos+length])
            pos += length
        arr = np.array(out)
    elif type_ == 6:            # known number of var-length strings
        arr = np.empty(num, dtype="O")
        pos = 0
        for i in range(num):
            length = struct.unpack('<l', bit[pos:pos+4])[0]
            pos += 4
            arr[i] = bit[pos:pos+length]
            pos += length            
    else:
        arr = np.fromstring(bit, dtype=d, count=num)            
    return arr


def read_unsigned_var_int(fo):
    result = 0
    shift = 0
    while True:
        byte = struct.unpack("<B", fo.read(1))[0]
        result |= ((byte & 0x7F) << shift)
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result


def byte_width(bit_width):
    "Returns the byte width for the given bit_width"
    return (bit_width + 7) / 8


def read_rle_old(fo, header, bit_width):
    """Read a run-length encoded run from the given fo with the given header
    and bit_width.

    The count is determined from the header and the width is used to grab the
    value that's repeated. Yields the value repeated count times.
    """
    count = header >> 1
    zero_data = b"\x00\x00\x00\x00"
    data = b""
    width = byte_width(bit_width)
    if width >= 1:
        data += fo.read(1)
    if width >= 2:
        data += fo.read(1)
    if width >= 3:
        data += fo.read(1)
    if width == 4:
        data += fo.read(1)
    data = data + zero_data[len(data):]
    value = struct.unpack("<i", data)[0]
    return count, value

if hasattr(int, 'from_bytes'):
    b2int = lambda x: int.from_bytes(x, 'big')
else:
    import codecs
    b2int = lambda x: int(codecs.encode(x, 'hex'), 16)


def read_rle(fo, header, bit_width):
    count = header >> 1
    width = (bit_width+7) // 8
    data = fo.read(min([width, 4]))
    value = int.from_bytes(data, 'little')
    return [value] * count    


def width_from_max_int(value):
    """Converts the value specified to a bit_width."""
    return int(math.ceil(math.log(value + 1, 2)))


def read_bitpacked(fo, header, width):
    num_groups = header >> 1
    count = num_groups * 8
    byte_count = int((width * count)/8)
    if width == 8:
        res = np.frombuffer(fo.read(byte_count), dtype=np.uint8)
    elif width == 16:
        res = np.frombuffer(fo.read(byte_count), dtype=np.uint16)
    else:
        arr = np.frombuffer(fo.read(byte_count), dtype=np.uint8)
        bits = np.unpackbits(arr)
        rarr = bits.reshape((-1, 8))[:, ::-1].ravel()
        count = len(rarr) // width
        rearr = rarr[:width*count].reshape((-1, width))
        res = (2**np.arange(width) * rearr).sum(axis=1)
    return res.tolist()      


def read_rle_bit_packed_hybrid(fo, width, length=None):
    """Implemenation of a decoder for the rel/bit-packed hybrid encoding.

    If length is not specified, then a 32-bit int is read first to grab the
    length of the encoded data.
    """
    io_obj = fo
    if length is None:
        length = struct.unpack('<l', fo.read(4))[0]
        raw_bytes = fo.read(length)
        if raw_bytes == '':
            return None
        io_obj = io.BytesIO(raw_bytes)
    res = []
    while io_obj.tell() < length:
        header = read_unsigned_var_int(io_obj)
        if header & 1 == 0:
            res += read_rle(io_obj, header, width)
        else:
            res += read_bitpacked(io_obj, header, width)
    return res
