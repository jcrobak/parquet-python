from fastparquet.util import thrift_copy
from fastparquet import parquet_thrift


def test_thrift_copy():
    fmd = parquet_thrift.FileMetaData()
    rg0 = parquet_thrift.RowGroup()
    rg0.num_rows = 5
    rg1 = parquet_thrift.RowGroup()
    rg1.num_rows = 15
    fmd.row_groups = [rg0, rg1]

    fmd2 = thrift_copy(fmd)

    assert fmd is not fmd2
    assert fmd == fmd2
    assert fmd2.row_groups[0] is not rg0
    rg0.num_rows = 25
    assert fmd2.row_groups[0].num_rows == 5
