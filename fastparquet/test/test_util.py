import pytest

from fastparquet.util import thrift_copy, analyse_paths
from fastparquet import parquet_thrift


def test_analyse_paths():
    file_list = ['a', 'b']
    base, out = analyse_paths(file_list, '/')
    assert (base, out) == ('', ['a', 'b'])

    file_list = ['c/a', 'c/b']
    base, out = analyse_paths(file_list, '/')
    assert (base, out) == ('c', ['a', 'b'])

    file_list = ['c/d/a', 'c/d/b']
    base, out = analyse_paths(file_list, '/')
    assert (base, out) == ('c/d', ['a', 'b'])

    file_list = ['c/cat=1/a', 'c/cat=2/b', 'c/cat=1/c']
    base, out = analyse_paths(file_list, '/')
    assert (base, out) == ('c', ['cat=1/a', 'cat=2/b', 'cat=1/c'])

    file_list = ['c\\cat=1\\a', 'c\\cat=2\\b', 'c\\cat=1\\c']
    base, out = analyse_paths(file_list, '\\')
    assert (base, out) == ('c', ['cat=1\\a', 'cat=2\\b', 'cat=1\\c'])

    file_list = ['c/cat=2/b', 'c/cat/a', 'c/cat=1/c']
    with pytest.raises(ValueError) as e:
        analyse_paths(file_list, '/')
    assert 'c/cat/a' in str(e)

    file_list = ['c/cat=2/b', 'c/fred=2/a', 'c/cat=1/c']
    with pytest.raises(ValueError) as e:
        analyse_paths(file_list, '/')
    assert 'directories' in str(e)

    file_list = ['c/cat=2/b', 'c/a', 'c/cat=1/c']
    with pytest.raises(ValueError) as e:
        analyse_paths(file_list, '/')
    assert 'nesting' in str(e)


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
