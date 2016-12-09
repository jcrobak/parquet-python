import numpy as np
from pandas.core.index import _ensure_index
from pandas.core.internals import BlockManager
from pandas.core.generic import NDFrame
from pandas.core.frame import DataFrame
from pandas.core.index import RangeIndex
from pandas.core.categorical import Categorical, CategoricalDtype


def empty(types, size, cats={}, cols=None):
    """
    Create empty DataFrame to assign into

    Parameters
    ----------
    types: like np record structure, 'i4,u2,f4,f2,f4,M8,m8', or using tuples
        applies to non-categorical columns. If there are only categorical
        columns, an empty string of None will do.
    size: int
        Number of rows to allocate
    cats: dict
        Location and labels for categorical columns, e.g., {1: ['mary', 'mo]}
        will create column index 1 (inserted amongst the numerical columns)
        with two possible values.
    cols: list of labels
        assigned column names, including categorical ones.

    Returns
    -------
    - dataframe with correct shape and data-types
    - list of numpy views, in order, of the columns of the dataframe. Assign
        to this.
    """
    if types:
        df = DataFrame(np.empty(0, dtype=types))
    else:
        df = DataFrame()
    for k in sorted(cats):
        df.insert(k, k, Categorical([], categories=cats[k]))
    if cols is not None:
        df.columns = cols
    axes = [df.columns.values.tolist(), RangeIndex(size)]

    # allocate and create blocks
    blocks = []
    codes = []
    for block, col in zip(df._data.blocks, df.columns):
        if isinstance(block.dtype, CategoricalDtype):
            categories = block.values.categories
            code = np.empty(shape=size, dtype=block.values.codes.dtype)
            values = Categorical(values=code,categories=categories,
                                 fastpath=True)
            codes.append(code)
        else:
            new_shape = (block.values.shape[0], size)
            values = np.empty(shape=new_shape, dtype=block.dtype)

        new_block = block.make_block_same_class(
                values=values, placement=block.mgr_locs.as_array)
        blocks.append(new_block)

    # create block manager
    df = DataFrame(BlockManager(blocks, axes))

    # create views
    views = []
    for col in df:
        dtype = df[col].dtype
        if str(dtype) == 'category':
            views.append(codes.pop(0))
        else:
            ind = [c for c in df if df.dtypes[c] == dtype].index(col)
            views.append([b for b in blocks if b.dtype == dtype][0].values[ind, :])
    return df, views
