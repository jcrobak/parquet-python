import numpy as np
from pandas.core.index import _ensure_index, CategoricalIndex
from pandas.core.internals import BlockManager
from pandas.core.generic import NDFrame
from pandas.core.frame import DataFrame
from pandas.core.index import RangeIndex, Index
from pandas.core.categorical import Categorical, CategoricalDtype


def empty(types, size, cats=None, cols=None, index_type=None, index_name=None):
    """
    Create empty DataFrame to assign into

    Parameters
    ----------
    types: like np record structure, 'i4,u2,f4,f2,f4,M8,m8', or using tuples
        applies to non-categorical columns. If there are only categorical
        columns, an empty string of None will do.
    size: int
        Number of rows to allocate
    cats: dict {col: labels}
        Location and labels for categorical columns, e.g., {1: ['mary', 'mo]}
        will create column index 1 (inserted amongst the numerical columns)
        with two possible values. If labels is an integers, `{'col': 5}`,
        will generate temporary labels using range. If None, or column name
        is missing, will assume 16-bit integers (a reasonable default).
    cols: list of labels
        assigned column names, including categorical ones.

    Returns
    -------
    - dataframe with correct shape and data-types
    - list of numpy views, in order, of the columns of the dataframe. Assign
        to this.
    """
    df = DataFrame()
    views = {}

    cols = cols or range(cols)
    if isinstance(types, str):
        types = types.split(',')
    for t, col in zip(types, cols):
        if str(t) == 'category':
            if cats is None or col not in cats:
                df[str(col)] = Categorical([], categories=range(2**14),
                                           fastpath=True)
            elif isinstance(cats[col], int):
                df[str(col)] = Categorical([], categories=range(cats[col]),
                                           fastpath=True)
            else:  # explicit labels list
                df[str(col)] = Categorical([], categories=cats[col],
                                           fastpath=True)
        else:
            df[str(col)] = np.empty(0, dtype=t)

    if index_type is not None:
        if index_name is None:
            raise ValueError('If using an index, must give an index name')
        if str(index_type) == 'category':
            vals = np.empty(size, dtype='int8')
            if cats is None or index_name not in cats:
                c = range(2**10)
            elif isinstance(cats[index_name], int):
                c = range(cats[index_name])
            else:  # explicit labels list
                c = cats[index_name]
            index = CategoricalIndex(vals, categories=c, fastpath=True)
            views[index_name] = vals
        else:
            index = np.empty(size, dtype=index_type)
            views[index_name] = index

        axes = [df.columns.values.tolist(), index]
    else:
        axes = [df.columns.values.tolist(), RangeIndex(size)]

    # allocate and create blocks
    blocks = []
    for block, col in zip(df._data.blocks, df.columns):
        if isinstance(block.dtype, CategoricalDtype):
            categories = block.values.categories
            code = np.zeros(shape=size, dtype=block.values.codes.dtype)
            values = Categorical(values=code, categories=categories,
                                 fastpath=True)
        else:
            new_shape = (block.values.shape[0], size)
            values = np.empty(shape=new_shape, dtype=block.values.dtype)

        new_block = block.make_block_same_class(
                values=values, placement=block.mgr_locs.as_array)
        blocks.append(new_block)

    # create block manager
    df = DataFrame(BlockManager(blocks, axes))

    # create views
    for block in df._data.blocks:
        dtype = block.dtype
        inds = block.mgr_locs.indexer
        if isinstance(inds, slice):
            inds = list(range(inds.start, inds.stop, inds.step))
        for i, ind in enumerate(inds):
            col = df.columns[ind]
            if str(dtype) == 'category':
                views[col] = block.values._codes
                views[col+'-catdef'] = block.values
            else:
                views[col] = block.values[i]

    df.index.name = index_name
    if str(index_type) == 'category':
        views[index_name+'-catdef'] = df._data.axes[1].values
    return df, views
