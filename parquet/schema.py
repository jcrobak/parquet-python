"""Utils for working with the parquet thrift models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
from collections import OrderedDict

import thriftpy


THRIFT_FILE = os.path.join(os.path.dirname(__file__), "parquet.thrift")
parquet_thrift = thriftpy.load(THRIFT_FILE, module_name=str("parquet_thrift"))  # pylint: disable=invalid-name

logger = logging.getLogger("parquet")  # pylint: disable=invalid-name


class SchemaHelper(object):
    """Utility providing convenience methods for schema_elements."""

    def __init__(self, schema_elements):
        """Initialize with the specified schema_elements."""
        self._se_paths = paths(schema_elements)
        self.schema_elements = schema_elements
        self.schema_elements_by_path = OrderedDict(
            [(tuple(self._se_paths[idx]), se) for idx, se in enumerate(schema_elements)])
        assert len(self.schema_elements) == len(self.schema_elements_by_path)

    def schema_element(self, path):
        """Get the schema element with the given name."""
        return self.schema_elements_by_path[tuple(path)]

    def leaf_node_dict(self):
        """Get a dict of path -> schema_elements."""
        return OrderedDict(
            [(tuple(self.path_for_index(idx)), s) for idx, s in enumerate(self.schema_elements) if s.type])

    def path_for_index(self, index):
        """Get the path array for the schema_element at the given index."""
        return self._se_paths[index]

    def is_required(self, path):
        """Return true iff the schema element with the given name is required."""
        return self.schema_element(path).repetition_type == parquet_thrift.FieldRepetitionType.REQUIRED

    def max_repetition_level(self, path):
        """Get the max repetition level for the given schema path."""
        max_level = 0
        partial_path = []
        for part in path:
            partial_path += [part]
            element = self.schema_element(partial_path)
            if element.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
                max_level += 1
        return max_level

    def max_definition_level(self, path):
        """Get the max definition level for the given schema path."""
        max_level = 0
        partial_path = []
        for part in path:
            partial_path += [part]
            element = self.schema_element(partial_path)
            if element.repetition_type != parquet_thrift.FieldRepetitionType.REQUIRED:
                max_level += 1
        return max_level


def paths(elements):
    """Compute the paths for all the elements.

    the returned value is a map from index -> list of name parts.
    """
    root = elements[0]
    idx = 1
    p_names = {0: [root.name]}
    while idx < len(elements):
        idx = _path_names(elements, idx, [], p_names)

    return p_names


def _path_names(elements, idx, parents, p_names):
    """Internal recursive function to compute pathnames."""
    element = elements[idx]
    logger.debug("%s ... %s", parents, element.name)
    num_children = element.num_children or 0
    p_names[idx] = [s.name for s in parents] + [element.name]
    if num_children == 0:
        return idx + 1

    next_idx = idx + 1
    for _ in range(element.num_children):
        next_idx = _path_names(elements, next_idx, parents + [element], p_names)
    return next_idx
