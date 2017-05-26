"""Tests for SchemaHelper and related functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

from parquet.schema import SchemaHelper

import thriftpy

THRIFT_FILE = os.path.join(os.path.dirname(__file__), "parquet.thrift")
parquet_thrift = thriftpy.load(THRIFT_FILE, module_name="parquet_thrift")  # pylint: disable=invalid-name


class SchemaHelperTest(unittest.TestCase):
    """Tests for the SchemaHelper class."""

    ELEMENTS = [
        parquet_thrift.SchemaElement(name='root', type=None, type_length=None, repetition_type=None, num_children=2,
                                     converted_type=None),
        parquet_thrift.SchemaElement(name='version', type=parquet_thrift.Type.INT64, type_length=None,
                                     repetition_type=parquet_thrift.FieldRepetitionType.OPTIONAL, num_children=None,
                                     converted_type=None),
        parquet_thrift.SchemaElement(name='geo', type=None, type_length=None,
                                     repetition_type=parquet_thrift.FieldRepetitionType.OPTIONAL, num_children=2,
                                     converted_type=None),
        parquet_thrift.SchemaElement(name='version', type=parquet_thrift.Type.INT64, type_length=None,
                                     repetition_type=parquet_thrift.FieldRepetitionType.OPTIONAL, num_children=None,
                                     converted_type=None),
        parquet_thrift.SchemaElement(name='country_code', type=parquet_thrift.Type.BYTE_ARRAY, type_length=None,
                                     repetition_type=parquet_thrift.FieldRepetitionType.OPTIONAL, num_children=None,
                                     converted_type=0)
    ]

    def test_schema_element_by_path(self):
        """Test lookup by path as array."""
        helper = SchemaHelper(SchemaHelperTest.ELEMENTS)
        self.assertEquals(SchemaHelperTest.ELEMENTS[1], helper.schema_element(['version']))
        self.assertEquals(SchemaHelperTest.ELEMENTS[3], helper.schema_element(['geo', 'version']))
        self.assertEquals(SchemaHelperTest.ELEMENTS[4], helper.schema_element(['geo', 'country_code']))

    def test_leaf_node_dict(self):
        """Test retreiving the leaf nodes for a list of elements."""
        helper = SchemaHelper(SchemaHelperTest.ELEMENTS)
        print(helper.leaf_node_dict())
        self.assertEquals(
            set({
                ('version',): SchemaHelperTest.ELEMENTS[1],
                ('geo', 'version'): SchemaHelperTest.ELEMENTS[3],
                ('geo', 'country_code'): SchemaHelperTest.ELEMENTS[4]
            }),
            set(helper.leaf_node_dict())
        )
