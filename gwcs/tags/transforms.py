# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from numpy.testing import assert_array_equal

from pyasdf import yamlutil
from pyasdf.tags.transform.basic import TransformType

from ..models import LookupTable


__all__ = ['LookupTableType']


class LookupTableType(TransformType):
    name = "transform/lookup_table"
    types = [LookupTable]

    @classmethod
    def from_tree_transform(cls, node, ctx):
        lookup_table = node.pop("lookup_table")
        fill_value = node.pop("fill_value", None)
        return LookupTable(lookup_table, fill_value=fill_value, **node)

    @classmethod
    def to_tree_transform(cls, model, ctx):
        node = {}
        node["fill_value"] = model.fill_value
        node["lookup_table"] = model.lookup_table
        node["points"] = list(model.points)
        node["method"] = model.method
        node["bounds_error"] = model.bounds_error

        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        assert (a.__class__ == b.__class__)
        assert_array_equal(a.lookup_table, b.lookup_table)
        assert_array_equal(a.points, b.points)
        assert (a.method == b.method)
        if a.fill_value is None:
            assert (b.fill_value is None)
        else:
            assert(a.fill_value == b.fill_value)
        assert(a.bounds_error == b.bounds_error)

