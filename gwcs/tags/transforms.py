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
        lookup_table = node['lookup_table']
        points = node.get('points', None)
        fill_value = node['fill_value']
        method = node['method']
        bounds_error = node['bounds_error']
        return LookupTable(lookup_table, points, fill_value=fill_value, method=method, bounds_error=bounds_error)

    @classmethod
    def to_tree_transform(cls, model, ctx):
        node = {}
        node['lookup_table'] = model.lookup_table
        node['points'] = list(model.points)
        node['fill_value'] = model.fill_value
        node['method'] = model.method
        node['bounds_error'] = model.bounds_error

        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        # TODO: If models become comparable themselves, remove this.
        assert (a.__class__ == b.__class__)
        assert_array_equal(a.lookup_table, b.lookup_table)
        assert_array_equal(a.points, b.points)
        assert (a.method == b.method)
        if a.fill_value is None:
            assert (b.fill_value is None)
        else:
            assert(a.fill_value == b.fill_value)
        assert(a.bounds_error == b.bounds_error)

