# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from numpy.testing import assert_array_equal
from pyasdf import yamlutil
from pyasdf.tags.transform.basic import TransformType

from gwcs import selector


__all__ = ['SelectorMaskType', 'RegionsSelectorType']


class SelectorMaskType(TransformType):
    name = "transform/selector_mask"
    types = [selector.SelectorMask]

    @classmethod
    def from_tree_transform(cls, node, ctx):
        mask = node['mask']
        if mask.ndim != 2:
            raise NotImplementedError(
                "GWCS currently only supports 2x2 masks ")

        return selector.SelectorMask(mask)

    @classmethod
    def to_tree_transform(cls, model, ctx):
        node = {'mask': model.mask}
        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        # TODO: If models become comparable themselves, remove this.
        assert (a.__class__ == b.__class__)
        assert_array_equal(a.mask, b.mask)


class RegionsSelectorType(TransformType):
    name = "transform/regions_selector"
    types = [selector.RegionsSelector]

    @classmethod
    def from_tree_transform(cls, node, ctx):
        inputs = node['inputs']
        outputs = node['outputs']
        mask = node['mask']
        undefined_transform_value = node['undefined_transform_value']
        sel = node['selector']

        return selector.RegionsSelector(inputs, outputs,
                                        sel, mask, undefined_transform_value)

    @classmethod
    def to_tree_transform(cls, model, ctx):
        node = {'inputs': model.inputs, 'outputs': model.outputs,
                'selector': model.selector,
                'mask': model.mask,
                'undefined_transform_value': model.undefined_transform_value}
        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        # TODO: If models become comparable themselves, remove this.
        assert (a.__class__ == b.__class__)
        assert_array_equal(a.mask.mask, b.mask.mask)
        assert_array_equal(a.inputs, b.inputs)
        assert_array_equal(a.outputs, b.outputs)
        assert_array_equal(a.selector.keys(), b.selector.keys())
        for i in a.selector.keys():
            assert_array_equal(a.selector[i].parameters, b.selector[i].parameters)
        assert_array_equal(a.undefined_transform_value, b.undefined_transform_value)
