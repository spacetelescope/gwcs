# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from numpy.testing import assert_array_equal
from astropy.modeling import models
from astropy.modeling.core import Model
from astropy.utils.misc import isiterable

from asdf import yamlutil
from asdf.tags.core.ndarray import NDArrayType
from ..gwcs_types import GWCSTransformType


from ..selector import *


__all__ = ['LabelMapperType', 'RegionsSelectorType']


class LabelMapperType(GWCSTransformType):
    name = "label_mapper"
    types = [LabelMapperArray, LabelMapperDict, LabelMapperRange, LabelMapper]
    version = "1.1.0"

    @classmethod
    def from_tree_transform(cls, node, ctx):
        inputs_mapping = node.get('inputs_mapping', None)
        if inputs_mapping is not None and not isinstance(inputs_mapping, models.Mapping):
            raise TypeError("inputs_mapping must be an instance"
                            "of astropy.modeling.models.Mapping.")
        mapper = node['mapper']
        atol = node.get('atol', 10**-8)
        no_label = node.get('no_label', np.nan)

        if isinstance(mapper, NDArrayType):
            if mapper.ndim != 2:
                raise NotImplementedError(
                    "GWCS currently only supports 2x2 masks ")
            return LabelMapperArray(mapper, inputs_mapping)
        elif isinstance(mapper, Model):
            inputs = node.get('inputs')
            return LabelMapper(inputs, mapper, inputs_mapping=inputs_mapping, no_label=no_label)
        else:
            inputs = node.get('inputs', None)
            if inputs is not None:
                inputs = tuple(inputs)
            labels = mapper.get('labels')
            transforms = mapper.get('models')
            if isiterable(labels[0]):
                labels = [tuple(l) for l in labels]
                dict_mapper = dict(zip(labels, transforms))
                return LabelMapperRange(inputs, dict_mapper, inputs_mapping)
            else:
                dict_mapper = dict(zip(labels, transforms))
                return LabelMapperDict(inputs, dict_mapper, inputs_mapping, atol=atol)

    @classmethod
    def to_tree_transform(cls, model, ctx):
        node = OrderedDict()
        node['no_label'] = model.no_label
        if model.inputs_mapping is not None:
            node['inputs_mapping'] = model.inputs_mapping

        if isinstance(model, LabelMapperArray):
            node['mapper'] = model.mapper
        elif isinstance(model, LabelMapper):
            node['mapper'] = model.mapper
            node['inputs'] = list(model.inputs)
        elif isinstance(model, (LabelMapperDict, LabelMapperRange)):
            if hasattr(model, 'atol'):
                node['atol'] = model.atol
            mapper = OrderedDict()
            labels = list(model.mapper)

            transforms = []
            for k in labels:
                transforms.append(model.mapper[k])
            if isiterable(labels[0]):
                labels = [list(l) for l in labels]
            mapper['labels'] = labels
            mapper['models'] = transforms
            node['mapper'] = mapper
            node['inputs'] = list(model.inputs)
        else:
            raise TypeError("Unrecognized type of LabelMapper - {0}".format(model))

        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        # TODO: If models become comparable themselves, remove this.
        assert (a.__class__ == b.__class__) # nosec
        if isinstance(a.mapper, dict):
            assert(a.mapper.__class__ == b.mapper.__class__) # nosec
            assert(all(np.in1d(list(a.mapper), list(b.mapper)))) # nosec
            for k in a.mapper:
                assert (a.mapper[k].__class__ == b.mapper[k].__class__) # nosec
                assert(all(a.mapper[k].parameters == b.mapper[k].parameters))  # nosec
            assert (a.inputs == b.inputs) # nosec
            assert (a.inputs_mapping.mapping == b.inputs_mapping.mapping) # nosec
        else:
            assert_array_equal(a.mapper, b.mapper)


class RegionsSelectorType(GWCSTransformType):
    name = "regions_selector"
    types = [RegionsSelector]
    version = "1.1.0"

    @classmethod
    def from_tree_transform(cls, node, ctx):
        inputs = node['inputs']
        outputs = node['outputs']
        label_mapper = node['label_mapper']
        undefined_transform_value = node['undefined_transform_value']
        sel = node['selector']
        sel = dict(zip(sel['labels'], sel['transforms']))
        return RegionsSelector(inputs, outputs,
                               sel, label_mapper, undefined_transform_value)

    @classmethod
    def to_tree_transform(cls, model, ctx):
        selector = OrderedDict()
        node = OrderedDict()
        labels = list(model.selector)
        values = []
        for l in labels:
            values.append(model.selector[l])
        selector['labels'] = labels
        selector['transforms'] = values
        node['inputs'] = list(model.inputs)
        node['outputs'] = list(model.outputs)
        node['selector'] = selector
        node['label_mapper'] = model.label_mapper
        node['undefined_transform_value'] = model.undefined_transform_value
        return yamlutil.custom_tree_to_tagged_tree(node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        # TODO: If models become comparable themselves, remove this.
        assert (a.__class__ == b.__class__) # nosec
        LabelMapperType.assert_equal(a.label_mapper, b.label_mapper)
        assert_array_equal(a.inputs, b.inputs)
        assert_array_equal(a.outputs, b.outputs)
        assert_array_equal(a.selector.keys(), b.selector.keys())
        for key in a.selector:
            assert_array_equal(a.selector[key].parameters, b.selector[key].parameters)
        assert_array_equal(a.undefined_transform_value, b.undefined_transform_value)
