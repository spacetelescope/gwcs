# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np

from astropy.modeling.models import Mapping, Shift, Scale
from astropy.tests.helper import pytest
from ... import selector, extension
from pyasdf.tests import helpers
from ...tests.test_region import create_range_mapper, create_scalar_mapper
from ...extension import GWCSExtension


def test_regions_selector(tmpdir):
    m1 = Mapping([0, 1, 1]) | Shift(1) & Shift(2) & Shift(3)
    m2 = Mapping([0, 1, 1]) | Scale(2) & Scale(3) & Scale(3)
    sel = {1:m1, 2:m2}
    a = np.zeros((5,6), dtype=np.int32)
    a[:, 1:3] = 1
    a[:, 4:5] = 2
    mask = selector.LabelMapperArray(a)
    rs = selector.RegionsSelector(inputs=('x', 'y'), outputs=('ra', 'dec', 'lam'),
                                  selector=sel, label_mapper=mask)
    tree = {'model': rs}
    helpers.assert_roundtrip_tree(tree, tmpdir, extensions=GWCSExtension())


def test_LabelMapperArray_str(tmpdir):
    a = np.array([["label1", "", "label2"],
                  ["label1", "", ""],
                  ["label1", "label2", "label2"]])
    mask = selector.LabelMapperArray(a)
    tree = {'model': mask}
    helpers.assert_roundtrip_tree(tree, tmpdir, extensions=GWCSExtension())


def test_labelMapperArray_int(tmpdir):

    a = np.array([[1, 0, 2],
                  [1, 0, 0],
                  [1, 2, 2]])
    mask = selector.LabelMapperArray(a)
    tree = {'model': mask}
    helpers.assert_roundtrip_tree(tree, tmpdir, extensions=GWCSExtension())


def test_LabelMapperDict(tmpdir):
    dmapper = create_scalar_mapper()
    sel = selector.LabelMapperDict(('x', 'y'), dmapper,
                                   inputs_mapping=Mapping((0,), n_inputs=2))
    tree = {'model': sel}
    helpers.assert_roundtrip_tree(tree, tmpdir, extensions=GWCSExtension())


def test_LabelMapperRange(tmpdir):
    rmapper = create_scalar_mapper()
    sel = selector.LabelMapperDict(('x', 'y'), rmapper,
                                   inputs_mapping=Mapping((0,), n_inputs=2))
    tree = {'model': sel}
    helpers.assert_roundtrip_tree(tree, tmpdir, extensions=GWCSExtension())
