# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np

from astropy.modeling import models as astmodels
from astropy.tests.helper import pytest
from gwcs import selector
from pyasdf.tests import helpers



def test_regions_selector(tmpdir):
    m1 = astmodels.Mapping([0, 1, 1]) | astmodels.Shift(1) & astmodels.Shift(2) & astmodels.Shift(3)
    m2 = astmodels.Mapping([0, 1, 1]) | astmodels.Scale(2) & astmodels.Scale(3) & astmodels.Scale(3)
    sel = {1:m1, 2:m2}
    a = np.zeros((5,6), dtype=np.int32)
    a[:, 1:3] = 1
    a[:, 4:5] = 2
    mask = selector.SelectorMask(a)
    rs = selector.RegionsSelector(inputs=['x','y'], outputs=['ra', 'dec', 'lam'], selector=sel, mask=mask)
    tree = {'model': rs}
    helpers.assert_roundtrip_tree(tree, tmpdir)


def test_selector_mask_str(tmpdir):
    a = np.array([["label1", "", "label2"],
                  ["label1", "", ""],
                  ["label1", "label2", "label2"]])
    mask = selector.SelectorMask(a)
    tree = {'model': mask}
    helpers.assert_roundtrip_tree(tree, tmpdir)


def test_selector_mask_int(tmpdir):

    a = np.array([[1, 0, 2],
                  [1, 0, 0],
                  [1, 2, 2]])
    mask = selector.SelectorMask(a)
    tree = {'model': mask}
    helpers.assert_roundtrip_tree(tree, tmpdir)