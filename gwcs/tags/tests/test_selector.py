# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import numpy as np

from astropy.modeling.models import Mapping, Shift, Scale, Polynomial2D
from ... import selector
from asdf.tests import helpers
from ...tests.test_region import create_scalar_mapper
from ...extension import GWCSExtension


def test_regions_selector(tmpdir):
    m1 = Mapping([0, 1, 1]) | Shift(1) & Shift(2) & Shift(3)
    m2 = Mapping([0, 1, 1]) | Scale(2) & Scale(3) & Scale(3)
    sel = {1: m1, 2: m2}
    a = np.zeros((5, 6), dtype=np.int32)
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
                                   inputs_mapping=Mapping((0,), n_inputs=2), atol=1e-3)
    tree = {'model': sel}
    helpers.assert_roundtrip_tree(tree, tmpdir, extensions=GWCSExtension())


def test_LabelMapperRange(tmpdir):
    m = []
    for i in np.arange(9) * .1:
        c0_0, c1_0, c0_1, c1_1 = np.ones((4,)) * i
        m.append(Polynomial2D(2, c0_0=c0_0,
                              c1_0=c1_0, c0_1=c0_1, c1_1=c1_1))
    keys = np.array([[4.88, 5.64],
                     [5.75, 6.5],
                     [6.67, 7.47],
                     [7.7, 8.63],
                     [8.83, 9.96],
                     [10.19, 11.49],
                     [11.77, 13.28],
                     [13.33, 15.34],
                     [15.56, 18.09]])
    rmapper = {}
    for k, v in zip(keys, m):
        rmapper[tuple(k)] = v
    sel = selector.LabelMapperRange(('x', 'y'), rmapper,
                                    inputs_mapping=Mapping((0,), n_inputs=2))
    tree = {'model': sel}
    helpers.assert_roundtrip_tree(tree, tmpdir, extensions=GWCSExtension())
