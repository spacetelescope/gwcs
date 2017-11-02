# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test regions
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from numpy.testing import utils
from astropy.modeling import models
import pytest
from .. import region, selector


def test_LabelMapperArray_from_vertices_int():
    regions = {1: [[795, 970], [2047, 970], [2047, 999], [795, 999], [795, 970]],
               2: [[844, 1067], [2047, 1067], [2047, 1113], [844, 1113], [844, 1067]],
               3: [[654, 1029], [2047, 1029], [2047, 1078], [654, 1078], [654, 1029]],
               4: [[772, 990], [2047, 990], [2047, 1042], [772, 1042], [772, 990]]
               }
    mask = selector.LabelMapperArray.from_vertices((2400, 2400), regions)
    labels = list(regions.keys())
    labels.append(0)
    mask_labels = np.unique(mask.mapper).tolist()
    assert(np.sort(labels) == np.sort(mask_labels)).all()


def test_LabelMapperArray_from_vertices_string():
    regions = {'S1600A1': [[795, 970], [2047, 970], [2047, 999], [795, 999], [795, 970]],
               'S200A1': [[844, 1067], [2047, 1067], [2047, 1113], [844, 1113], [844, 1067]],
               'S200A2': [[654, 1029], [2047, 1029], [2047, 1078], [654, 1078], [654, 1029]],
               'S400A1': [[772, 990], [2047, 990], [2047, 1042], [772, 1042], [772, 990]]
               }
    mask = selector.LabelMapperArray.from_vertices((1400, 1400), regions)
    labels = list(regions.keys())
    labels.append('')
    mask_labels = np.unique(mask.mapper).tolist()
    assert(np.sort(labels) == np.sort(mask_labels)).all()


#### These tests below check the scanning algorithm for two shapes ##########
def polygon1(shape=(9, 9)):
    ar = np.zeros(shape)
    ar[1, 2] = 1
    ar[2][2:4] = 1
    ar[3][1:4] = 1
    ar[4][:4] =1
    ar[5][1:4] =1
    ar[6][2:7] =1
    ar[7][3:6] =1
    #ar[8][3:4] =1 ##need to include this in the future if padding top and left
    return ar


def two_polygons():
    ar = np.zeros((301, 301))
    ar[1, 2] = 1
    ar[2][2:4] = 1
    ar[3][1:4] = 1
    ar[4][:4] = 1
    ar[5][1:4] = 1
    ar[6][2:7] = 1
    ar[7][3:6] = 1
    ar[:30, 10:31] = 2
    return ar


def test_polygon1():
    vert = [(2, 1), (3, 5), (6, 6), (3, 8), (0, 4), (2, 1)]
    pol = region.Polygon('1', vert)
    mask = np.zeros((9, 9), dtype=np.int)
    mask = pol.scan(mask)
    pol1 = polygon1()
    utils.assert_equal(mask, pol1)


def test_create_mask_two_polygons():
    vertices = {1: [[2, 1], [3, 5], [6, 6], [3, 8], [0, 4], [2, 1]],
                2: [[10, 0], [30, 0], [30, 30], [10, 30], [10, 0]]}
    mask = selector.LabelMapperArray.from_vertices((301, 301), vertices)
    pol2 = two_polygons()
    utils.assert_equal(mask.mapper, pol2)


def create_range_mapper():
    m = []
    for i in np.arange(1, 10) *.1:
        c0_0, c1_0, c0_1, c1_1 = np.ones((4,)) * i
        m.append(models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1))

    keys = np.array([[  4.88,   5.64],
                     [  5.75,   6.5],
                     [  6.67,   7.47 ],
                     [  7.7,   8.63],
                     [  8.83,  9.96],
                     [  10.19  ,  11.49],
                     [ 11.77,  13.28],
                     [ 13.33,  15.34],
                     [ 15.56,  18.09]])

    rmapper = {}
    for k, v in zip(keys, m):
        rmapper[tuple(k)] = v

    sel = selector.LabelMapperRange(('x', 'y'), rmapper,
                                   inputs_mapping=models.Mapping((0,), n_inputs=2))
    return sel


def create_scalar_mapper():
    m = []
    for i in np.arange(5) *.1:
        c0_0, c1_0, c0_1, c1_1 = np.ones((4,)) * i
        m.append(models.Polynomial2D(2, c0_0=c0_0,
                                     c1_0=c1_0, c0_1=c0_1, c1_1=c1_1))
    keys = [ -1.95805483,  -1.67833272,  -1.39861060,
             -1.11888848,  -8.39166358]

    dmapper = {}
    for k, v in zip(keys, m):
        dmapper[k] = v
    return dmapper


def test_LabelMapperDict():
    dmapper = create_scalar_mapper()
    sel = selector.LabelMapperDict(('x', 'y'), dmapper, atol=10**-3,
                                   inputs_mapping=models.Mapping((0,), n_inputs=2))
    assert(sel(-1.9580, 2) == dmapper[-1.95805483](-1.95805483, 2))


def test_LabelMapperRange():
    sel = create_range_mapper()
    assert(sel(6, 2) == 4.2)


def test_overalpping_ranges():
    """
    Initializing a ``LabelMapperRange`` with overlapping ranges should raise an error.
    """
    keys = np.array([[  4.88,   5.75],
                     [  5.64,   6.5],
                     [  6.67,   7.47 ]])
    rmapper = {}
    for key in keys:
        rmapper[tuple(key)] = models.Const1D(4)
    with pytest.raises(ValueError):
        lmr = selector.LabelMapperRange(('x', 'y'), rmapper, inputs_mapping=((0,)))


def test_outside_range():
    """
    Return ``_no_label`` value when keys are outside the range.
    """
    lmr = create_range_mapper()
    assert lmr(1, 1) == 0
    assert lmr(5, 1) == 1.2

