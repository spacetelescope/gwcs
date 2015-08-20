# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test regions
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from numpy.testing import utils
from astropy.modeling import models
from astropy.tests.helper import pytest
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
    for i in np.arange(9) *.1:
        c0_0, c1_0, c0_1, c1_1 = np.ones((4,)) * i
        m.append(models.Polynomial2D(2, c0_0=c0_0,
                                     c1_0=c1_0, c0_1=c0_1, c1_1=c1_1))
    keys = np.array([[  4.88,   5.77],
                     [  5.64,   6.67],
                     [  6.5 ,   7.7 ],
                     [  7.47,   8.83],
                     [  8.63,  10.19],
                     [  9.96,  11.77],
                     [ 11.49,  13.55],
                     [ 13.28,  15.66],
                     [ 15.34,  18.09]])

    rmapper = {}
    for k, v in zip(keys, m):
        rmapper[tuple(k)] = v

    sel = selector.LabelMapperRange(('x', 'y'), rmapper,
                                   inputs_mapping=models.Mapping((0,), n_inputs=2))
    return sel


def test_LabelMapperDict():
    m = []
    for i in np.arange(5) *.1:
        c0_0, c1_0, c0_1, c1_1 = np.ones((4,)) * i
        m.append(models.Polynomial2D(2, c0_0=c0_0,
                                     c1_0=c1_0, c0_1=c0_1, c1_1=c1_1))
    keys = [ -1.95805483e+00,  -1.67833272e+00,  -1.39861060e+00,
             -1.11888848e+00,  -8.39166358e-01]

    dmapper = {}
    for k, v in zip(keys, m):
        dmapper[k] = v

    sel = selector.LabelMapperDict(('x', 'y'), dmapper,
                                   inputs_mapping=models.Mapping((0,), n_inputs=2))
    assert(sel(-1.9580, 2) == dmapper[-1.95805483](-1.95805483, 2))


def test_LabelMapperRange():
    sel = create_range_mapper()
    assert(sel(6, 2) == 2.1)


def test_outside_range():
    """
    Overlapping ranges should raise an error.
    """
    sel = create_range_mapper()
    key = sel.mapper.keys()[0]
    newkey = (key[0], key[1]+5)
    sel.mapper[newkey] = sel.mapper[key]
    with pytest.raises(ValueError):
        sel(key[0] + .5, 3)
