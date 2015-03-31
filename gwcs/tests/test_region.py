# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test regions
"""
from __future__ import division, print_function

import numpy as np
from numpy.testing import utils

from .. import region, selector


def test_SelectorMask_from_vertices_int():
    regions = {1: [[795, 970], [2047, 970], [2047, 999], [795, 999], [795, 970]],
               2: [[844, 1067], [2047, 1067], [2047, 1113], [844, 1113], [844, 1067]],
               3: [[654, 1029], [2047, 1029], [2047, 1078], [654, 1078], [654, 1029]],
               4: [[772, 990], [2047, 990], [2047, 1042], [772, 1042], [772, 990]]
               }
    mask = selector.SelectorMask.from_vertices((2400, 2400), regions)
    labels = regions.keys()
    labels.append(0)
    mask_labels = np.unique(mask.mask).tolist()
    assert(np.sort(labels) == np.sort(mask_labels)).all()


def test_SelectorMask_from_vertices_string():
    regions = {'S1600A1': [[795, 970], [2047, 970], [2047, 999], [795, 999], [795, 970]],
               'S200A1': [[844, 1067], [2047, 1067], [2047, 1113], [844, 1113], [844, 1067]],
               'S200A2': [[654, 1029], [2047, 1029], [2047, 1078], [654, 1078], [654, 1029]],
               'S400A1': [[772, 990], [2047, 990], [2047, 1042], [772, 1042], [772, 990]]
               }
    mask = selector.SelectorMask.from_vertices((1400, 1400), regions)
    labels = regions.keys()
    labels.append('')
    mask_labels = np.unique(mask.mask).tolist()
    assert(np.sort(labels) == np.sort(mask_labels)).all()


#### These tests below check the scanning algorithm for two shapes ##########
def polygon1(shape=(9, 9)):
    ar = np.zeros(shape)
    ar[1, 2] = 1
    ar[2][2:4] = 1
    ar[3][1:4] = 1
    ar[4][:4] = 1
    ar[5][1:4] = 1
    ar[6][2:7] = 1
    ar[7][3:6] = 1
    # ar[8][3:4] =1 ##need to include this in the future if padding top and left
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
    mask = selector.SelectorMask.from_vertices((301, 301), vertices)
    pol2 = two_polygons()
    #pol2 = np.zeros((301, 301))
    #pol2[1, 2] = 1
    #pol2[2][2:4] = 1
    #pol2[3][1:4] = 1
    #pol2[4][:4] = 1
    #pol2[5][1:4] = 1
    #pol2[6][2:7] = 1
    #pol2[7][3:6] = 1
    #pol2[:30, 10:31] = 2
    utils.assert_equal(mask.mask, pol2)
