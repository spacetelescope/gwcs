# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test separability of WCS axes.

"""
from __future__ import absolute_import, division, unicode_literals, print_function
from astropy.modeling import models
from astropy.modeling.models import Mapping
import pytest
import numpy as np
from numpy.testing import utils

from .. utils import _coord_matrix, is_separable


sh1 = models.Shift(1, name='shift1')
sh2 = models.Shift(2, name='sh2')
scl1 = models.Scale(1, name='scl1')
scl2 = models.Scale(2, name='scl2')
map1 = Mapping([0, 1, 0, 1], name='map1')
map2 = Mapping([0, 0, 1], name='map2')
map3 = Mapping([0, 0], name='map3')
rot = models.Rotation2D(2, name='rotation')
p2 = models.Polynomial2D(1, name='p2')
p22 = models.Polynomial2D(2, name='p22')
p1 = models.Polynomial1D(1, name='p1')


# Note: remove these later when Model.separable is implemented
sh1.separable = True
sh2.separable = True
scl1.separable = True
scl2.separable = True
rot.separable = False
p2.separable = False
p22.separable = False
p1.separable = True


compound_models = {'cm1': (map3 & sh1 | rot & sh1 | sh1 & sh2 & sh1,
                           np.array([False, False, True])),
                   'cm2': (sh1 & sh2 | rot | map1 | p2 & p22,
                           np.array([False, False])),
                   'cm3': (map2 | rot & scl1,
                           np.array([False, False, True])),
                   'cm4': (sh1 & sh2 | map2 | rot & scl1,
                           np.array([False, False, True])),
                   'cm5': (map3 | sh1 & sh2 | scl1 & scl2,
                           np.array([False, False])),
                   'cm7': (map2 | p2 & sh1,
                           np. array([False, True]))
                   }


def test_coord_matrix():
    c = _coord_matrix(p2, 'left', 2)
    utils.assert_allclose(np.array([[1, 1], [0, 0]]), c)
    c = _coord_matrix(p2, 'right', 2)
    utils.assert_allclose(np.array([[0, 0], [1, 1]]), c)
    c = _coord_matrix(p1, 'left', 2)
    utils.assert_allclose(np.array([[1], [0]]), c)
    c = _coord_matrix(p1, 'left', 1)
    utils.assert_allclose(np.array([[1]]), c)
    c = _coord_matrix(sh1, 'left', 2)
    utils.assert_allclose(np.array([[1], [0]]), c)
    c = _coord_matrix(sh1, 'right', 2)
    utils.assert_allclose(np.array([[0], [1]]), c)
    c = _coord_matrix(sh1, 'right', 3)
    utils.assert_allclose(np.array([[0], [0], [1]]), c)
    c = _coord_matrix(map3, 'left', 2)
    utils.assert_allclose(np.array([[1], [1]]), c)
    c = _coord_matrix(map3, 'left', 3)
    utils.assert_allclose(np.array([[1], [1], [0]]), c)


@pytest.mark.parametrize(('compound_model', 'result'), compound_models.values())
def test_separable(compound_model, result):
    utils.assert_allclose(is_separable(compound_model), result)

