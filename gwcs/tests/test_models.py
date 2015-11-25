from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pytest

from ..models import LookupTable

def test_interp_1d():
    points = np.arange(0, 5)
    values = [1., 10, 2, 45, -3]
    model = LookupTable(values, points=(points,))
    xnew = [0., .7, 1.4, 2.1, 3.9]
    assert_allclose(model(xnew), [1., 7.3, 6.8, 6.3, 1.8])
    model = LookupTable(values)
    assert_allclose(model(xnew), [1., 7.3, 6.8, 6.3, 1.8])
    with pytest.raises(ValueError):
        model([0., .7, 1.4, 2.1, 3.9, 4.1])
    # test extrapolation and fill value
    model = LookupTable(values, bounds_error=False, fill_value=None)
    assert_allclose(model([0., .7, 1.4, 2.1, 3.9, 4.1]), [ 1. ,  7.3,  6.8,  6.3,  1.8, -7.8])


def test_interp_2d():
    values = np.array([[-0.04614432, -0.02512547, -0.00619557, 0.0144165 , 0.0297525 ],
       [-0.04510594, -0.03183369, -0.01118008, 0.01201388, 0.02496205],
       [-0.05464094, -0.02804499, -0.00960086, 0.01134333, 0.02284104],
       [-0.04879338, -0.02539565, -0.00440462, 0.01795145, 0.02122417],
       [-0.03637372, -0.01630025, -0.00157902, 0.01649774, 0.01952131]])

    points = (np.arange(0, 5), np.arange(0, 5))

    xnew = np.array([0., .7, 1.4, 2.1, 3.9])

    model = LookupTable(values, points)
    znew = model(xnew, xnew)
    result = np.array([-0.04614432, -0.03450009, -0.02241028, -0.0069727 ,  0.01938675])
    assert_allclose(znew, result, atol=10**-7)
