# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np

from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter

try:
    from scipy.interpolate import interpn
    has_scipy = True
except ImportError:
    has_scipy = False


__all__ = ['LookupTable']


class LookupTable(Model):
    """
    Returns an interpolated lookup table value.

    Parameters
    ----------
    lookup_table : array_like, shape (m1, ..., mn, ...)
        The data on a regular grid in n dimensions.
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, ), optional
        The points defining the regular grid in n dimensions.
    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest", and "splinef2d". "splinef2d" is only supported for
        2-dimensional data. Default is "linear".
    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
    fill_value : float, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d".

    Returns
    -------
    value : ndarray
        Interpolated values at input coordinates.

    Raises
    ------
    ImportError
        Scipy is not installed.

    Examples
    --------
    >>> from gwcs.models import LookupTable
    >>> table=np.array([[ 3.,  0.,  0.],
                        [ 0.,  2.,  0.],
                        [ 0.,  0.,  0.]])
    >>> points = [1, 2, 3]
    >>> xinterp = [0, 1, 1.5, 2.72, 3.14]

    Setting fill_value to None, allows extrapolation.

    >>> ltmodel = LookupTable(table, points, bounds_error=False, fill_value=None, method='nearest')
    >>> ltmodel(xinterp, xinterp)
        array([ 3.,  3.,  3.,  0.,  0.])

    Notes
    -----
    Uses ``scipy.interpolate.interpn``
    """
    linear = False
    fittable = False

    standard_broadcasting = False
    outputs = ('lookup_table_value',)

    lookup_table = Parameter()

    def __init__(self, lookup_table, points=None, method='linear',
                 bounds_error=True, fill_value=np.nan):
        if not has_scipy:
            raise ImportError("LookupTable model needs scipy.")
        if points is None:
            self._points = tuple(np.arange(x, dtype=np.float) for x in lookup_table.shape)
        else:
            self._points = points
        super(LookupTable, self).__init__(lookup_table=lookup_table)
        self.inputs = tuple('x' + str(idx)
                            for idx in range(self.lookup_table.value.ndim))

        self._bounds_error = bounds_error
        self._method = method
        self._fill_value = fill_value

    @property
    def bounds_error(self):
        return self._bounds_error

    @bounds_error.setter
    def bounds_error(self, value):
        self._bounds_error = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def fill_value(self):
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        self._fill_value = value

    @property
    def points(self):
        return self._points

    def evaluate(self, *inputs):
        inputs = np.array(inputs[: self.n_inputs]).T
        return interpn(self.points, self.lookup_table.value, inputs, method=self.method,
                       bounds_error=self.bounds_error, fill_value=self.fill_value)
