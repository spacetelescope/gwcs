# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectroscopy related models.
"""

import numpy as np
from astropy.modeling.core import Model
from astropy import units as u

__all__ = ['ToDirectionCosines', 'FromDirectionCosines',
           'SphericalToCartesian', 'CartesianToSpherical']


class ToDirectionCosines(Model):
    """
    Transform a vector to direction cosines.
    """
    _separable = False

    n_inputs = 3
    n_outputs = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('x', 'y', 'z')
        self.outputs = ('cosa', 'cosb', 'cosc', 'length')

    def evaluate(self, x, y, z):
        vabs = np.sqrt(1. + x**2 + y**2)
        cosa = x / vabs
        cosb = y / vabs
        cosc = 1. / vabs
        return cosa, cosb, cosc, vabs

    def inverse(self):
        return FromDirectionCosines()


class FromDirectionCosines(Model):
    """
    Transform directional cosines to vector.
    """
    _separable = False

    n_inputs = 4
    n_outputs = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('cosa', 'cosb', 'cosc', 'length')
        self.outputs = ('x', 'y', 'z')

    def evaluate(self, cosa, cosb, cosc, length):

        return cosa * length, cosb * length, cosc * length

    def inverse(self):
        return ToDirectionCosines()


class SphericalToCartesian(Model):
    """
    Convert spherical coordinates on a unit sphere to cartesian coordinates.
    Spherical coordinates, when not provided as ``Quantity``, are assumed
    to be in degrees with ``phi`` being the *azimuthal* angle ``[0, 360)``
    and ``theta`` being the *elevation* angle ``[-90, 90]``.

    """
    _separable = False

    _input_units_strict = True
    _input_units_allow_dimensionless = True

    n_inputs = 2
    n_outputs = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('phi', 'theta')
        self.outputs = ('x', 'y', 'z')

    @staticmethod
    def evaluate(phi, theta):
        if isinstance(phi, u.Quantity) != isinstance(theta, u.Quantity):
            raise TypeError("All arguments must be of the same type "
                            "(i.e., quantity or not).")

        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

        cs = np.cos(theta)
        x = cs * np.cos(phi)
        y = cs * np.sin(phi)
        z = np.sin(theta)

        return x, y, z

    def inverse(self):
        return CartesianToSpherical()

    @property
    def input_units(self):
        return {'phi': u.deg, 'theta': u.deg}


class CartesianToSpherical(Model):
    """
    Convert cartesian coordinates to spherical coordinates on a unit sphere.
    Spherical coordinates are assumed to be in degrees with ``phi`` being
    the *azimuthal* angle ``[0, 360)`` and ``theta`` being the *elevation*
    angle ``[-90, 90]``.

    """
    _separable = False

    n_inputs = 3
    n_outputs = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('x', 'y', 'z')
        self.outputs = ('phi', 'theta')

    @staticmethod
    def evaluate(x, y, z):
        nquant = [isinstance(i, u.Quantity) for i in (x, y, z)].count(True)
        if nquant in [1, 2]:
            raise TypeError("All arguments must be of the same type "
                            "(i.e., quantity or not).")

        h = np.hypot(x, y)
        phi = np.mod(
            np.rad2deg(np.arctan2(y, x)),
            360.0 * u.deg if nquant else 360.0
        )
        theta = np.rad2deg(np.arctan2(z, h))

        return phi, theta

    def inverse(self):
        return SphericalToCartesian()
