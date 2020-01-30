# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectroscopy related models.
"""

import numbers
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


def _is_int(n):
    return isinstance(n, numbers.Integral) and not isinstance(n, bool)


class SphericalToCartesian(Model):
    """
    Convert spherical coordinates on a unit sphere to cartesian coordinates.
    Spherical coordinates, when not provided as ``Quantity``, are assumed
    to be in degrees with ``phi`` being the *azimuthal angle* ``[0, 360)``
    (or ``[-180, 180)``) and ``theta`` being the *elevation angle*
    ``[-90, 90]`` or the *inclination (polar) angle* in range
    ``[0, 180]`` depending on the used definition.

    """
    _separable = False

    _input_units_strict = True
    _input_units_allow_dimensionless = True

    n_inputs = 2
    n_outputs = 3

    def __init__(self, wrap_phi_at=360, theta_def='elevation', **kwargs):
        """
        Parameters
        ----------
        wrap_phi_at : {180, 360}, optional
            Specifies the range of the azimuthal angle. When ``wrap_phi_at`` is
            180, azimuthal angle will have a range of ``[-180, 180)`` and
            when ``wrap_phi_at`` is 360 (default), the azimuthal angle will have
            a range of ``[0, 360)``

        theta_def : {'elevation', 'inclination', 'polar'}, optional
            Specifies the definition used for the ``theta``: 'elevation' from
            the reference plane or 'inclination' (or 'polar') angle measured
            from the pole.

        """
        super().__init__(**kwargs)
        self.wrap_phi_at = wrap_phi_at
        self.theta_def = theta_def
        self.inputs = ('phi', 'theta')
        self.outputs = ('x', 'y', 'z')

    @property
    def wrap_phi_at(self):
        return self._wrap_phi_at

    @wrap_phi_at.setter
    def wrap_phi_at(self, wrap_angle):
        if not _is_int(wrap_angle):
            raise TypeError("'wrap_phi_at' must be an integer number: 180 or 360")
        if wrap_angle not in [180, 360]:
            raise ValueError("Allowed 'wrap_phi_at' values are 180 and 360")
        self._wrap_phi_at = wrap_angle

    @property
    def theta_def(self):
        return self._theta_def

    @theta_def.setter
    def theta_def(self, theta_def):
        if not isinstance(theta_def, str):
            raise TypeError("'theta_def' must be a string.")
        if theta_def not in ['elevation', 'inclination', 'polar']:
            raise ValueError("Allowed 'theta_def' values are: 'elevation', "
                             "'inclination', or 'polar'.")
        self._theta_def = theta_def

    def evaluate(self, phi, theta):
        if isinstance(phi, u.Quantity) != isinstance(theta, u.Quantity):
            raise TypeError("All arguments must be of the same type "
                            "(i.e., quantity or not).")

        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

        if self._theta_def == 'elevation':
            cs = np.cos(theta)
            si = np.sin(theta)
        else:
            cs = np.sin(theta)
            si = np.cos(theta)

        x = cs * np.cos(phi)
        y = cs * np.sin(phi)
        z = si

        return x, y, z

    def inverse(self):
        return CartesianToSpherical(
            wrap_phi_at=self._wrap_phi_at,
            theta_def=self._theta_def
        )

    @property
    def input_units(self):
        return {'phi': u.deg, 'theta': u.deg}


class CartesianToSpherical(Model):
    """
    Convert cartesian coordinates to spherical coordinates on a unit sphere.
    Spherical coordinates are assumed to be in degrees with ``phi`` being
    the *azimuthal angle* ``[0, 360)`` (or ``[-180, 180)``) and ``theta``
    being the *elevation angle* ``[-90, 90]`` or the *inclination (polar)
    angle* in range ``[0, 180]`` depending on the used definition.

    """
    _separable = False

    n_inputs = 3
    n_outputs = 2

    def __init__(self, wrap_phi_at=360, theta_def='elevation', **kwargs):
        """
        Parameters
        ----------
        wrap_phi_at : {180, 360}, optional
            Specifies the range of the azimuthal angle. When ``wrap_phi_at`` is
            180, azimuthal angle will have a range of ``[-180, 180)`` and
            when ``wrap_phi_at`` is 360 (default), the azimuthal angle will have
            a range of ``[0, 360)``

        theta_def : {'elevation', 'inclination', 'polar'}, optional
            Specifies the definition used for the ``theta``: 'elevation' from
            the reference plane or 'inclination' (or 'polar') angle measured
            from the pole.

        """
        super().__init__(**kwargs)
        self.wrap_phi_at = wrap_phi_at
        self.theta_def = theta_def
        self.inputs = ('x', 'y', 'z')
        self.outputs = ('phi', 'theta')

    @property
    def wrap_phi_at(self):
        return self._wrap_phi_at

    @wrap_phi_at.setter
    def wrap_phi_at(self, wrap_angle):
        if not _is_int(wrap_angle):
            raise TypeError("'wrap_phi_at' must be an integer number: 180 or 360")
        if wrap_angle not in [180, 360]:
            raise ValueError("Allowed 'wrap_phi_at' values are 180 and 360")
        self._wrap_phi_at = wrap_angle

    @property
    def theta_def(self):
        return self._theta_def

    @theta_def.setter
    def theta_def(self, theta_def):
        if not isinstance(theta_def, str):
            raise TypeError("'theta_def' must be a string.")
        if theta_def not in ['elevation', 'inclination', 'polar']:
            raise ValueError("Allowed 'theta_def' values are: 'elevation', "
                             "'inclination', or 'polar'.")
        self._theta_def = theta_def

    def evaluate(self, x, y, z):
        nquant = [isinstance(i, u.Quantity) for i in (x, y, z)].count(True)
        if nquant in [1, 2]:
            raise TypeError("All arguments must be of the same type "
                            "(i.e., quantity or not).")

        h = np.hypot(x, y)
        phi = np.rad2deg(np.arctan2(y, x))
        if h == 0.0:
            phi *= 0.0

        if self._wrap_phi_at != 180:
            phi = np.mod(phi, 360.0 * u.deg if nquant else 360.0)

        theta = np.rad2deg(np.arctan2(z, h))
        if self._theta_def != 'elevation':
            theta = (90.0 * u.deg if nquant else 90.0) - theta

        return phi, theta

    def inverse(self):
        return SphericalToCartesian(
            wrap_phi_at=self._wrap_phi_at,
            theta_def=self._theta_def
        )
