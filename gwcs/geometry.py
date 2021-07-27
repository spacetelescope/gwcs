# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models for general analytical geometry transformations.
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


class SphericalToCartesian(Model):
    """
    Convert spherical coordinates on a unit sphere to cartesian coordinates.
    Spherical coordinates when not provided as ``Quantity`` are assumed
    to be in degrees with ``lon`` being the *longitude (or azimuthal) angle*
    ``[0, 360)`` (or ``[-180, 180)``) and angle ``lat`` is the *latitude*
    (or *elevation angle*) in range ``[-90, 90]``.

    """
    _separable = False

    _input_units_allow_dimensionless = True

    n_inputs = 2
    n_outputs = 3

    def __init__(self, wrap_lon_at=360, **kwargs):
        """
        Parameters
        ----------
        wrap_lon_at : {360, 180}, optional
            An **integer number** that specifies the range of the longitude
            (azimuthal) angle. When ``wrap_lon_at`` is 180, the longitude angle
            will have a range of ``[-180, 180)`` and when ``wrap_lon_at``
            is 360 (default), the longitude angle will have a range of
            ``[0, 360)``.

        """
        super().__init__(**kwargs)
        self.inputs = ('lon', 'lat')
        self.outputs = ('x', 'y', 'z')
        self.wrap_lon_at = wrap_lon_at

    @property
    def wrap_lon_at(self):
        """ An **integer number** that specifies the range of the longitude
        (azimuthal) angle.

        Allowed values are 180 and 360. When ``wrap_lon_at``
        is 180, the longitude angle will have a range of ``[-180, 180)`` and
        when ``wrap_lon_at`` is 360 (default), the longitude angle will have a
        range of ``[0, 360)``.

        """
        return self._wrap_lon_at

    @wrap_lon_at.setter
    def wrap_lon_at(self, wrap_angle):
        if not (isinstance(wrap_angle, numbers.Integral) and wrap_angle in [180, 360]):
            raise ValueError("'wrap_lon_at' must be an integer number: 180 or 360")
        self._wrap_lon_at = wrap_angle

    def evaluate(self, lon, lat):
        if isinstance(lon, u.Quantity) != isinstance(lat, u.Quantity):
            raise TypeError("All arguments must be of the same type "
                            "(i.e., quantity or not).")

        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)

        cs = np.cos(lat)
        x = cs * np.cos(lon)
        y = cs * np.sin(lon)
        z = np.sin(lat)

        return x, y, z

    def inverse(self):
        return CartesianToSpherical(wrap_lon_at=self._wrap_lon_at)

    @property
    def input_units(self):
        return {'lon': u.deg, 'lat': u.deg}


class CartesianToSpherical(Model):
    """
    Convert cartesian coordinates to spherical coordinates on a unit sphere.
    Output spherical coordinates are in degrees. When input cartesian
    coordinates are quantities (``Quantity`` objects), output angles
    will also be quantities in degrees. Angle ``lon`` is the *longitude*
    (or *azimuthal angle*) in range ``[0, 360)`` (or ``[-180, 180)``) and
    angle ``lat`` is the *latitude* (or *elevation angle*) in the
    range ``[-90, 90]``.

    """
    _separable = False

    _input_units_allow_dimensionless = True

    n_inputs = 3
    n_outputs = 2

    def __init__(self, wrap_lon_at=360, **kwargs):
        """
        Parameters
        ----------
        wrap_lon_at : {360, 180}, optional
            An **integer number** that specifies the range of the longitude
            (azimuthal) angle. When ``wrap_lon_at`` is 180, the longitude angle
            will have a range of ``[-180, 180)`` and when ``wrap_lon_at``
            is 360 (default), the longitude angle will have a range of
            ``[0, 360)``.

        """
        super().__init__(**kwargs)
        self.inputs = ('x', 'y', 'z')
        self.outputs = ('lon', 'lat')
        self.wrap_lon_at = wrap_lon_at

    @property
    def wrap_lon_at(self):
        """ An **integer number** that specifies the range of the longitude
        (azimuthal) angle.

        Allowed values are 180 and 360. When ``wrap_lon_at``
        is 180, the longitude angle will have a range of ``[-180, 180)`` and
        when ``wrap_lon_at`` is 360 (default), the longitude angle will have a
        range of ``[0, 360)``.

        """
        return self._wrap_lon_at

    @wrap_lon_at.setter
    def wrap_lon_at(self, wrap_angle):
        if not (isinstance(wrap_angle, numbers.Integral) and wrap_angle in [180, 360]):
            raise ValueError("'wrap_lon_at' must be an integer number: 180 or 360")
        self._wrap_lon_at = wrap_angle

    def evaluate(self, x, y, z):
        nquant = [isinstance(i, u.Quantity) for i in (x, y, z)].count(True)
        if nquant in [1, 2]:
            raise TypeError("All arguments must be of the same type "
                            "(i.e., quantity or not).")

        h = np.hypot(x, y)
        lat = np.rad2deg(np.arctan2(z, h))
        lon = np.rad2deg(np.arctan2(y, x))
        lon[h == 0] *= 0

        if self._wrap_lon_at != 180:
            lon = np.mod(lon, 360.0 * u.deg if nquant else 360.0, where=np.isfinite(lon), out=lon)

        return lon, lat

    def inverse(self):
        return SphericalToCartesian(wrap_lon_at=self._wrap_lon_at)
