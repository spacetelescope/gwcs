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

# allowable values for angle theta definition in spherical<->Cartesian
# transformations:
_THETA_NAMES = ['latitude', 'colatitude', 'altitude', 'elevation',
                'inclination', 'polar']

# theta definitions for which angle is measured from the "reference" plane:
_LAT_LIKE = ['latitude', 'altitude', 'elevation']


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
    to be in degrees with ``phi`` being the *azimuthal angle* (or *longitude*)
    ``[0, 360)`` (or ``[-180, 180)``) and ``theta`` being the *elevation angle*
    (or *latitude*) ``[-90, 90]`` or the *inclination (polar, colatitude) angle*
    in range ``[0, 180]`` depending on the used definition.

    """
    _separable = False

    _input_units_strict = True
    _input_units_allow_dimensionless = True

    n_inputs = 2
    n_outputs = 3

    def __init__(self, wrap_phi_at=360, theta_def='latitude', **kwargs):
        """
        Parameters
        ----------
        wrap_phi_at : {360, 180}, optional
            An **integer number** that specifies the range of the azimuthal
            (longitude) angle. When ``wrap_phi_at`` is 180, azimuthal angle
            will have a range of ``[-180, 180)`` and when ``wrap_phi_at``
            is 360 (default), the azimuthal angle will have a range of
            ``[0, 360)``.

        theta_def : {'latitude', 'colatitude', 'altitude', 'elevation', \
                     'inclination', 'polar'}, optional
            Specifies the definition used for the angle ``theta``:
            either 'elevation' angle (synonyms: 'latitude', 'altitude') from
            the reference plane or 'inclination' angle (synonyms: 'polar',
            'colatitude') measured from the pole (zenith direction).

        """
        super().__init__(**kwargs)
        self.inputs = ('phi', 'theta')
        self.outputs = ('x', 'y', 'z')
        self.wrap_phi_at = wrap_phi_at
        self.theta_def = theta_def

    @property
    def wrap_phi_at(self):
        """ An **integer number** that specifies the range of the azimuthal
        (longitude) angle.

        Allowed values are 180 and 360. When ``wrap_phi_at``
        is 180, azimuthal angle will have a range of ``[-180, 180)`` and when
        ``wrap_phi_at`` is 360 (default), the azimuthal angle will have a
        range of ``[0, 360)``.

        """
        return self._wrap_phi_at

    @wrap_phi_at.setter
    def wrap_phi_at(self, wrap_angle):
        if not (isinstance(wrap_angle, numbers.Integral) and wrap_angle in [180, 360]):
            raise ValueError("'wrap_phi_at' must be an integer number: 180 or 360")
        self._wrap_phi_at = wrap_angle

    @property
    def theta_def(self):
        """ Definition used for the ``theta`` angle, i.e., latitude or colatitude.

        When ``theta_def`` is either 'elevation', or 'latitude', or 'altitude',
        angle ``theta_def`` is measured from the reference plane.
        When ``theta_def`` is either 'inclination', or 'polar', or 'colatitude',
        angle ``theta_def`` is measured from the pole (zenith direction).

        """
        return self._theta_def

    @theta_def.setter
    def theta_def(self, theta_def):
        if theta_def not in _THETA_NAMES:
            raise ValueError(
                "'theta_def' must be a string with one of the following "
                "values: {:s}".format(','.join(map(repr, _THETA_NAMES)))
            )
        self._theta_def = theta_def
        self._is_theta_latitude = theta_def in _LAT_LIKE

    def evaluate(self, phi, theta):
        if isinstance(phi, u.Quantity) != isinstance(theta, u.Quantity):
            raise TypeError("All arguments must be of the same type "
                            "(i.e., quantity or not).")

        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

        if self._is_theta_latitude:
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
    Output spherical coordinates are in degrees. When input cartesian
    coordinates are quantities (``Quantity`` objects), output angles
    will also be quantities in degrees. Angle ``phi`` is the *azimuthal angle*
    (or *longitude*) ``[0, 360)`` (or ``[-180, 180)``) and angle ``theta``
    is the *elevation angle* (or *latitude*) ``[-90, 90]`` or the *inclination
    (polar, colatitude) angle* in range ``[0, 180]`` depending on the used
    definition (see documentation for ``theta_def``).

    """
    _separable = False

    n_inputs = 3
    n_outputs = 2

    def __init__(self, wrap_phi_at=360, theta_def='latitude', **kwargs):
        """
        Parameters
        ----------
        wrap_phi_at : {360, 180}, optional
            An **integer number** that specifies the range of the azimuthal
            (longitude) angle. When ``wrap_phi_at`` is 180, azimuthal angle
            will have a range of ``[-180, 180)`` and when ``wrap_phi_at``
            is 360 (default), the azimuthal angle will have a range of
            ``[0, 360)``.

        theta_def : {'latitude', 'colatitude', 'altitude', 'elevation', \
                     'inclination', 'polar'}, optional
            Specifies the definition used for the angle ``theta``:
            either 'elevation' angle (synonyms: 'latitude', 'altitude') from
            the reference plane or 'inclination' angle (synonyms: 'polar',
            'colatitude') measured from the pole (zenith direction).

        """
        super().__init__(**kwargs)
        self.inputs = ('x', 'y', 'z')
        self.outputs = ('phi', 'theta')
        self.wrap_phi_at = wrap_phi_at
        self.theta_def = theta_def

    @property
    def wrap_phi_at(self):
        """ An **integer number** that specifies the range of the azimuthal
        (longitude) angle.

        Allowed values are 180 and 360. When ``wrap_phi_at``
        is 180, azimuthal angle will have a range of ``[-180, 180)`` and when
        ``wrap_phi_at`` is 360 (default), the azimuthal angle will have a
        range of ``[0, 360)``.

        """
        return self._wrap_phi_at

    @wrap_phi_at.setter
    def wrap_phi_at(self, wrap_angle):
        if not (isinstance(wrap_angle, numbers.Integral) and wrap_angle in [180, 360]):
            raise ValueError("'wrap_phi_at' must be an integer number: 180 or 360")
        self._wrap_phi_at = wrap_angle

    @property
    def theta_def(self):
        """ Definition used for the ``theta`` angle, i.e., latitude or colatitude.

        When ``theta_def`` is either 'elevation', or 'latitude', or 'altitude',
        angle ``theta_def`` is measured from the reference plane.
        When ``theta_def`` is either 'inclination', or 'polar', or 'colatitude',
        angle ``theta_def`` is measured from the pole (zenith direction).

        """
        return self._theta_def

    @theta_def.setter
    def theta_def(self, theta_def):
        if theta_def not in _THETA_NAMES:
            raise ValueError(
                "'theta_def' must be a string with one of the following "
                "values: {:s}".format(','.join(map(repr, _THETA_NAMES)))
            )
        self._theta_def = theta_def
        self._is_theta_latitude = theta_def in _LAT_LIKE

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
        if not self._is_theta_latitude:
            theta = (90.0 * u.deg if nquant else 90.0) - theta

        return phi, theta

    def inverse(self):
        return SphericalToCartesian(
            wrap_phi_at=self._wrap_phi_at,
            theta_def=self._theta_def
        )
