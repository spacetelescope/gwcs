# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectroscopy related models.
"""

import numpy as np
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter
import astropy.units as u


__all__ = ['ToDirectionCosines', 'FromDirectionCosines',
           'WavelengthFromGratingEquation', 'AnglesFromGratingEquation3D']


__doctest_skip__ = ['AnglesFromGratingEquation3D', 'WavelengthFromGratingEquation']


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


class WavelengthFromGratingEquation(Model):
    r""" Solve the Grating Dispersion Law for the wavelength.

    .. Note:: This form of the equation can be used for paraxial
    (small angle approximation) as well as oblique incident angles.
    With paraxial systems the inputs are sin of the angles and it
    transforms to :math:`\sin(alpha_in) + \sin(alpha_out) / m * d` .
    With oblique angles the inputs are the direction cosines of the
    angles.

    Parameters
    ----------
    groove_density : int
        Grating ruling density in units of 1/length.
    spectral_order : int
        Spectral order.

    Examples
    --------
    >>> from astropy.modeling.models import math
    >>> model = WavelengthFromGratingEquation(groove_density=20000*1/u.m, spectral_order=-1)
    >>> alpha_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> alpha_out = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> lam = model(alpha_in, alpha_out)
    >>> print(lam)
    -1.7453292519934437e-10 m

    """

    _separable = False

    linear = False

    n_inputs = 2
    n_outputs = 1

    groove_density = Parameter(default=1)
    """ Grating ruling density in units of 1/m."""
    spectral_order = Parameter(default=1)
    """ Spectral order."""

    def __init__(self, groove_density, spectral_order, **kwargs):
        super().__init__(groove_density=groove_density,
                         spectral_order=spectral_order, **kwargs)
        self.inputs = ("alpha_in", "alpha_out")
        """ Sine function of the angles or the direction cosines."""
        self.outputs = ("wavelength",)
        """ Wavelength."""

    def evaluate(self, alpha_in, alpha_out, groove_density, spectral_order):
        return (alpha_in + alpha_out) / (groove_density * spectral_order)

    @property
    def return_units(self):
        if self.groove_density.unit is None:
            return None
        else:
            return {'wavelength': u.Unit(1 / self.groove_density.unit)}


class AnglesFromGratingEquation3D(Model):
    """
    Solve the 3D Grating Dispersion Law in Direction Cosine
    space for the refracted angle.

    Parameters
    ----------
    groove_density : int
        Grating ruling density in units of 1/m.
    order : int
        Spectral order.

    Examples
    --------
    >>> from astropy.modeling.models import math
    >>> model = AnglesFromGratingEquation3D(groove_density=20000*1/u.m, spectral_order=-1)
    >>> alpha_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> beta_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> lam = 2e-6 * u.m
    >>> alpha_out, beta_out, gamma_out = model(lam, alpha_in, beta_in)
    >>> print(alpha_out, beta_out, gamma_out)
    0.04000174532925199 -1.7453292519934436e-06 0.9991996098716049

    """

    _separable = False

    linear = False

    n_inputs = 3
    n_outputs = 3

    groove_density = Parameter(default=1)
    """ Grating ruling density in units 1/ length."""

    spectral_order = Parameter(default=1)
    """ Spectral order."""

    def __init__(self, groove_density, spectral_order, **kwargs):
        super().__init__(groove_density=groove_density,
                         spectral_order=spectral_order, **kwargs)
        self.inputs = ("wavelength", "alpha_in", "beta_in")
        """ Wavelength and 2 angle coordinates going into the grating."""

        self.outputs = ("alpha_out", "beta_out", "gamma_out")
        """ Two angles coming out of the grating. """

    def evaluate(self, wavelength, alpha_in, beta_in,
                 groove_density, spectral_order):
        if alpha_in.shape != beta_in.shape:
            raise ValueError("Expected input arrays to have the same shape.")

        if isinstance(groove_density, u.Quantity):
            alpha_in = u.Quantity(alpha_in)
            beta_in = u.Quantity(beta_in)

        alpha_out = -groove_density * spectral_order * wavelength + alpha_in
        beta_out = - beta_in
        gamma_out = np.sqrt(1 - alpha_out ** 2 - beta_out ** 2)
        return alpha_out, beta_out, gamma_out

    @property
    def input_units(self):
        if self.groove_density.unit is None:
            return None
        else:
            return {'wavelength': 1 / self.groove_density.unit,
                    'alpha_in': u.Unit(1),
                    'beta_in': u.Unit(1)}
