import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
from .. import spectroscopy as sp# noqa


def test_angles_grating_equation():
    """ Test agaibst the Nispec implementation."""
    lam = np.array([2e-6] * 4)
    alpha_in = np.linspace(.01, .05, 4)

    model = sp.AnglesFromGratingEquation3D(20000, -1)

    # Eq. from Nirspec model.
    xout = -alpha_in - (model.groove_density * model.spectral_order * lam)

    alpha_out, beta_out, gamma_out = model(lam, -alpha_in, alpha_in)
    assert_allclose(alpha_out, xout)
    assert_allclose(beta_out, -alpha_in)
    assert_allclose(gamma_out, np.sqrt(1 -  alpha_out**2 - beta_out**2))

    # Now with units
    model = sp.AnglesFromGratingEquation3D(20000 * 1/u.m, -1)

    # Eq. from Nirspec model.
    xout = -alpha_in - (model.groove_density * model.spectral_order * lam * u.m)

    alpha_out, beta_out, gamma_out = model(lam * u.m, -u.Quantity(alpha_in), u.Quantity(alpha_in))
    assert_allclose(alpha_out, xout)
    assert_allclose(beta_out, -alpha_in)
    assert_allclose(gamma_out, np.sqrt(1 -  alpha_out**2 - beta_out**2))


def test_wavelength_grating_equation_units():
    alpha_in = np.linspace(.01, .05, 4)

    model = sp.WavelengthFromGratingEquation(20000, -1)
    # Eq. from Nirspec model.
    wave = -(alpha_in + alpha_in) / (20000 * -1)
    result = model(-alpha_in, -alpha_in)
    assert_allclose(result, wave)

    # Now with units
    model = sp.WavelengthFromGratingEquation(20000 * 1/u.m, -1)
    # Eq. from Nirspec model.
    wave = -(u.Quantity(alpha_in) + u.Quantity(alpha_in)) / (20000 * 1/u.m * -1)

    result = model(-u.Quantity(alpha_in), -u.Quantity(alpha_in))
    assert_allclose(result, wave)
