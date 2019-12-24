import pytest
import astropy.units as u
from astropy.modeling.models import Identity
import numpy as np
from numpy.testing import assert_allclose
from .. import spectroscopy as sp# noqa
from .. import geometry# noqa


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


@pytest.mark.parametrize(('wavelength', 'n'),
                         [(1, 1.43079543),
                          (2, 1.42575377),
                          (5, 1.40061966)
                          ])
def test_SellmeierGlass(wavelength, n, sellmeier_glass):
    """ Test from Nirspec team.

    Wavelength is in microns.
    """
    n_result = sellmeier_glass(wavelength)
    assert_allclose(n_result, n)


def test_SellmeierZemax(sellmeier_zemax):
    """ The data for this test come from Nirspec."""
    n = 1.4254647475849418
    assert_allclose(sellmeier_zemax(2), n)


def test_Snell3D(sellmeier_glass):
    """ Test from Nirspec."""
    expected = (0.07015255913513296, 0.07015255913513296, 0.9950664484814988)
    model = sp.Snell3D()
    n = 1.4254647475849418
    assert_allclose(model(n, .1, .1, .9), expected)


def test_snell_sellmeier_combined(sellmeier_glass):
    fromdircos = geometry.FromDirectionCosines()
    todircos = geometry.ToDirectionCosines()
    model = sellmeier_glass & todircos | sp.Snell3D() & Identity(1) | fromdircos

    expected = (0.07013833805527926, 0.07013833805527926, 1.0050677723764139)
    assert_allclose(model(2, .1, .1, .9), expected)
