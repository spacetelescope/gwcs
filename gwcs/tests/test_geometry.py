# Licensed under a 3-clause BSD style license - see LICENSE.rst
from itertools import product, permutations

import numpy as np
import pytest
from astropy import units as u

from .. import geometry


_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def test_spherical_cartesian_inverse():
    t = geometry.SphericalToCartesian()
    assert type(t.inverse) == geometry.CartesianToSpherical

    t = geometry.CartesianToSpherical()
    assert type(t.inverse) == geometry.SphericalToCartesian


@pytest.mark.parametrize(
    'testval,unit',
    product(
        [
            (45.0, -90.0, (0.0, 0.0, -1.0)),
            (45.0, -45.0, (0.5, 0.5, -_INV_SQRT2)),
            (45, 0.0, (_INV_SQRT2, _INV_SQRT2, 0.0)),
            (45.0, 45, (0.5, 0.5, _INV_SQRT2)),
            (45.0, 90.0, (0.0, 0.0, 1.0)),
            (135.0, -90.0, (0.0, 0.0, -1.0)),
            (135.0, -45.0, (-0.5, 0.5, -_INV_SQRT2)),
            (135.0, 0.0, (-_INV_SQRT2, _INV_SQRT2, 0.0)),
            (135.0, 45.0, (-0.5, 0.5, _INV_SQRT2)),
            (135.0, 90.0, (0.0, 0.0, 1.0)),
            (225.0, -90.0, (0.0, 0.0, -1.0)),
            (225.0, -45.0, (-0.5, -0.5, -_INV_SQRT2)),
            (225.0, 0.0, (-_INV_SQRT2, -_INV_SQRT2, 0.0)),
            (225.0, 45.0, (-0.5, -0.5, _INV_SQRT2)),
            (225.0, 90.0, (0.0, 0.0, 1.0)),
            (315.0, -90.0, (0.0, 0.0, -1.0)),
            (315.0, -45.0, (0.5, -0.5, -_INV_SQRT2)),
            (315.0, 0.0, (_INV_SQRT2, -_INV_SQRT2, 0.0)),
            (315.0, 45.0, (0.5, -0.5, _INV_SQRT2)),
            (315.0, 90.0, (0.0, 0.0, 1.0)),
        ],
        [1, 1 * u.deg, 3600.0 * u.arcsec, np.pi / 180.0 * u.rad]
    )
)
def test_spherical_to_cartesian(spher_to_cart, testval, unit):
    ounit = 1 if unit == 1 else u.dimensionless_unscaled
    phi, theta, expected = testval
    xyz = spher_to_cart(phi * unit, theta * unit)
    if unit != 1:
        assert xyz[0].unit == u.dimensionless_unscaled
    assert u.allclose(xyz, u.Quantity(expected, ounit), atol=1e-15 * ounit)


@pytest.mark.parametrize(
    'phi,theta,unit',
    list(product(
        45.0 * np.arange(8),
        [-90, -55, 0, 25, 90],
        [1, 1 * u.deg, 3600.0 * u.arcsec, np.pi / 180.0 * u.rad]
    ))
)
def test_spher2cart_roundrip(spher_to_cart, cart_to_spher, phi, theta, unit):
    ounit = 1 if unit == 1 else u.deg
    assert u.allclose(
        cart_to_spher(*spher_to_cart(phi * unit, theta * unit)),
        (phi * ounit, theta * ounit),
        atol=1e-15 * ounit
    )


@pytest.mark.parametrize('unit1, unit2', [(u.deg, 1), (1, u.deg)])
def test_spherical_to_cartesian_mixed_Q(spher_to_cart, unit1, unit2):
    with pytest.raises(TypeError) as arg_err:
        spher_to_cart(135.0 * unit1, 45.0 * unit2)
    assert (arg_err.value.args[0] == "All arguments must be of the same type "
            "(i.e., quantity or not).")


@pytest.mark.parametrize(
    'x, y, z',
    list(set(
        tuple(permutations([1 * u.m, 1, 1])) + tuple(permutations([1 * u.m, 1 * u.m, 1]))
    ))
)
def test_cartesian_to_spherical_mixed_Q(cart_to_spher, x, y, z):
    with pytest.raises(TypeError) as arg_err:
        cart_to_spher(x, y, z)
    assert (arg_err.value.args[0] == "All arguments must be of the same type "
            "(i.e., quantity or not).")
