# Licensed under a 3-clause BSD style license - see LICENSE.rst
from itertools import product, permutations
import io
import asdf

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
    'testval, unit, wrap_at, theta_def',
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
        [1, 1 * u.deg, 3600.0 * u.arcsec, np.pi / 180.0 * u.rad],
        [180, 360],
        geometry._THETA_NAMES,
    )
)
def test_spherical_to_cartesian(testval, unit, wrap_at, theta_def):
    s2c = geometry.SphericalToCartesian(wrap_phi_at=wrap_at, theta_def=theta_def)
    ounit = 1 if unit == 1 else u.dimensionless_unscaled
    phi, theta, expected = testval

    if wrap_at == 180:
        phi = np.mod(phi - 180.0, 360.0) - 180.0

    if theta_def not in geometry._LAT_LIKE:
        theta = 90 - theta

    xyz = s2c(phi * unit, theta * unit)
    if unit != 1:
        assert xyz[0].unit == u.dimensionless_unscaled
    assert u.allclose(xyz, u.Quantity(expected, ounit), atol=1e-15 * ounit)


@pytest.mark.parametrize(
    'phi, theta, unit, wrap_at, theta_def',
    list(product(
        45.0 * np.arange(8),
        [-90, -89, -55, 0, 25, 89, 90],
        [1, 1 * u.deg, 3600.0 * u.arcsec, np.pi / 180.0 * u.rad],
        [180, 360],
        ['elevation', 'polar'],
    ))
)
def test_spher2cart_roundrip(phi, theta, unit, wrap_at, theta_def):
    s2c = geometry.SphericalToCartesian(wrap_phi_at=wrap_at, theta_def=theta_def)
    c2s = geometry.CartesianToSpherical(wrap_phi_at=wrap_at, theta_def=theta_def)
    ounit = 1 if unit == 1 else u.deg

    if wrap_at == 180:
        phi = np.mod(phi - 180.0, 360.0) - 180.0

    sl = slice(int(theta == 90 or theta == -90), 2)

    if theta_def not in geometry._LAT_LIKE:
        theta = 90 - theta

    assert s2c.wrap_phi_at == wrap_at
    assert s2c.theta_def == theta_def
    assert c2s.wrap_phi_at == wrap_at
    assert c2s.theta_def == theta_def

    assert u.allclose(
        c2s(*s2c(phi * unit, theta * unit))[sl],
        (phi * ounit, theta * ounit)[sl],
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


@pytest.mark.parametrize('wrap_at', ['1', 180., True, 180j, [180], -180, 0])
def test_c2s2c_wrong_wrap_type(spher_to_cart, cart_to_spher, wrap_at):
    err_msg = "'wrap_phi_at' must be an integer number: 180 or 360"
    with pytest.raises(ValueError) as arg_err:
        geometry.SphericalToCartesian(wrap_phi_at=wrap_at)
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        spher_to_cart.wrap_phi_at = wrap_at
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        geometry.CartesianToSpherical(wrap_phi_at=wrap_at)
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        cart_to_spher.wrap_phi_at = wrap_at
    assert arg_err.value.args[0] == err_msg


@pytest.mark.parametrize(
    'theta_def',
    [180., True, 180j, 'ele', '']
)
def test_c2s2c_wrong_theta_value(spher_to_cart, cart_to_spher, theta_def):
    err_msg = ("'theta_def' must be a string with one of the following "
               "values: {:s}".format(','.join(map(repr, geometry._THETA_NAMES))))
    with pytest.raises(ValueError) as arg_err:
        geometry.SphericalToCartesian(theta_def=theta_def)
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        spher_to_cart.theta_def = theta_def
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        geometry.CartesianToSpherical(theta_def=theta_def)
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        cart_to_spher.theta_def = theta_def
    assert arg_err.value.args[0] == err_msg


def test_cartesian_spherical_asdf(spher_to_cart, cart_to_spher):
    # create file object
    f = asdf.AsdfFile({'c2s': cart_to_spher, 's2c': spher_to_cart})

    # write to...
    buf = io.BytesIO()
    f.write_to(buf)

    # read back:
    buf.seek(0)
    f = asdf.open(buf)

    # retreave transformations:
    c2s = f['c2s']
    s2c = f['s2c']

    coords = [
        (45.0, -90.0), (45.0, -45.0), (45, 0.0),
        (45.0, 45), (45.0, 90.0), (135.0, -90.0),
        (135.0, -45.0), (135.0, 0.0), (135.0, 45.0),
        (135.0, 90.0), (225.0, -90.0), (225.0, -45.0),
        (225.0, 0.0), (225.0, 45.0), (225.0, 90.0),
        (315.0, -90.0), (315.0, -45.0), (315.0, 0.0),
        (315.0, 45.0), (315.0, 90.0)
    ]

    for phi, th in coords:
        xyz = s2c(phi, th)
        assert xyz == spher_to_cart(phi, th)
        phi2, th2 = c2s(*xyz)
        assert phi2, th2 == cart_to_spher(*xyz)
        assert np.allclose((phi, th), (phi2, th2))
