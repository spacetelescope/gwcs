# Licensed under a 3-clause BSD style license - see LICENSE.rst
from itertools import product, permutations
import io

import pytest

import asdf
import numpy as np
from astropy import units as u
from asdf_astropy.converters.transform.tests.test_transform import (
     assert_model_roundtrip)

from .. import geometry


_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def test_spherical_cartesian_inverse():
    t = geometry.SphericalToCartesian()
    assert type(t.inverse) == geometry.CartesianToSpherical

    t = geometry.CartesianToSpherical()
    assert type(t.inverse) == geometry.SphericalToCartesian


@pytest.mark.parametrize(
    'testval, unit, wrap_at',
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
    )
)
def test_spherical_to_cartesian(testval, unit, wrap_at):
    s2c = geometry.SphericalToCartesian(wrap_lon_at=wrap_at)
    ounit = 1 if unit == 1 else u.dimensionless_unscaled
    lon, lat, expected = testval

    if wrap_at == 180:
        lon = np.mod(lon - 180.0, 360.0) - 180.0

    xyz = s2c(lon * unit, lat * unit)
    if unit != 1:
        assert xyz[0].unit == u.dimensionless_unscaled
    assert u.allclose(xyz, u.Quantity(expected, ounit), atol=1e-15 * ounit)


@pytest.mark.parametrize(
    'lon, lat, unit, wrap_at',
    list(product(
        [0, 45, 90, 135, 180, 225, 270, 315, 360],
        [-90, -89, -55, 0, 25, 89, 90],
        [1, 1 * u.deg, 3600.0 * u.arcsec, np.pi / 180.0 * u.rad],
        [180, 360],
    ))
)
def test_spher2cart_roundrip(lon, lat, unit, wrap_at):
    s2c = geometry.SphericalToCartesian(wrap_lon_at=wrap_at)
    c2s = geometry.CartesianToSpherical(wrap_lon_at=wrap_at)
    ounit = 1 if unit == 1 else u.deg

    if wrap_at == 180:
        lon = np.mod(lon - 180.0, 360.0) - 180.0

    assert s2c.wrap_lon_at == wrap_at
    assert c2s.wrap_lon_at == wrap_at

    assert u.allclose(
        c2s(*s2c(lon * unit, lat * unit)),
        (lon * ounit, lat * ounit),
        atol=1e-15 * ounit
    )


def test_cart2spher_at_pole(cart_to_spher):
    assert np.allclose(cart_to_spher(0, 0, 1), (0, 90), rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    'lonlat, unit, wrap_at',
    list(product(
        [
            [[1], [-80]],
            [[325], [-89]],
            [[0, 1, 120, 180, 225, 325, 359], [-89, 0, 89, 10, -15, 45, -30]],
            [np.array([0.0, 1, 120, 180, 225, 325, 359]), np.array([-89, 0.0, 89, 10, -15, 45, -30])]
        ],
        [None, 1 * u.deg],
        [180, 360],
    ))
)
def test_spher2cart_roundrip_arr(lonlat, unit, wrap_at):
    lon, lat = lonlat
    s2c = geometry.SphericalToCartesian(wrap_lon_at=wrap_at)
    c2s = geometry.CartesianToSpherical(wrap_lon_at=wrap_at)

    if wrap_at == 180:
        if isinstance(lon, np.ndarray):
            lon = np.mod(lon - 180.0, 360.0) - 180.0
        else:
            lon = [((l - 180.0) % 360.0) - 180.0 for l in lon]

    atol = 1e-15
    if unit is None:
        olon = lon
        olat = lat
    else:
        olon = lon * u.deg
        olat = lat * u.deg
        lon = lon * unit
        lat = lat * unit
        atol = atol * u.deg

    assert u.allclose(
        c2s(*s2c(lon, lat)),
        (olon, olat),
        atol=atol
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
    err_msg = "'wrap_lon_at' must be an integer number: 180 or 360"
    with pytest.raises(ValueError) as arg_err:
        geometry.SphericalToCartesian(wrap_lon_at=wrap_at)
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        spher_to_cart.wrap_lon_at = wrap_at
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        geometry.CartesianToSpherical(wrap_lon_at=wrap_at)
    assert arg_err.value.args[0] == err_msg

    with pytest.raises(ValueError) as arg_err:
        cart_to_spher.wrap_lon_at = wrap_at
    assert arg_err.value.args[0] == err_msg


def test_cartesian_spherical_asdf(tmpdir):
    s2c0 = geometry.SphericalToCartesian(wrap_lon_at=360)
    c2s0 = geometry.CartesianToSpherical(wrap_lon_at=180)

    # asdf round-trip test:
    assert_model_roundtrip(c2s0, tmpdir)
    assert_model_roundtrip(s2c0, tmpdir)

    # create file object
    f = asdf.AsdfFile({'c2s': c2s0, 's2c': s2c0})

    # write to...
    buf = io.BytesIO()
    f.write_to(buf)

    # read back:
    buf.seek(0)
    f = asdf.open(buf)

    # retrieve transformations:
    c2s = f['c2s']
    s2c = f['s2c']

    pcoords = [
        (45.0, -90.0), (45.0, -45.0), (45, 0.0),
        (45.0, 45), (45.0, 90.0), (135.0, -90.0),
        (135.0, -45.0), (135.0, 0.0), (135.0, 45.0),
        (135.0, 90.0)
    ]

    ncoords = [
        (225.0, -90.0), (225.0, -45.0),
        (225.0, 0.0), (225.0, 45.0), (225.0, 90.0),
        (315.0, -90.0), (315.0, -45.0), (315.0, 0.0),
        (315.0, 45.0), (315.0, 90.0)
    ]

    for lon, lat in pcoords:
        xyz = s2c(lon, lat)
        assert xyz == s2c0(lon, lat)
        lon2, lat2 = c2s(*xyz)
        assert lon2, lat2 == c2s0(*xyz)
        assert np.allclose((lon, lat), (lon2, lat2))

    for lon, lat in ncoords:
        xyz = s2c(lon, lat)
        assert xyz == s2c0(lon, lat)
        lon2, lat2 = c2s(*xyz)
        lon3, lat3 = s2c.inverse(*xyz)
        assert lon2, lat2 == c2s0(*xyz)
        assert np.allclose((lon, lat), (lon2 + 360, lat2))
        assert np.allclose((lon, lat), (lon3, lat2))
