# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests the API defined in astropy APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy import coordinates as coord


# Shorthand the name of the 2d gwcs fixture
@pytest.fixture
def wcsobj(request):
    return request.getfixturevalue(request.param)


wcs_objs = pytest.mark.parametrize("wcsobj", ['gwcs_2d_spatial_shift'], indirect=True)


@pytest.fixture
def wcs_ndim_types_units(request):
    """
    Generate a wcs and the expected ndim, types, and units.
    """
    ndim = {'gwcs_2d_spatial_shift': (2, 2),
            'gwcs_1d_freq': (2, 1),
            'gwcs_3d_spatial_wave': (3, 3)}
    types = {'gwcs_2d_spatial_shift': ("pos.eq.ra", "pos.eq.dec"),
             'gwcs_1d_freq': ("em.freq",),
             'gwcs_3d_spatial_wave': ("pos.eq.ra", "pos.eq.dec", "em.wl")}
    units = {'gwcs_2d_spatial_shift': ("deg", "deg"),
             'gwcs_1d_freq': ("Hz",),
             'gwcs_3d_spatial_wave': ("deg", "deg", "m")}

    return (request.getfixturevalue(request.param),
            ndim[request.param],
            types[request.param],
            units[request.param])


# # x, y inputs - scalar and array
x, y = 1, 2
xarr, yarr = np.ones((3, 4)), np.ones((3, 4)) + 1

fixture_names = ['gwcs_2d_spatial_shift', 'gwcs_1d_freq', 'gwcs_3d_spatial_wave']
fixture_wcs_ndim_types_units = pytest.mark.parametrize("wcs_ndim_types_units", fixture_names, indirect=True)
all_wcses_names = fixture_names + ['gwcs_3d_spatial_wave_units', 'gwcs_4d_spatial_wave_time_units']
fixture_all_wcses = pytest.mark.parametrize("wcsobj", all_wcses_names, indirect=True)


@fixture_wcs_ndim_types_units
def test_pixel_n_dim(wcs_ndim_types_units):
    wcsobj, ndims, *_ = wcs_ndim_types_units
    assert wcsobj.pixel_n_dim == ndims[0]


@fixture_wcs_ndim_types_units
def test_world_n_dim(wcs_ndim_types_units):
    wcsobj, ndims, *_ = wcs_ndim_types_units
    assert wcsobj.world_n_dim == ndims[1]


@fixture_wcs_ndim_types_units
def test_world_axis_physical_types(wcs_ndim_types_units):
    wcsobj, ndims, physical_types, world_units = wcs_ndim_types_units
    assert wcsobj.world_axis_physical_types == physical_types


@fixture_wcs_ndim_types_units
def test_world_axis_units(wcs_ndim_types_units):
    wcsobj, ndims, physical_types, world_units = wcs_ndim_types_units
    assert wcsobj.world_axis_units == world_units


@pytest.mark.parametrize(("x", "y"), zip((x, xarr), (y, yarr)))
def test_pixel_to_world_values(gwcs_2d_spatial_shift, x, y):
    wcsobj = gwcs_2d_spatial_shift
    assert_allclose(wcsobj.pixel_to_world_values(x, y), wcsobj(x, y, with_units=False))


@pytest.mark.parametrize(("x", "y"), zip((x, xarr), (y, yarr)))
def test_array_index_to_world_values(gwcs_2d_spatial_shift, x, y):
    wcsobj = gwcs_2d_spatial_shift
    assert_allclose(wcsobj.array_index_to_world_values(x, y), wcsobj(y, x, with_units=False))


@wcs_objs
def test_serialized_classes(wcsobj):
    assert not wcsobj.serialized_classes


@wcs_objs
def test_low_level_wcs(wcsobj):
    assert id(wcsobj.low_level_wcs) == id(wcsobj)


@wcs_objs
def test_pixel_to_world(wcsobj):
    comp = wcsobj(x, y, with_units=True)
    comp = wcsobj.output_frame.coordinates(comp)
    result = wcsobj.pixel_to_world(x, y)
    assert isinstance(comp, coord.SkyCoord)
    assert isinstance(result, coord.SkyCoord)
    assert_allclose(comp.data.lon, result.data.lon)
    assert_allclose(comp.data.lat, result.data.lat)


@wcs_objs
def test_array_index_to_world(wcsobj):
    comp = wcsobj(x, y, with_units=True)
    comp = wcsobj.output_frame.coordinates(comp)
    result = wcsobj.array_index_to_world(y, x)
    assert isinstance(comp, coord.SkyCoord)
    assert isinstance(result, coord.SkyCoord)
    assert_allclose(comp.data.lon, result.data.lon)
    assert_allclose(comp.data.lat, result.data.lat)


def test_pixel_to_world_quantity(gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity):
    result1 = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result2 = gwcs_2d_shift_scale_quantity.pixel_to_world(x, y)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)

    # test with Quantity pixel inputs
    result1 = gwcs_2d_shift_scale.pixel_to_world(x * u.pix, y * u.pix)
    result2 = gwcs_2d_shift_scale_quantity.pixel_to_world(x * u.pix, y * u.pix)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)

    # test for pixel units
    with pytest.raises(ValueError):
        gwcs_2d_shift_scale.pixel_to_world(x * u.Jy, y * u.Jy)


def test_array_index_to_world_quantity(gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity):
    result0 = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result1 = gwcs_2d_shift_scale.array_index_to_world(y, x)
    result2 = gwcs_2d_shift_scale_quantity.array_index_to_world(y, x)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)
    assert_allclose(result0.data.lon, result1.data.lon)
    assert_allclose(result0.data.lat, result1.data.lat)

    # test with Quantity pixel inputs
    result0 = gwcs_2d_shift_scale.pixel_to_world(x * u.pix, y * u.pix)
    result1 = gwcs_2d_shift_scale.array_index_to_world(y * u.pix, x * u.pix)
    result2 = gwcs_2d_shift_scale_quantity.array_index_to_world(y * u.pix, x * u.pix)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)
    assert_allclose(result0.data.lon, result1.data.lon)
    assert_allclose(result0.data.lat, result1.data.lat)

    # test for pixel units
    with pytest.raises(ValueError):
        gwcs_2d_shift_scale.array_index_to_world(x * u.Jy, y * u.Jy)


def test_world_to_pixel_quantity(gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity):
    skycoord = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result1 = gwcs_2d_shift_scale.world_to_pixel(skycoord)
    result2 = gwcs_2d_shift_scale_quantity.world_to_pixel(skycoord)
    assert_allclose(result1, (x, y))
    assert_allclose(result2, (x, y))


def test_world_to_array_index_quantity(gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity):
    skycoord = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result0 = gwcs_2d_shift_scale.world_to_pixel(skycoord)
    result1 = gwcs_2d_shift_scale.world_to_array_index(skycoord)
    result2 = gwcs_2d_shift_scale_quantity.world_to_array_index(skycoord)
    assert_allclose(result0, (x, y))
    assert_allclose(result1, (y, x))
    assert_allclose(result2, (y, x))


@pytest.fixture(params=[0, 1])
def sky_ra_dec(request, gwcs_2d_spatial_shift):
    ref_frame = gwcs_2d_spatial_shift.output_frame.reference_frame
    ra, dec = 2, 4
    if request.param == 0:
        sky = coord.SkyCoord(ra * u.deg, dec * u.deg, frame=ref_frame)
    else:
        ra = np.ones((3, 4)) * ra
        dec = np.ones((3, 4)) * dec
        sky = coord.SkyCoord(ra * u.deg, dec * u.deg, frame=ref_frame)
    return sky, ra, dec


def test_world_to_pixel(gwcs_2d_spatial_shift, sky_ra_dec):
    wcsobj = gwcs_2d_spatial_shift
    sky, ra, dec = sky_ra_dec
    assert_allclose(wcsobj.world_to_pixel(sky), wcsobj.invert(ra, dec, with_units=False))


def test_world_to_array_index(gwcs_2d_spatial_shift, sky_ra_dec):
    wcsobj = gwcs_2d_spatial_shift
    sky, ra, dec = sky_ra_dec
    assert_allclose(wcsobj.world_to_array_index(sky), wcsobj.invert(ra, dec, with_units=False)[::-1])


def test_world_to_pixel_values(gwcs_2d_spatial_shift, sky_ra_dec):
    wcsobj = gwcs_2d_spatial_shift
    sky, ra, dec = sky_ra_dec

    assert_allclose(wcsobj.world_to_pixel_values(sky), wcsobj.invert(ra, dec, with_units=False))


def test_world_to_array_index_values(gwcs_2d_spatial_shift, sky_ra_dec):
    wcsobj = gwcs_2d_spatial_shift
    sky, ra, dec = sky_ra_dec

    assert_allclose(wcsobj.world_to_array_index_values(sky),
                    wcsobj.invert(ra, dec, with_units=False)[::-1])
