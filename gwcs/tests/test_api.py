# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests the API defined in astropy APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from astropy import coordinates as coord
from astropy.modeling import models
from astropy import units as u

from .. import wcs
from .. import coordinate_frames as cf

import pytest

# frames
sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), axes_order=(0, 1))
detector = cf.Frame2D(name='detector', axes_order=(0, 1))
detector1d = cf.CoordinateFrame(naxes=1, axes_type=('SPECTRAL',), axes_order=(0,))
spec_only = cf.SpectralFrame(unit=u.AA, axes_order=(0,))

spec1 = cf.SpectralFrame(name='freq', unit=[u.Hz, ], axes_order=(2, ))
spec2 = cf.SpectralFrame(name='wave', unit=[u.m, ], axes_order=(2, ), axes_names=('lambda', ))

comp1 = cf.CompositeFrame([sky_frame, spec1])

# transforms
m1 = models.Shift(1) & models.Shift(2)
m2 = models.Scale(2)
m = m1 & m2

pipe = [(detector, m1),
        (sky_frame, None)
        ]

example_wcs = wcs.WCS(pipe)

# transform without Quantity
m1 = models.Shift(1) & models.Shift(2)
m2 = models.Scale(5) & models.Scale(10)
m3 = m1 | m2
pipe = [(detector, m3), (sky_frame, None)]
wcs_noquantity = wcs.WCS(pipe)

# equivalent transform with Quantity
m4 = models.Shift(1 * u.pix) & models.Shift(2 * u.pix)
m5 = models.Scale(5 * u.deg)
m6 = models.Scale(10 * u.deg)
m5.input_units_equivalencies = {'x': u.pixel_scale(1*u.deg/u.pix)}
m6.input_units_equivalencies = {'x': u.pixel_scale(1*u.deg/u.pix)}
m5.inverse = models.Scale(1./5 * u.pix)
m6.inverse = models.Scale(1./10 * u.pix)
m5.inverse.input_units_equivalencies = {'x': u.pixel_scale(1*u.pix/u.deg)}
m6.inverse.input_units_equivalencies = {'x': u.pixel_scale(1*u.pix/u.deg)}
m7 = m5 & m6
m8 = m4 | m7
pipe2 = [(detector, m8), (sky_frame, None)]
wcs_quantity = wcs.WCS(pipe2)


def create_example_wcs():
    example_wcs = [wcs.WCS([(detector, m1),
                            (sky_frame, None)]),
                   wcs.WCS([(detector, m2),
                            (spec1, None)]),
                   wcs.WCS([(detector, m),
                            (comp1, None)])
                   ]

    pixel_world_ndim = [(2, 2), (2, 1), (2, 3)]
    physical_types = [("pos.eq.ra", "pos.eq.dec"), ("em.freq",), ("pos.eq.ra", "pos.eq.dec", "em.freq")]
    world_units = [("deg", "deg"), ("Hz",), ("deg", "deg", "Hz")]

    return example_wcs, pixel_world_ndim, physical_types, world_units


# x, y inputs - scalar and array
x, y = 1, 2
xarr, yarr = np.ones((3, 4)), np.ones((3, 4)) + 1

# ra, dec inputs - scalar, arrays and SkyCoord objects
ra, dec = 2, 4
sky = coord.SkyCoord(ra * u.deg, dec * u.deg, frame=sky_frame.reference_frame)
raarr = np.ones((3, 4)) * ra
decarr = np.ones((3, 4)) * dec
skyarr = coord.SkyCoord(raarr * u.deg, decarr * u.deg,
                        frame=sky_frame.reference_frame)

ex_wcs, dims, physical_types, world_units = create_example_wcs()


@pytest.mark.parametrize(("wcsobj", "ndims"), zip(ex_wcs, dims))
def test_pixel_n_dim(wcsobj, ndims):
    assert wcsobj.pixel_n_dim == ndims[0]


@pytest.mark.parametrize(("wcsobj", "ndims"), zip(ex_wcs, dims))
def test_world_n_dim(wcsobj, ndims):
    assert wcsobj.world_n_dim == ndims[1]


@pytest.mark.parametrize(("wcsobj", "physical_types"), zip(ex_wcs, physical_types))
def test_world_axis_physical_types(wcsobj, physical_types):
    assert wcsobj.world_axis_physical_types == physical_types


@pytest.mark.parametrize(("wcsobj", "world_units"), zip(ex_wcs, world_units))
def test_world_axis_units(wcsobj, world_units):
    assert wcsobj.world_axis_units == world_units


@pytest.mark.parametrize(("x", "y"), zip((x, xarr), (y, yarr)))
def test_pixel_to_world_values(x, y):
    wcsobj = example_wcs
    assert_allclose(wcsobj.pixel_to_world_values(x, y), wcsobj(x, y, with_units=False))


@pytest.mark.parametrize(("x", "y"), zip((x, xarr), (y, yarr)))
def test_array_index_to_world_values(x, y):
    wcsobj = example_wcs
    assert_allclose(wcsobj.array_index_to_world_values(x, y), wcsobj(y, x, with_units=False))


@pytest.mark.parametrize(("sky", "ra", "dec"), zip((sky, skyarr), (ra, raarr), (dec, decarr)))
def test_world_to_pixel_values(sky, ra, dec):
    wcsobj = example_wcs
    assert_allclose(wcsobj.world_to_pixel_values(sky), wcsobj.invert(ra, dec, with_units=False))


@pytest.mark.parametrize(("sky", "ra", "dec"), zip((sky, skyarr), (ra, raarr), (dec, decarr)))
def test_world_to_array_index_values(sky, ra, dec):
    wcsobj = example_wcs
    assert_allclose(wcsobj.world_to_array_index_values(sky),
                    wcsobj.invert(ra, dec, with_units=False)[::-1])


def test_world_axis_object_components():
    wcsobj = example_wcs
    with pytest.raises(NotImplementedError):
        wcsobj.world_axis_object_components()


def test_world_axis_object_classes():
    wcsobj = example_wcs
    with pytest.raises(NotImplementedError):
        wcsobj.world_axis_object_classes()


def test_array_shape():
    wcsobj = example_wcs
    assert wcsobj.array_shape is None

    wcsobj.array_shape = (2040, 1020)
    assert_array_equal(wcsobj.array_shape, (2040, 1020))


def test_pixel_bounds():
    wcsobj = example_wcs
    assert wcsobj.pixel_bounds is None

    wcsobj.bounding_box = ((-0.5, 2039.5), (-0.5, 1019.5))
    assert_array_equal(wcsobj.pixel_bounds, wcsobj.bounding_box)


def test_axis_correlation_matrix():
    wcsobj = example_wcs
    assert_array_equal(wcsobj.axis_correlation_matrix, np.identity(2))


def test_serialized_classes():
    wcsobj = example_wcs
    assert not wcsobj.serialized_classes()


def test_low_level_wcs():
    wcsobj = example_wcs
    assert id(wcsobj.low_level_wcs()) == id(wcsobj)


def test_pixel_to_world():
    wcsobj = example_wcs
    comp = wcsobj(x, y, with_units=True)
    comp = wcsobj.output_frame.coordinates(comp)
    result = wcsobj.pixel_to_world(x, y)
    assert isinstance(comp, coord.SkyCoord)
    assert isinstance(result, coord.SkyCoord)
    assert_allclose(comp.data.lon, result.data.lon)
    assert_allclose(comp.data.lat, result.data.lat)


def test_array_index_to_world():
    wcsobj = example_wcs
    comp = wcsobj(x, y, with_units=True)
    comp = wcsobj.output_frame.coordinates(comp)
    result = wcsobj.array_index_to_world(y, x)
    assert isinstance(comp, coord.SkyCoord)
    assert isinstance(result, coord.SkyCoord)
    assert_allclose(comp.data.lon, result.data.lon)
    assert_allclose(comp.data.lat, result.data.lat)


def test_world_to_pixel():
    wcsobj = example_wcs
    assert_allclose(wcsobj.world_to_pixel(sky), wcsobj.invert(ra, dec, with_units=False))


def test_world_to_array_index():
    wcsobj = example_wcs
    assert_allclose(wcsobj.world_to_array_index(sky), wcsobj.invert(ra, dec, with_units=False)[::-1])


def test_pixel_to_world_quantity():
    result1 = wcs_noquantity.pixel_to_world(x, y)
    result2 = wcs_quantity.pixel_to_world(x, y)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)

    # test with Quantity pixel inputs
    result1 = wcs_noquantity.pixel_to_world(x * u.pix, y * u.pix)
    result2 = wcs_quantity.pixel_to_world(x * u.pix, y * u.pix)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)

    # test for pixel units
    with pytest.raises(ValueError):
        wcs_noquantity.pixel_to_world(x * u.Jy, y * u.Jy)


def test_array_index_to_world_quantity():
    result0 = wcs_noquantity.pixel_to_world(x, y)
    result1 = wcs_noquantity.array_index_to_world(y, x)
    result2 = wcs_quantity.array_index_to_world(y, x)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)
    assert_allclose(result0.data.lon, result1.data.lon)
    assert_allclose(result0.data.lat, result1.data.lat)

    # test with Quantity pixel inputs
    result0 = wcs_noquantity.pixel_to_world(x * u.pix, y * u.pix)
    result1 = wcs_noquantity.array_index_to_world(y * u.pix, x * u.pix)
    result2 = wcs_quantity.array_index_to_world(y * u.pix, x * u.pix)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)
    assert_allclose(result0.data.lon, result1.data.lon)
    assert_allclose(result0.data.lat, result1.data.lat)

    # test for pixel units
    with pytest.raises(ValueError):
        wcs_noquantity.array_index_to_world(x * u.Jy, y * u.Jy)


def test_world_to_pixel_quantity():
    skycoord = wcs_noquantity.pixel_to_world(x, y)
    result1 = wcs_noquantity.world_to_pixel(skycoord)
    result2 = wcs_quantity.world_to_pixel(skycoord)
    assert_allclose(result1, (x, y))
    assert_allclose(result2, (x, y))

    skycoord = wcs_noquantity.pixel_to_world(xarr, yarr)
    result1 = wcs_noquantity.world_to_pixel(skycoord)
    result2 = wcs_quantity.world_to_pixel(skycoord)
    assert_allclose(result1, (xarr, yarr))
    assert_allclose(result2, (xarr, yarr))


def test_world_to_array_index_quantity():
    skycoord = wcs_noquantity.pixel_to_world(x, y)
    result0 = wcs_noquantity.world_to_pixel(skycoord)
    result1 = wcs_noquantity.world_to_array_index(skycoord)
    result2 = wcs_quantity.world_to_array_index(skycoord)
    assert_allclose(result0, (x, y))
    assert_allclose(result1, (y, x))
    assert_allclose(result2, (y, x))


def test_api_one_axis():
    lt = np.arange(5) * u.AA
    forward_transform = models.Tabular1D(lookup_table=lt, bounds_error=False)
    forward_transform.inverse = models.Tabular1D(points=lt,
                                                 lookup_table=np.arange(len(lt)),
                                                 bounds_error=False)
    w = wcs.WCS(forward_transform=forward_transform,
                input_frame=detector1d,
                output_frame=spec_only)

    q = w.pixel_to_world(x)
    x1 = w.world_to_pixel(q)
    assert isinstance(q, u.Quantity)
    assert_allclose(x, x1)

    qq = w.pixel_to_world(xarr)
    x1 = w.world_to_pixel(qq)
    assert isinstance(qq, u.Quantity)
    assert_allclose(x, x1)


def test_pixel_shape():
    assert example_wcs.pixel_shape is None
    example_wcs.pixel_shape = (20, 30)
    assert example_wcs.pixel_shape == (20, 30)
    with pytest.raises(ValueError):
        example_wcs.pixel_shape = (10, 2, 3)
