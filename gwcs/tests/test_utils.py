# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy import wcs as fitswcs
from astropy import units as u
from astropy import coordinates as coord
from astropy.modeling import models
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import assert_quantity_allclose
import pytest
from numpy.testing.utils import assert_allclose

from .. import utils as gwutils
from ..utils import UnsupportedProjectionError


def test_wrong_projcode():
    with pytest.raises(UnsupportedProjectionError):
        ctype = {"CTYPE": ["RA---TAM", "DEC--TAM"]}
        gwutils.get_projcode(ctype)


def test_wrong_projcode2():
    with pytest.raises(UnsupportedProjectionError):
        gwutils.create_projection_transform("TAM")


def test_fits_transform():
    hdr = fits.Header.fromfile(get_pkg_data_filename('data/simple_wcs2.hdr'))
    gw1 = gwutils.make_fitswcs_transform(hdr)
    w1 = fitswcs.WCS(hdr)
    assert_allclose(gw1(1, 2), w1.wcs_pix2world(1, 2, 1), atol=10**-8)


def test_lon_pole():
    tan = models.Pix2Sky_TAN()
    car = models.Pix2Sky_CAR()
    sky_positive_lat = coord.SkyCoord(3 * u.deg, 1 * u.deg)
    sky_negative_lat = coord.SkyCoord(3 * u.deg, -1 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_positive_lat, tan), 180 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_negative_lat, tan), 180 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_positive_lat, car), 0 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_negative_lat, car), 180 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole((0, 34 * u.rad), tan), 180 * u.deg)
    assert_allclose(gwutils._compute_lon_pole((1, -34), tan), 180)


def test_unknown_ctype():
    wcsinfo = {'CDELT': np.array([3.61111098e-05, 3.61111098e-05, 2.49999994e-03]),
               'CRPIX': np.array([17., 16., 1.]),
               'CRVAL': np.array([4.49999564e+01, 1.72786731e-04, 4.84631542e+00]),
               'CTYPE': np.array(['MRSAL1A', 'MRSBE1A', 'WAVE']),
               'CUNIT': np.array([u.Unit("deg"), u.Unit("deg"), u.Unit("um")], dtype=object),
               'PC': np.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]]),
               'WCSAXES': 3,
               'has_cd': False
               }
    transform = gwutils.make_fitswcs_transform(wcsinfo)
    x = np.linspace(-5, 7, 10)
    y = np.linspace(-5, 7, 10)
    expected = (np.array([-0.00079444, -0.0007463, -0.00069815, -0.00065, -0.00060185,
                       -0.0005537, -0.00050556, -0.00045741, -0.00040926, -0.00036111]
                      ),
                np.array([-0.00075833, -0.00071019, -0.00066204, -0.00061389, -0.00056574,
                       -0.00051759, -0.00046944, -0.0004213, -0.00037315, -0.000325]
                      )
                )
    a, b = transform(x, y)
    assert_allclose(a, expected[0], atol=10**-8)
    assert_allclose(b, expected[1], atol=10**-8)


# https://github.com/spacetelescope/gwcs/issues/139
def test_transform_hdr_dict():
    """2D test case from Ginga."""
    header = {
        'BITPIX': -32, 'BLANK': -32768, 'BUNIT': 'ADU', 'EXTEND': False,
        'CD1_1': -5.611e-05, 'CD1_2': 0.0, 'CD2_1': 0.0, 'CD2_2': 5.611e-05,
        'CDELT1': -5.611e-05, 'CDELT2': 5.611e-05,
        'CRPIX1': 5276.0, 'CRPIX2': 25.0,
        'CRVAL1': 299.91736667, 'CRVAL2': 22.68769444,
        'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
        'CUNIT1': 'degree', 'CUNIT2': 'degree', 'EQUINOX': 2000.0,
        'NAXIS': 2, 'NAXIS1': 2272, 'NAXIS2': 4273,
        'RA': '19:59:40.168', 'RA2000': '19:59:40.168',
        'DEC': '+22:41:15.70', 'DEC2000': '+22:41:15.70', 'RADECSYS': 'FK5',
        'SIMPLE': True, 'WCS-ORIG': 'SUBARU Toolkit'}
    w = gwutils.make_fitswcs_transform(header, dict_to_fits=True)
    xy_ans = np.array([120, 100])
    radec_deg_ans = (300.2308791294835, 22.691653517073615)

    # TODO: Check with Nadia
    assert_allclose(w(*xy_ans), radec_deg_ans, rtol=1e-5)
    assert_allclose(w.inverse(*radec_deg_ans), xy_ans + 1)


def test_get_axes():
    wcsinfo = {'CTYPE': np.array(['MRSAL1A', 'MRSBE1A', 'WAVE'])}
    cel, spec, other = gwutils.get_axes(wcsinfo)
    assert not cel
    assert spec == [2]
    assert other == [0, 1]
    wcsinfo = {'CTYPE': np.array(['RA---TAN', 'WAVE', 'DEC--TAN'])}
    cel, spec, other = gwutils.get_axes(wcsinfo)
    assert cel == [0, 2]
    assert spec == [1]
    assert not other
