# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

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
