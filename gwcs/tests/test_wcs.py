# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from astropy.modeling import models
from astropy import coordinates as coord
from astropy.io import fits
from astropy import units as u
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy import wcs as astwcs

from .. import wcs
from .. import coordinate_frames as cf


def test_create_wcs():
    m = models.Shift(12.4)
    icrs = cf.CelestialFrame(coord.ICRS())
    det = cf.DetectorFrame()
    gw1 = wcs.WCS(output_frame='icrs', input_frame='detector', forward_transform=m)
    gw2 = wcs.WCS(output_frame='icrs', forward_transform=m)
    gw3 = wcs.WCS(output_frame=icrs, input_frame=det, forward_transform=m)
    assert(gw1._input_frame == gw2._input_frame == gw3._input_frame == 'detector')
    assert(gw1._output_frame == gw2._output_frame == gw3._output_frame == 'icrs')
    assert(gw1.forward_transform.__class__ == gw2.forward_transform.__class__ ==
           gw3.forward_transform.__class__ == m.__class__)
    assert(gw1._coord_frames == gw2._coord_frames == {'detector': None, 'icrs': None})


def test_pipeline_init():
    gw = wcs.WCS(output_frame='icrs')
    assert gw._pipeline == [('detector', None), ('icrs', None)]
    assert(gw.available_frames == {'detector': None, 'icrs': None})
    icrs = cf.CelestialFrame(coord.ICRS())
    det = cf.DetectorFrame()
    gw = wcs.WCS(output_frame=icrs, input_frame=det)
    assert gw._pipeline == [('detector', None), ('icrs', None)]
    assert(gw.available_frames == {'detector': det, 'icrs': icrs})


def test_insert_transform():
    m1 = models.Shift(12.4)
    m2 = models.Scale(3.1)
    gw = wcs.WCS(output_frame='icrs', forward_transform=m1)
    assert(gw.forward_transform(1.2) == m1(1.2))
    gw.insert_transform(frame='icrs', transform=m2)
    assert(gw.forward_transform(1.2) == (m1 | m2)(1.2))


class TestImaging(object):

    def setup_class(self):
        hdr = fits.Header.fromtextfile(get_pkg_data_filename("data/acs.hdr"), endcard=False)
        self.fitsw = astwcs.WCS(hdr)
        a_coeff = hdr['A_*']
        a_order = a_coeff.pop('A_ORDER')
        b_coeff = hdr['B_*']
        b_order = b_coeff.pop('B_ORDER')

        crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
        distortion = models.SIP(
            crpix, a_order, b_order, a_coeff, b_coeff, name='sip_distorion') + models.Identity(2)

        cdmat = np.array([[hdr['CD1_1'], hdr['CD1_2']], [hdr['CD2_1'], hdr['CD2_2']]])
        aff = models.AffineTransformation2D(matrix=cdmat, name='rotation')

        offx = models.Shift(-hdr['CRPIX1'], name='x_translation')
        offy = models.Shift(-hdr['CRPIX2'], name='y_translation')

        wcslin = (offx & offy) | aff

        phi = hdr['CRVAL1']
        lon = hdr['CRVAL2']
        theta = 180
        n2c = models.RotateNative2Celestial(phi, lon, theta, name='sky_rotation')

        tan = models.Pix2Sky_TAN(name='tangent_projection')
        sky_cs = cf.CelestialFrame(reference_frame=coord.ICRS())
        wcs_forward = wcslin | tan | n2c
        pipeline = [('detector', distortion),
                    ('focal', wcs_forward),
                    (sky_cs, None)
                    ]

        self.wcs = wcs.WCS(input_frame='detector',
                           output_frame=sky_cs,
                           forward_transform=pipeline)
        nx, ny = (5, 2)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        self.xv, self.yv = np.meshgrid(x, y)

    def test_forward(self):
        sky_coord = self.wcs(self.xv, self.yv)
        ra, dec = self.fitsw.all_pix2world(self.xv, self.yv, 1)
        assert_almost_equal(sky_coord[0], ra)
        assert_almost_equal(sky_coord[1], dec)

    def test_backward(self):
        transform = self.wcs.get_transform(from_frame='focal', to_frame=self.wcs.output_frame)
        sky_coord = self.wcs.transform('focal', self.wcs.output_frame, self.xv, self.yv)
        px_coord = transform.inverse(*sky_coord)
        assert_allclose(px_coord[0], self.xv, atol=10**-6)
        assert_allclose(px_coord[1], self.yv, atol=10**-6)

    def test_footprint(self):
        footprint = self.wcs.footprint((4096, 2048))
        fits_footprint = self.fitsw.calc_footprint(axes=(4096, 2048))
        assert_allclose(footprint, fits_footprint)

    def test_inverse(self):
        sky_coord = self.wcs(1, 2)
        with pytest.raises(NotImplementedError) as exc:
            detector_coord = self.wcs.invert(sky_coord[0], sky_coord[1])

    def test_units(self):
        assert(self.wcs.unit == [u.degree, u.degree])

    def test_get_transform(self):
        with pytest.raises(wcs.CoordinateFrameError):
            assert(self.wcs.get_transform('x_translation', 'sky_rotation').submodel_names ==
                   self.wcs.forward_transform[1:].submodel_names)
