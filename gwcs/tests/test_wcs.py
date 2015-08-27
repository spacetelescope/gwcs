# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

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
from ..utils import ModelDimensionalityError, CoordinateFrameError


m1 = models.Shift(12.4) & models.Shift(-2)
m2 = models.Scale(2) & models.Scale(-2)
icrs = cf.CelestialFrame(reference_frame=coord.ICRS())
det = cf.Frame2D(name='detector', axes_order=(0, 1))
focal = cf.Frame2D(name='focal', axes_order=(0, 1), unit=(u.m, u.m))
pipe = [(det, m1),
        (focal, m2),
        (icrs, None)
        ]


def test_create_wcs():
    """
    Tests initializing a WCS object with frames of type str or CoorindtaeFrame.
    """
    icrs = cf.CelestialFrame(name='icrs', reference_frame=coord.ICRS())
    det = cf.Frame2D(name='detector', axes_order=(0,1))
    gw1 = wcs.WCS(output_frame='icrs', input_frame='detector', forward_transform=m1)
    gw2 = wcs.WCS(output_frame='icrs', forward_transform=m1)
    gw3 = wcs.WCS(output_frame=icrs, input_frame=det, forward_transform=m1)
    assert(gw1.input_frame == gw2.input_frame == gw3.input_frame == 'detector')
    assert(gw1.output_frame == gw2.output_frame == gw3.output_frame == 'icrs')
    assert(gw1.forward_transform.__class__ == gw2.forward_transform.__class__ ==
           gw3.forward_transform.__class__ == m1.__class__)
    assert np.in1d(gw1._available_frames, gw2._available_frames).all()


def test_init_no_transform():
    """
    Tests initializing a WCS object without a forward_transform.
    """
    gw = wcs.WCS(output_frame='icrs')
    assert gw._pipeline == [('detector', None), ('icrs', None)]
    assert np.in1d(gw.available_frames, ['detector', 'icrs']).all()
    icrs = cf.CelestialFrame(reference_frame=coord.ICRS())
    det = cf.Frame2D(name='detector', axes_order=(0, 1))
    gw = wcs.WCS(output_frame=icrs, input_frame=det)
    assert gw._pipeline == [('detector', None), ('CelestialFrame', None)]
    assert np.in1d(gw.available_frames, ['detector', 'CelestialFrame']).all()


def test_pipeline_init():
    """ Tests initializing a WCS object with a pipeline list."""

    gw = wcs.WCS(input_frame=det, output_frame=icrs, forward_transform=pipe)
    assert np.in1d(gw.available_frames, ['detector', 'focal', 'CelestialFrame']).all()
    assert_allclose(gw(1, 2), [26.8, 0] )


def test_insert_transform():
    """ Tests inserting a transform."""
    m1 = models.Shift(12.4)
    m2 = models.Scale(3.1)
    gw = wcs.WCS(output_frame='icrs', forward_transform=m1)
    assert(gw.forward_transform(1.2) == m1(1.2))
    gw.insert_transform(frame='icrs', transform=m2)
    assert(gw.forward_transform(1.2) == (m1 | m2)(1.2))


def test_wrong_ndim():
    """
    Tests that exception is raised if n_inputs/n_outputs does not
    match number of input/output axes.
    """
    det = cf.Frame2D(name='detector')
    icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')
    m = models.Shift(1)
    with pytest.raises(ModelDimensionalityError):
        w = wcs.WCS(output_frame='icrs', forward_transform=m, input_frame=det)
    with pytest.raises(ModelDimensionalityError):
        w = wcs.WCS(output_frame=icrs, forward_transform=m)


def test_set_transform():
    """ Tests setting a transform between two frames in the pipeline."""
    w = wcs.WCS(input_frame=det, output_frame=icrs, forward_transform=pipe)
    w.set_transform('detector', 'focal', models.Identity(2))
    assert_allclose(w(1, 1), (2, -2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform('detector1', 'focal', models.Identity(2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform('detector', 'focal1', models.Identity(2))


def test_get_transform():
    w = wcs.WCS(pipe)
    tr_forward = w.get_transform('detector', 'focal')
    tr_back = w.get_transform('focal', 'detector')
    x, y = 1, 2
    fx, fy = tr_forward(1, 2)
    assert_allclose((x, y), tr_back(fx, fy))
    assert( w.get_transform('detector', 'detector') is None)


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

    def test_distortion(self):
        sipx, sipy = self.fitsw.sip_pix2foc(self.xv, self.yv, 1)
        sipx = np.array(sipx) + 2048
        sipy = np.array(sipy) + 1024
        sip_coord = self.wcs.get_transform('detector', 'focal')(self.xv, self.yv)
        assert_allclose(sipx, sip_coord[0])
        assert_allclose(sipy, sip_coord[1])

    def test_wcslinear(self):
        ra, dec = self.fitsw.wcs_pix2world(self.xv, self.yv, 1)
        sky = self.wcs.get_transform('focal', 'CelestialFrame')(self.xv, self.yv)
        assert_allclose(ra, sky[0])
        assert_allclose(dec, sky[1])

    def test_forward(self):
        sky_coord = self.wcs(self.xv, self.yv)
        ra, dec = self.fitsw.all_pix2world(self.xv, self.yv, 1)
        assert_allclose(sky_coord[0], ra)
        assert_allclose(sky_coord[1], dec)

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
        assert(self.wcs.unit == (u.degree, u.degree))

    def test_get_transform(self):
        with pytest.raises(wcs.CoordinateFrameError):
            assert(self.wcs.get_transform('x_translation', 'sky_rotation').submodel_names ==
                   self.wcs.forward_transform[1:].submodel_names)

