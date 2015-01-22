
import numpy as np
from numpy.testing import assert_almost_equal
from astropy.modeling import models
from astropy import coordinates as coord
from astropy.io import fits
from astropy import units as u
from astropy.tests.helper import pytest

from ..wcs import WCS
from .. import coordinate_frames as cs

class TestImaging(object):
    def setup_class(self):
        hdr = fits.Header.fromtextfile('acs_wfc.hdr', endcard=False)
        a_coeff = hdr['A_*']
        a_order = a_coeff.pop('A_ORDER')
        b_coeff = hdr['B_*']
        b_order = b_coeff.pop('B_ORDER')

        crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
        distortion = models.SIP(crpix, a_order, b_order, a_coeff, b_coeff, name='sip_distorion')

        cdmat = np.array([[hdr['CD1_1'], hdr['CD1_2']], [hdr['CD2_1'], hdr['CD2_2']]])
        aff = models.AffineTransformation2D(matrix=cdmat, name='rotation')

        offx = models.Shift(-hdr['CRPIX1'], name='x_translation')
        offy = models.Shift(-hdr['CRPIX2'], name='y_translation')

        wcslin = (offx & offy) | aff

        phi = hdr['CRVAL1']
        lon = hdr['CRVAL2']
        theta= 180
        n2c = models.RotateNative2Celestial(phi, lon, theta, name='sky_rotation')

        tan = models.Pix2Sky_TAN(name='tangent_projection')

        wcs_forward = distortion | wcslin | tan | n2c
        sky_cs = cs.CelestialFrame(reference_frame=coord.ICRS())
        self.wcs = WCS(input_coordinate_system = 'detector',
                     output_coordinate_system = sky_cs,
                     forward_transform = wcs_forward)
        nx, ny = (5, 2)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        self.xv, self.yv = np.meshgrid(x, y)


    def test_forward(self):
        sky_coord = self.wcs(self.xv, self.yv)
        ra = np.array([[ 5.5263946 ,  5.52639421,  5.52639382,  5.52639343,  5.52639304],
                       [ 5.52639473,  5.52639434,  5.52639395,  5.52639356,  5.52639317]])
        dec = np.array([[-72.0517097 , -72.05170975, -72.05170979, -72.05170984, -72.05170989],
                        [-72.05170966, -72.05170971, -72.05170976, -72.05170981, -72.05170986]])
        assert_almost_equal(sky_coord.ra.value, ra)
        assert_almost_equal(sky_coord.dec.value, dec)

    def test_footprint(self):
        footprint = self.wcs.footprint((4096, 2048))
        #wfits.footprint()
        #assert_almost_equal([footprint.ra.value, footprint.dec.value], fits_footprint)

    def test_inverse(self):
        sky_coord = self.wcs(1, 2)
        with pytest.raises(NotImplementedError) as exc:
            detector_coord = self.wcs.invert(sky_coord.ra.value, sky_coord.dec.value)

    def test_units(self):
        assert(self.wcs.unit == [u.degree, u.degree])

    def test_slicing(self):
        foc2sky = self.wcs.forward_transform['x_translation' : ]
        foc2sky2 = self.wcs.forward_transform['x_translation' : 'sky_rotation']
        foc2sky3 = self.wcs.forward_transform[1 : ]
        assert (foc2sky.submodel_names == foc2sky2.submodel_names)
        assert (foc2sky.submodel_names == foc2sky3.submodel_names)

    def output_coordinate_system(self):
        #need to impement equality
        pass


    def test_name(self):
        pass

    def test_get_transform(self):
        #with valid and invalid cs
        assert(self.wcs.get_transform('x_translation', 'sky_rotation').submodel_names ==
               self.wcs.forward_transform[1:].submodel_names)

