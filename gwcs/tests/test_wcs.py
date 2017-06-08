# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from numpy.testing import assert_allclose
from astropy.modeling import models
from astropy import coordinates as coord
from astropy.io import fits
from astropy import units as u
import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy import wcs as astwcs

from .. import wcs
from ..wcstools import *
from .. import coordinate_frames as cf
from .. import utils
from ..utils import CoordinateFrameError, DimensionalityError


m1 = models.Shift(12.4) & models.Shift(-2)
m2 = models.Scale(2) & models.Scale(-2)
m = m1 | m2

icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs')
detector = cf.Frame2D(name='detector', axes_order=(0, 1))
focal = cf.Frame2D(name='focal', axes_order=(0, 1), unit=(u.m, u.m))
spec = cf.SpectralFrame(name='wave', unit=[u.m, ], axes_order=(2, ), axes_names=('lambda', ))

pipe = [(detector, m1),
        (focal, m2),
        (icrs, None)
        ]

# Test initializing a WCS

def test_create_wcs():
    """
    Test initializing a WCS object.
    """
    # use only frame names
    gw1 = wcs.WCS(output_frame='icrs', input_frame='detector', forward_transform=m)
    # omit input_frame
    gw2 = wcs.WCS(output_frame='icrs', forward_transform=m)
    # use CoordinateFrame objects
    gw3 = wcs.WCS(output_frame=icrs, input_frame=detector, forward_transform=m)
    # use a pipeline to initialize
    pipe = [(detector, m1), (icrs, None)]
    gw4 = wcs.WCS(forward_transform=pipe)
    assert(gw1.available_frames == gw2.available_frames ==
           gw3.available_frames == gw4.available_frames == ['detector', 'icrs'])
    res = m(1, 2)
    assert_allclose(gw1(1, 2), res)
    assert_allclose(gw2(1, 2), res)
    assert_allclose(gw3(1, 2), res)
    assert_allclose(gw3(1, 2), res)


def test_init_no_transform():
    """
    Test initializing a WCS object without a forward_transform.
    """
    gw = wcs.WCS(output_frame='icrs')
    assert gw._pipeline == [('detector', None), ('icrs', None)]
    assert np.in1d(gw.available_frames, ['detector', 'icrs']).all()
    gw = wcs.WCS(output_frame=icrs, input_frame=detector)
    assert gw._pipeline == [('detector', None), ('icrs', None)]
    assert np.in1d(gw.available_frames, ['detector', 'icrs']).all()
    with pytest.raises(NotImplementedError):
        gw(1, 2)


def test_init_no_output_frame():
    """
    Test initializing a WCS without an output_frame raises an error.
    """
    with pytest.raises(CoordinateFrameError):
        wcs.WCS(forward_transform=m1)


def test_insert_transform():
    """ Test inserting a transform."""
    gw = wcs.WCS(output_frame='icrs', forward_transform=m1)
    assert_allclose(gw.forward_transform(1, 2), m1(1, 2))
    gw.insert_transform(frame='icrs', transform=m2)
    assert_allclose(gw.forward_transform(1, 2), (m1 | m2)(1, 2))


def test_set_transform():
    """ Test setting a transform between two frames in the pipeline."""
    w = wcs.WCS(input_frame=detector, output_frame=icrs, forward_transform=pipe)
    w.set_transform('detector', 'focal', models.Identity(2))
    assert_allclose(w(1, 1), (2, -2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform('detector1', 'focal', models.Identity(2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform('detector', 'focal1', models.Identity(2))


def test_get_transform():
    """ Test getting a transform between two frames in the pipeline."""
    w = wcs.WCS(pipe)
    tr_forward = w.get_transform('detector', 'focal')
    tr_back = w.get_transform('icrs', 'detector')
    x, y = 1, 2
    fx, fy = tr_forward(1, 2)
    assert_allclose(w.pipeline[0][1](x, y), (fx, fy))
    assert_allclose((x, y), tr_back(*w(x, y)))
    assert(w.get_transform('detector', 'detector') is None)


def test_backward_transform():
    """
    Test backward transform raises an error when an analytical
    inverse is not available.
    """
    w = wcs.WCS(forward_transform=models.Polynomial1D(1) & models.Scale(2), output_frame='sky')
    with pytest.raises(NotImplementedError):
        w.backward_transform


def test_return_coordinates():
    """Test converting to coordinate objects or quantities."""
    w = wcs.WCS(pipe)
    x = 1
    y = 2.3
    numerical_result = (26.8, -0.6)
    # Celestial frame
    num_plus_output = w(x, y, output='numericals_plus')
    assert_allclose(w(x, y), numerical_result)
    assert_allclose(utils._get_values(w.unit, num_plus_output), numerical_result)
    assert_allclose(w.invert(num_plus_output), (x, y))
    assert isinstance(num_plus_output, coord.SkyCoord)
    # Spectral frame
    poly = models.Polynomial1D(1, c0=1, c1=2)
    w = wcs.WCS(forward_transform=poly, output_frame=spec)
    numerical_result = poly(y)
    num_plus_output = w(y, output='numericals_plus')
    assert_allclose(utils._get_values(w.unit, num_plus_output),  numerical_result)
    assert isinstance(num_plus_output, u.Quantity)
    # CompositeFrame - [celestial, spectral]
    output_frame = cf.CompositeFrame(frames=[icrs, spec])
    transform = m1 & poly
    w = wcs.WCS(forward_transform=transform, output_frame=output_frame)
    numerical_result = transform(x, y, y)
    num_plus_output = w(x, y, y, output='numericals_plus')
    assert_allclose(utils._get_values(w.unit, *num_plus_output), numerical_result)


def test_from_fiducial_sky():
    sky = coord.SkyCoord(1.63 * u.radian, -72.4 * u.deg, frame='fk5')
    tan = models.Pix2Sky_TAN()
    w = wcs_from_fiducial(sky, projection=tan)
    assert isinstance(w.CelestialFrame.reference_frame, coord.FK5)
    assert_allclose(w(.1, .1), (93.7210280925364, -72.29972666307474))


def test_from_fiducial_composite():
    sky = coord.SkyCoord(1.63 * u.radian, -72.4 * u.deg, frame='fk5')
    tan = models.Pix2Sky_TAN()
    spec = cf.SpectralFrame(unit=(u.micron,), axes_order=(0,))
    celestial = cf.CelestialFrame(reference_frame=sky.frame, unit=(sky.spherical.lon.unit,
                                  sky.spherical.lat.unit), axes_order=(1, 2))
    coord_frame = cf.CompositeFrame([spec, celestial], name='cube_frame')
    w = wcs_from_fiducial([.5 * u.micron, sky], coord_frame, projection=tan)
    assert isinstance(w.cube_frame.frames[1].reference_frame, coord.FK5)
    assert_allclose(w(1, 1, 1), (1.5, 96.52373368309931, -71.37420187296995))
    # test returning coordinate objects with composite output_frame
    res = w(1, 2, 2, output='numericals_plus')
    assert_allclose(res[0], u.Quantity(1.5 * u.micron))
    assert isinstance(res[1], coord.SkyCoord)
    assert_allclose(res[1].ra.value, 99.329496642319)
    assert_allclose(res[1].dec.value, -70.30322020351122)

    trans = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    w = wcs_from_fiducial([.5 * u.micron, sky], coord_frame, projection=tan,
                          transform=trans)
    assert_allclose(w(1, 1, 1), (11.5, 99.97738475762152, -72.29039139739766))
    # test coordinate object output
    coord_result = w(1, 1, 1, output='numericals_plus')
    assert_allclose(coord_result[0], u.Quantity(11.5 * u.micron))


def test_from_fiducial_frame2d():
    fiducial = (34.5, 12.3)
    w = wcs_from_fiducial(fiducial, coordinate_frame=cf.Frame2D())
    assert (w.output_frame.name == 'Frame2D')
    assert_allclose(w(1, 1), (35.5, 13.3))


def test_domain():
    trans3 = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    pipeline = [('detector', trans3), ('sky', None)]
    w = wcs.WCS(pipeline)
    bb = ((-1, 10), (6, 15))
    with pytest.raises(DimensionalityError):
        w.bounding_box = bb
    trans2 = models.Shift(10) & models.Scale(2)
    pipeline = [('detector', trans2), ('sky', None)]
    w = wcs.WCS(pipeline)
    w.bounding_box = bb
    assert w.bounding_box == w.forward_transform.bounding_box[::-1]


def test_grid_from_bounding_box():
    bb = ((-1, 9.9), (6.5, 15))
    x, y = grid_from_bounding_box(bb, step=[.1, .5], center=False)
    assert_allclose(x[:, 0], -1)
    assert_allclose(x[:, -1], 9.9)
    assert_allclose(y[0], 6.5)
    assert_allclose(y[-1], 15)


def test_bounding_box_eval():
    """
    Tests evaluation with and without respecting the bounding_box.
    """
    trans3 = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    pipeline = [('detector', trans3), ('sky', None)]
    w = wcs.WCS(pipeline)
    w.bounding_box = ((-1, 10), (6, 15), (4.3, 6.9))

    # test scalar outside bbox
    assert_allclose(w(1, 7, 3), [np.nan, np.nan, np.nan])
    assert_allclose(w(1, 7, 3, with_bounding_box=False), [11, 14, 2])
    assert_allclose(w(1, 7, 3, fill_value=100.3), [100.3, 100.3, 100.3])
    assert_allclose(w(1, 7, 3, fill_value=np.inf), [np.inf, np.inf, np.inf])
    # test scalar inside bbox
    assert_allclose(w(1, 7, 5), [11, 14, 4])
    # test arrays
    assert_allclose(w([1, 1], [7, 7], [3, 5]), [[np.nan, 11], [np.nan, 14], [np.nan, 4]])

    # test ``transform`` method
    assert_allclose(w.transform('detector', 'sky', 1, 7, 3), [np.nan, np.nan, np.nan])

    
def test_format_output():
    points = np.arange(5)
    values = np.array([1.5, 3.4, 6.7, 7, 32])
    t = models.Tabular1D(points, values)
    pipe = [('detector', t), ('world', None)]
    w = wcs.WCS(pipe)
    assert_allclose(w(1), 3.4)
    assert_allclose(w([1, 2]), [3.4, 6.7])
    assert np.isscalar(w(1))


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
        sky_cs = cf.CelestialFrame(reference_frame=coord.ICRS(), name='sky')
        det = cf.Frame2D('detector')
        wcs_forward = wcslin | tan | n2c
        pipeline = [('detector', distortion),
                    ('focal', wcs_forward),
                    (sky_cs, None)
                    ]

        self.wcs = wcs.WCS(input_frame = det,
                           output_frame = sky_cs,
                           forward_transform = pipeline)
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
        sky = self.wcs.get_transform('focal', 'sky')(self.xv, self.yv)
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
        bb = ((1, 4096), (1, 2048))
        footprint = (self.wcs.footprint(bb)).T
        fits_footprint = self.fitsw.calc_footprint(axes=(4096, 2048))
        assert_allclose(footprint, fits_footprint)

    def test_inverse(self):
        sky_coord = self.wcs(1, 2, output="numericals_plus")
        with pytest.raises(NotImplementedError):
            self.wcs.invert(sky_coord)

    def test_back_coordinates(self):
        sky_coord = self.wcs(1, 2, output="numericals_plus")
        sky2foc = self.wcs.get_transform('sky', 'focal')
        res = self.wcs.transform('sky', 'focal', sky_coord)
        assert_allclose(res, self.wcs.get_transform('detector', 'focal')(1, 2))

    def test_units(self):
        assert(self.wcs.unit == (u.degree, u.degree))

    def test_get_transform(self):
        with pytest.raises(wcs.CoordinateFrameError):
            assert(self.wcs.get_transform('x_translation', 'sky_rotation').submodel_names ==
                   self.wcs.forward_transform[1:].submodel_names)
