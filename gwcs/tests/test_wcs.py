# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.modeling import models
from astropy import coordinates as coord
from astropy.io import fits
from astropy import units as u
import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy import wcs as astwcs

from .. import wcs
from ..wcstools import (wcs_from_fiducial, grid_from_bounding_box, wcs_from_points)
from .. import coordinate_frames as cf
from .. import utils
from ..utils import CoordinateFrameError
import asdf

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

# Create some data.
nx, ny = (5, 2)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)

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
    gw5 = wcs.WCS(output_frame=icrs, input_frame=detector, forward_transform=[m1, m2])
    assert(gw1.available_frames == gw2.available_frames == \
           gw3.available_frames == gw4.available_frames == ['detector', 'icrs'])
    res = m(1, 2)
    assert_allclose(gw1(1, 2), res)
    assert_allclose(gw2(1, 2), res)
    assert_allclose(gw3(1, 2), res)
    assert_allclose(gw3(1, 2), res)
    assert_allclose(gw5(1, 2), res)

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
    w = wcs.WCS(forward_transform=pipe[:])
    w.set_transform('detector', 'focal', models.Identity(2))
    assert_allclose(w(1, 1), (2, -2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform('detector1', 'focal', models.Identity(2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform('detector', 'focal1', models.Identity(2))


def test_get_transform():
    """ Test getting a transform between two frames in the pipeline."""
    w = wcs.WCS(pipe[:])
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
    # Test that an error is raised when one of the models has not inverse.
    poly = models.Polynomial1D(1, c0=4)
    w = wcs.WCS(forward_transform=poly & models.Scale(2), output_frame='sky')
    with pytest.raises(NotImplementedError):
        w.backward_transform

    # test backward transform
    poly.inverse = models.Shift(-4)
    w = wcs.WCS(forward_transform=poly & models.Scale(2), output_frame='sky')
    assert_allclose(w.backward_transform(1, 2), (-3, 1))


def test_return_coordinates():
    """Test converting to coordinate objects or quantities."""
    w = wcs.WCS(pipe[:])
    x = 1
    y = 2.3
    numerical_result = (26.8, -0.6)
    # Celestial frame
    num_plus_output = w(x, y, with_units=True)
    output_quant = w.output_frame.coordinate_to_quantity(num_plus_output)
    assert_allclose(w(x, y), numerical_result)
    assert_allclose(utils.get_values(w.unit, *output_quant), numerical_result)
    assert_allclose(w.invert(num_plus_output), (x, y))
    assert isinstance(num_plus_output, coord.SkyCoord)

    # Spectral frame
    poly = models.Polynomial1D(1, c0=1, c1=2)
    w = wcs.WCS(forward_transform=poly, output_frame=spec)
    numerical_result = poly(y)
    num_plus_output = w(y, with_units=True)
    output_quant = w.output_frame.coordinate_to_quantity(num_plus_output)
    assert_allclose(utils.get_values(w.unit, output_quant), numerical_result)
    assert isinstance(num_plus_output, u.Quantity)

    # CompositeFrame - [celestial, spectral]
    output_frame = cf.CompositeFrame(frames=[icrs, spec])
    transform = m1 & poly
    w = wcs.WCS(forward_transform=transform, output_frame=output_frame)
    numerical_result = transform(x, y, y)
    num_plus_output = w(x, y, y, with_units=True)
    output_quant = w.output_frame.coordinate_to_quantity(*num_plus_output)
    assert_allclose(utils.get_values(w.unit, *output_quant), numerical_result)


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
    w = wcs_from_fiducial([.5, sky], coord_frame, projection=tan)
    assert isinstance(w.cube_frame.frames[1].reference_frame, coord.FK5)
    assert_allclose(w(1, 1, 1), (1.5, 96.52373368309931, -71.37420187296995))
    # test returning coordinate objects with composite output_frame
    res = w(1, 2, 2, with_units=True)
    assert_allclose(res[0], u.Quantity(1.5 * u.micron))
    assert isinstance(res[1], coord.SkyCoord)
    assert_allclose(res[1].ra.value, 99.329496642319)
    assert_allclose(res[1].dec.value, -70.30322020351122)

    trans = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    w = wcs_from_fiducial([.5, sky], coord_frame, projection=tan,
                          transform=trans)
    assert_allclose(w(1, 1, 1), (11.5, 99.97738475762152, -72.29039139739766))
    # test coordinate object output

    coord_result = w(1, 1, 1, with_units=True)
    assert_allclose(coord_result[0], u.Quantity(11.5 * u.micron))


def test_from_fiducial_frame2d():
    fiducial = (34.5, 12.3)
    w = wcs_from_fiducial(fiducial, coordinate_frame=cf.Frame2D())
    assert (w.output_frame.name == 'Frame2D')
    assert_allclose(w(1, 1), (35.5, 13.3))


def test_bounding_box():
    trans3 = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    pipeline = [('detector', trans3), ('sky', None)]
    w = wcs.WCS(pipeline)
    bb = ((-1, 10), (6, 15))
    with pytest.raises(ValueError):
        w.bounding_box = bb
    trans2 = models.Shift(10) & models.Scale(2)
    pipeline = [('detector', trans2), ('sky', None)]
    w = wcs.WCS(pipeline)
    w.bounding_box = bb
    assert w.bounding_box == w.forward_transform.bounding_box[::-1]

    pipeline = [("detector", models.Shift(2)), ("sky", None)]
    w = wcs.WCS(pipeline)
    w.bounding_box = (1, 5)
    assert w.bounding_box == w.forward_transform.bounding_box
    with pytest.raises(ValueError):
        w.bounding_box = ((1, 5), (2, 6))


def test_grid_from_bounding_box():
    bb = ((-1, 9.9), (6.5, 15))
    x, y = grid_from_bounding_box(bb, step=[.1, .5], center=False)
    assert_allclose(x[:, 0], -1)
    assert_allclose(x[:, -1], 9.9)
    assert_allclose(y[0], 6.5)
    assert_allclose(y[-1], 15)


def test_grid_from_bounding_box_1d():
    # Test 1D case
    x = grid_from_bounding_box((-.5, 4.5))
    assert_allclose(x, [0., 1., 2., 3., 4.])


def test_grid_from_bounding_box_step():
    bb = ((-0.5, 5.5), (-0.5, 4.5))
    x, y = grid_from_bounding_box(bb)
    x1, y1 = grid_from_bounding_box(bb, step=(1, 1))
    assert_allclose(x, x1)
    assert_allclose(y, y1)

    with pytest.raises(ValueError):
        grid_from_bounding_box(bb, step=(1, 2, 1))


@pytest.mark.remote_data
def test_wcs_from_points():
    np.random.seed(0)
    hdr = fits.Header.fromtextfile(get_pkg_data_filename("data/acs.hdr"), endcard=False)
    with warnings.catch_warnings() as w:
        warnings.simplefilter("ignore")
        w = astwcs.WCS(hdr)
    y, x = np.mgrid[:2046:20j, :4023:10j]
    ra, dec = w.wcs_pix2world(x, y, 1)
    fiducial = coord.SkyCoord(ra.mean()*u.deg, dec.mean()*u.deg, frame="icrs")
    w = wcs_from_points(xy=(x, y), world_coordinates=(ra, dec), fiducial=fiducial)
    newra, newdec = w(x, y)
    assert_allclose(newra, ra)
    assert_allclose(newdec, dec)

    n = np.random.randn(ra.size)
    n.shape = ra.shape
    nra = n * 10 ** -2
    ndec = n * 10 ** -2
    w = wcs_from_points(xy=(x + nra, y + ndec),
                        world_coordinates=(ra, dec),
                        fiducial=fiducial)
    newra, newdec = w(x, y)
    assert_allclose(newra, ra, atol=10**-6)
    assert_allclose(newdec, dec, atol=10**-6)


def test_grid_from_bounding_box_2():
    bb = ((-0.5, 5.5), (-0.5, 4.5))
    x, y = grid_from_bounding_box(bb)
    assert_allclose(x, np.repeat([np.arange(6)], 5, axis=0))
    assert_allclose(y, np.repeat(np.array([np.arange(5)]), 6, 0).T)

    bb = ((-0.5, 5.5), (-0.5, 4.6))
    x, y = grid_from_bounding_box(bb, center=True)
    assert_allclose(x, np.repeat([np.arange(6)], 6, axis=0))
    assert_allclose(y, np.repeat(np.array([np.arange(6)]), 6, 0).T)


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


def test_available_frames():
    w = wcs.WCS(pipe)
    assert w.available_frames == ['detector', 'focal', 'icrs']


def test_footprint():
    icrs = cf.CelestialFrame(name='icrs', reference_frame=coord.ICRS(),
                             axes_order=(0, 1))
    spec = cf.SpectralFrame(name='freq', unit=[u.Hz, ], axes_order=(2, ))
    world = cf.CompositeFrame([icrs, spec])
    transform = (models.Shift(10) & models.Shift(-1)) & models.Scale(2)
    pipe = [('det', transform), (world, None)]
    w = wcs.WCS(pipe)

    with pytest.raises(TypeError):
        w.footprint()

    w.bounding_box = ((1,5), (1,3), (1, 6))

    assert_equal(w.footprint(), np.array([[11, 0, 2],
                                          [11, 0, 12],
                                          [11, 2, 2],
                                          [11, 2, 12],
                                          [15, 0, 2],
                                          [15, 0, 12],
                                          [15, 2, 2],
                                          [15, 2, 12]]))
    assert_equal(w.footprint(axis_type='spatial'), np.array([[11., 0.],
                                                             [11., 2.],
                                                             [15., 2.],
                                                             [15., 0.]]))

    assert_equal(w.footprint(axis_type='spectral'), np.array([2, 12]))


def test_high_level_api():
    """
    Test WCS high level API.
    """
    output_frame = cf.CompositeFrame(frames=[icrs, spec])
    transform = m1 & models.Scale(1.5)
    det = cf.CoordinateFrame(naxes=3, unit=(u.pix, u.pix, u.pix),
                             axes_order=(0, 1, 2),
                             axes_type=('length', 'length', 'length'))
    w = wcs.WCS(forward_transform=transform, output_frame=output_frame, input_frame=det)

    r, d, lam = w(xv, yv, xv)
    world_coord = w.pixel_to_world(xv, yv, xv)
    assert isinstance(world_coord[0], coord.SkyCoord)
    assert isinstance(world_coord[1], u.Quantity)
    assert_allclose(world_coord[0].data.lon.value, r)
    assert_allclose(world_coord[0].data.lat.value, d)
    assert_allclose(world_coord[1].value, lam)

    x1, y1, z1 = w.world_to_pixel(*world_coord)
    assert_allclose(x1, xv)
    assert_allclose(y1, yv)
    assert_allclose(z1, xv)


@pytest.mark.remote_data
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
        det = cf.Frame2D(name='detector')
        wcs_forward = wcslin | tan | n2c
        pipeline = [('detector', distortion),
                    ('focal', wcs_forward),
                    (sky_cs, None)
                    ]

        self.wcs = wcs.WCS(input_frame=det,
                           output_frame=sky_cs,
                           forward_transform=pipeline)
        self.xv, self.yv = xv, yv

    @pytest.mark.filterwarnings('ignore')
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
        footprint = (self.wcs.footprint(bb))
        fits_footprint = self.fitsw.calc_footprint(axes=(4096, 2048))
        assert_allclose(footprint, fits_footprint)

    def test_inverse(self):
        sky_coord = self.wcs(1, 2, with_units=True)
        with pytest.raises(NotImplementedError):
            self.wcs.invert(sky_coord)

    def test_back_coordinates(self):
        sky_coord = self.wcs(1, 2, with_units=True)
        res = self.wcs.transform('sky', 'focal', sky_coord)
        assert_allclose(res, self.wcs.get_transform('detector', 'focal')(1, 2))

    def test_units(self):
        assert(self.wcs.unit == (u.degree, u.degree))

    def test_get_transform(self):
        with pytest.raises(wcs.CoordinateFrameError):
            assert(self.wcs.get_transform('x_translation', 'sky_rotation').submodel_names == \
                   self.wcs.forward_transform[1:].submodel_names)

    def test_pixel_to_world(self):
        sky_coord = self.wcs.pixel_to_world(self.xv, self.yv)
        ra, dec = self.fitsw.all_pix2world(self.xv, self.yv, 1)
        assert isinstance(sky_coord, coord.SkyCoord)
        assert_allclose(sky_coord.data.lon.value, ra)
        assert_allclose(sky_coord.data.lat.value, dec)


def test_to_fits_sip():
    y, x = np.mgrid[:1024:10, :1024:10]
    xflat = np.ravel(x[1:-1, 1:-1])
    yflat = np.ravel(y[1:-1, 1:-1])
    af = asdf.open(get_pkg_data_filename('data/miriwcs.asdf'))
    miriwcs = af.tree['wcs']
    bounding_box = ((0, 1024), (0, 1024))
    mirisip = miriwcs.to_fits_sip(bounding_box, max_inv_pix_error=0.1)
    fitssip = astwcs.WCS(mirisip)
    fitsvalx, fitsvaly = fitssip.all_pix2world(xflat+1, yflat+1, 1)
    gwcsvalx, gwcsvaly = miriwcs(xflat, yflat)
    assert_allclose(gwcsvalx, fitsvalx, atol=1e-10, rtol=0)
    assert_allclose(gwcsvaly, fitsvaly, atol=1e-10, rtol=0)    
    fits_inverse_valx, fits_inverse_valy = fitssip.all_world2pix(fitsvalx, fitsvaly, 1)
    assert_allclose(xflat, fits_inverse_valx - 1, atol=0.1, rtol=0)
    assert_allclose(yflat, fits_inverse_valy - 1, atol=0.1, rtol=0)


def test_replacing_models():
    res = m(1, 2)
    m1.name = 'shift'
    gw = wcs.WCS(output_frame='icrs', forward_transform=[m1, m2])
    assert_allclose(gw(1, 2), res)

    m1a = m1[0] & models.Shift(2)
    res2 = (m1a | m2)(1, 2)
    utils.replace_model(gw.pipeline[0][1], 'shift', m1[0] & models.Shift(2))
    assert_allclose(gw(1, 2), res2)
