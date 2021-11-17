"""
This file contains a set of pytest fixtures which are different gwcses for testing.
"""
import pytest

import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy import coordinates as coord
from astropy.modeling import models

from .. import coordinate_frames as cf
from .. import spectroscopy as sp
from .. import wcs
from .. import geometry

# frames
detector_1d = cf.CoordinateFrame(name='detector', axes_order=(0,), naxes=1, axes_type="detector")
detector_2d = cf.Frame2D(name='detector', axes_order=(0, 1))
icrs_sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(),
                                   axes_order=(0, 1))

freq_frame = cf.SpectralFrame(name='freq', unit=u.Hz, axes_order=(0, ))
wave_frame = cf.SpectralFrame(name='wave', unit=u.m, axes_order=(2, ),
                              axes_names=('lambda', ))

# transforms
model_2d_shift = models.Shift(1) & models.Shift(2)
model_1d_scale = models.Scale(2)


@pytest.fixture
def gwcs_2d_quantity_shift():
    frame = cf.CoordinateFrame(name="quantity", axes_order=(0, 1), naxes=2, axes_type=("SPATIAL", "SPATIAL"), unit=(u.km, u.km))
    pipe = [(detector_2d, model_2d_shift), (frame, None)]

    return wcs.WCS(pipe)


@pytest.fixture
def gwcs_2d_spatial_shift():
    """
    A simple one step spatial WCS, in ICRS with a 1 and 2 px shift.
    """
    pipe = [(detector_2d, model_2d_shift), (icrs_sky_frame, None)]

    return wcs.WCS(pipe)


@pytest.fixture
def gwcs_2d_spatial_reordered():
    """
    A simple one step spatial WCS, in ICRS with a 1 and 2 px shift.
    """
    out_frame = cf.CelestialFrame(reference_frame=coord.ICRS(),
                                   axes_order=(1, 0))
    return wcs.WCS(model_2d_shift | models.Mapping((1, 0)), input_frame=detector_2d, output_frame=out_frame)


@pytest.fixture
def gwcs_1d_freq():
    return wcs.WCS([(detector_1d, model_1d_scale), (freq_frame, None)])


@pytest.fixture
def gwcs_3d_spatial_wave():
    comp1 = cf.CompositeFrame([icrs_sky_frame, wave_frame])
    m = model_2d_shift & model_1d_scale

    detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "z"), unit=(u.pix, u.pix, u.pix))

    return wcs.WCS([(detector_frame, m),
                    (comp1, None)])


@pytest.fixture
def gwcs_2d_shift_scale():
    m1 = models.Shift(1) & models.Shift(2)
    m2 = models.Scale(5) & models.Scale(10)
    m3 = m1 | m2
    pipe = [(detector_2d, m3), (icrs_sky_frame, None)]
    return wcs.WCS(pipe)


@pytest.fixture
def gwcs_1d_freq_quantity():

    detector_1d = cf.CoordinateFrame(name='detector', axes_order=(0,), naxes=1, unit=u.pix, axes_type="detector")
    return wcs.WCS([(detector_1d, models.Multiply(1 * u.Hz / u.pix)), (freq_frame, None)])



@pytest.fixture
def gwcs_2d_shift_scale_quantity():
    m4 = models.Shift(1 * u.pix) & models.Shift(2 * u.pix)
    m5 = models.Scale(5 * u.deg)
    m6 = models.Scale(10 * u.deg)
    m5.input_units_equivalencies = {'x': u.pixel_scale(1 * u.deg / u.pix)}
    m6.input_units_equivalencies = {'x': u.pixel_scale(1 * u.deg / u.pix)}
    m5.inverse = models.Scale(1. / 5 * u.pix)
    m6.inverse = models.Scale(1. / 10 * u.pix)
    m5.inverse.input_units_equivalencies = {
        'x': u.pixel_scale(1 * u.pix / u.deg)
    }
    m6.inverse.input_units_equivalencies = {
        'x': u.pixel_scale(1 * u.pix / u.deg)
    }
    m7 = m5 & m6
    m8 = m4 | m7
    pipe2 = [(detector_2d, m8), (icrs_sky_frame, None)]
    return wcs.WCS(pipe2)


@pytest.fixture
def gwcs_3d_identity_units():
    """
    A simple 1-1 gwcs that converts from pixels to arcseconds
    """
    identity = (models.Multiply(1 * u.arcsec / u.pixel) &
                models.Multiply(1 * u.arcsec / u.pixel) &
                models.Multiply(1 * u.nm / u.pixel))
    sky_frame = cf.CelestialFrame(axes_order=(0, 1), name='icrs',
                                  reference_frame=coord.ICRS(),
                                  axes_names=("longitude", "latitude"))
    wave_frame = cf.SpectralFrame(axes_order=(2, ), unit=u.nm, axes_names=("wavelength",))

    frame = cf.CompositeFrame([sky_frame, wave_frame])

    detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "z"), unit=(u.pix, u.pix, u.pix))

    return wcs.WCS(forward_transform=identity, output_frame=frame, input_frame=detector_frame)


@pytest.fixture
def gwcs_4d_identity_units():
    """
    A simple 1-1 gwcs that converts from pixels to arcseconds
    """
    identity = (models.Multiply(1*u.arcsec/u.pixel) & models.Multiply(1*u.arcsec/u.pixel) &
                models.Multiply(1*u.nm/u.pixel) & models.Multiply(1*u.s/u.pixel))
    sky_frame = cf.CelestialFrame(axes_order=(0, 1), name='icrs',
                                  reference_frame=coord.ICRS())
    wave_frame = cf.SpectralFrame(axes_order=(2, ), unit=u.nm)
    time_frame = cf.TemporalFrame(axes_order=(3, ), unit=u.s,
                                  reference_frame=Time("2000-01-01T00:00:00"))

    frame = cf.CompositeFrame([sky_frame, wave_frame, time_frame])

    detector_frame = cf.CoordinateFrame(name="detector", naxes=4,
                                        axes_order=(0, 1, 2, 3),
                                        axes_type=("pixel", "pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "z", "s"), unit=(u.pix, u.pix, u.pix, u.pix))

    return wcs.WCS(forward_transform=identity, output_frame=frame, input_frame=detector_frame)


@pytest.fixture
def gwcs_simple_imaging_units():
    shift_by_crpix = models.Shift(-2048*u.pix) & models.Shift(-1024*u.pix)
    matrix = np.array([[1.290551569736E-05, 5.9525007864732E-06],
                       [5.0226382102765E-06 , -1.2644844123757E-05]])
    rotation = models.AffineTransformation2D(matrix * u.deg,
                                             translation=[0, 0] * u.deg)
    rotation.input_units_equivalencies = {"x": u.pixel_scale(1*u.deg/u.pix),
                                          "y": u.pixel_scale(1*u.deg/u.pix)}
    rotation.inverse = models.AffineTransformation2D(np.linalg.inv(matrix) * u.pix,
                                                     translation=[0, 0] * u.pix)
    rotation.inverse.input_units_equivalencies = {"x": u.pixel_scale(1*u.pix/u.deg),
                                                  "y": u.pixel_scale(1*u.pix/u.deg)}
    tan = models.Pix2Sky_TAN()
    celestial_rotation =  models.RotateNative2Celestial(5.63056810618*u.deg,
                                                        -72.05457184279*u.deg,
                                                        180*u.deg)
    det2sky = shift_by_crpix | rotation | tan | celestial_rotation
    det2sky.name = "linear_transform"

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
                                  unit=(u.deg, u.deg))
    pipeline = [(detector_frame, det2sky),
                (sky_frame, None)
                ]
    return wcs.WCS(pipeline)


@pytest.fixture
def gwcs_stokes_lookup():
    transform = models.Tabular1D([0, 1, 2, 3] * u.pix, [0, 1, 2, 3] * u.one,
                                 method="nearest", fill_value=np.nan, bounds_error=False)
    frame = cf.StokesFrame()

    detector_frame = cf.CoordinateFrame(name="detector", naxes=1,
                                        axes_order=(0,),
                                        axes_type=("pixel",),
                                        axes_names=("x",), unit=(u.pix,))

    return wcs.WCS(forward_transform=transform, output_frame=frame, input_frame=detector_frame)


@pytest.fixture
def gwcs_3spectral_orders():
    comp1 = cf.CompositeFrame([icrs_sky_frame, wave_frame])
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    m = model_2d_shift & model_1d_scale

    return wcs.WCS([(detector_frame, m),
                    (comp1, None)])


@pytest.fixture
def gwcs_with_frames_strings():
    transform = models.Shift(1) & models.Shift(1) & models.Polynomial2D(1)
    pipe = [('detector', transform),
            ('world', None)
           ]
    return wcs.WCS(pipe)


@pytest.fixture
def sellmeier_glass():
    B_coef =  [0.58339748, 0.46085267, 3.8915394]
    C_coef = [0.00252643, 0.010078333, 1200.556]
    return sp.SellmeierGlass(B_coef, C_coef)


@pytest.fixture
def sellmeier_zemax():
    B_coef =  [0.58339748, 0.46085267, 3.8915394]
    C_coef = [0.00252643, 0.010078333, 1200.556]
    D_coef = [-2.66e-05, 0.0, 0.0]
    E_coef = [0., 0., 0.]
    return sp.SellmeierZemax(65, 35, 0, 0, B_coef = B_coef,
                             C_coef=C_coef, D_coef=D_coef,
                             E_coef=E_coef)


@pytest.fixture(scope="function")
def gwcs_3d_galactic_spectral():
    """
    This fixture has the axes ordered as lat, spectral, lon.
    """
    #                       lat,wav,lon
    crpix1, crpix2, crpix3 = 29, 39, 44
    crval1, crval2, crval3 = 10, 20, 25
    cdelt1, cdelt2, cdelt3 = -0.01, 0.5, 0.01

    shift = models.Shift(-crpix3) & models.Shift(-crpix1)
    scale = models.Multiply(cdelt3) & models.Multiply(cdelt1)
    proj = models.Pix2Sky_CAR()
    skyrot = models.RotateNative2Celestial(crval3, 90 + crval1, 180)
    celestial = shift | scale | proj | skyrot

    wave_model = models.Shift(-crpix2) | models.Multiply(cdelt2) | models.Shift(crval2)

    transform = models.Mapping((2, 0, 1)) | celestial & wave_model | models.Mapping((1, 2, 0))
    transform.bounding_box = ((5, 50), (-2, 45), (-1, 35))

    sky_frame = cf.CelestialFrame(axes_order=(2, 0),
                                  reference_frame=coord.Galactic(), axes_names=("Longitude", "Latitude"))
    wave_frame = cf.SpectralFrame(axes_order=(1, ), unit=u.Hz, axes_names=("Frequency",))

    frame = cf.CompositeFrame([sky_frame, wave_frame])

    detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        unit=(u.pix, u.pix, u.pix))

    owcs = wcs.WCS(forward_transform=transform, output_frame=frame, input_frame=detector_frame)
    owcs.array_shape = (30, 20, 10)
    owcs.pixel_shape = (10, 20, 30)

    return owcs


@pytest.fixture(scope="function")
def gwcs_1d_spectral():
    """
    A simple 1D spectral WCS.
    """
    wave_model = models.Shift(-5) | models.Multiply(3.7) | models.Shift(20)
    wave_model.bounding_box = (7, 50)
    wave_frame = cf.SpectralFrame(axes_order=(0, ), unit=u.Hz, axes_names=("Frequency",))

    detector_frame = cf.CoordinateFrame(
        name="detector", naxes=1, axes_order=(0, ),
        axes_type=("pixel",), unit=(u.pix, )
    )

    owcs = wcs.WCS(forward_transform=wave_model, output_frame=wave_frame, input_frame=detector_frame)
    owcs.array_shape = (44, )
    owcs.pixel_shape = (44, )

    return owcs


@pytest.fixture(scope="function")
def gwcs_spec_cel_time_4d():
    """
    A complex 4D mixed celestial + spectral + time WCS.
    """
    # spectroscopic frame:
    wave_model = models.Shift(-5) | models.Multiply(3.7) | models.Shift(20)
    wave_model.bounding_box = (7, 50)
    wave_frame = cf.SpectralFrame(name='wave', unit=u.m, axes_order=(0,), axes_names=('lambda',))

    # time frame:
    time_model = models.Identity(1)  # models.Linear1D(10, 0)
    time_frame = cf.TemporalFrame(Time("2010-01-01T00:00"), name='time', unit=u.s, axes_order=(3,))

    # Values from data/acs.hdr:
    crpix = (12, 13)
    crval = (5.63, -72.05)
    cd = [[1.291E-05, 5.9532E-06], [5.02215E-06, -1.2645E-05]]
    aff = models.AffineTransformation2D(matrix=cd, name='rotation')
    offx = models.Shift(-crpix[0], name='x_translation')
    offy = models.Shift(-crpix[1], name='y_translation')
    wcslin = models.Mapping((1, 0)) | (offx & offy) | aff
    tan = models.Pix2Sky_TAN(name='tangent_projection')
    n2c = models.RotateNative2Celestial(*crval, 180, name='sky_rotation')
    cel_model = wcslin | tan | n2c
    icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), name='sky', axes_order=(2, 1))

    wcs_forward = wave_model & cel_model & time_model

    comp_frm = cf.CompositeFrame(frames=[wave_frame, icrs, time_frame], name='TEST 4D FRAME')

    detector_frame = cf.CoordinateFrame(
        name="detector", naxes=4, axes_order=(0, 1, 2, 3),
        axes_type=("pixel", "pixel", "pixel", "pixel"),
        unit=(u.pix, u.pix, u.pix, u.pix)
    )

    w = wcs.WCS(forward_transform=wcs_forward, output_frame=comp_frm, input_frame=detector_frame)

    w.bounding_box = ((0, 63), (0, 127), (0, 255), (0, 9))
    w.array_shape = (10, 256, 128, 64)
    w.pixel_shape = (64, 128, 256, 10)
    return w


@pytest.fixture(
    scope="function",
    params=[
        (2, 1, 0),
        (2, 0, 1),
        pytest.param((1, 0, 2), marks=pytest.mark.skip(reason="Fails round-trip for -TAB axis 3")),
    ]
)
def gwcs_cube_with_separable_spectral(request):
    cube_size = (128, 64, 100)

    axes_order = request.param
    spectral_axes_order = (axes_order.index(2), )
    cel_axes_order = (axes_order.index(0), axes_order.index(1))

    # Values from data/acs.hdr:
    crpix = (64, 32)
    crval = (5.63056810618, -72.0545718428)
    cd = [[1.29058667557984E-05, 5.95320245884555E-06],
          [5.02215195623825E-06, -1.2645010396976E-05]]

    aff = models.AffineTransformation2D(matrix=cd, name='rotation')
    offx = models.Shift(-crpix[0], name='x_translation')
    offy = models.Shift(-crpix[1], name='y_translation')

    wcslin = (offx & offy) | aff
    tan = models.Pix2Sky_TAN(name='tangent_projection')
    n2c = models.RotateNative2Celestial(*crval, 180, name='sky_rotation')
    icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), name='sky',
                             axes_order=cel_axes_order)
    spec = cf.SpectralFrame(
        name='wave', unit=[u.m,], axes_order=spectral_axes_order,
        axes_names=('lambda',)
    )
    comp_frm = cf.CompositeFrame(frames=[icrs, spec], name='TEST 3D FRAME WITH SPECTRAL AXIS')
    wcs_forward = ((wcslin & models.Identity(1)) |
                   (tan & models.Identity(1)) |
                   (n2c & models.Identity(1)) |
                   models.Mapping(axes_order))

    detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        unit=(u.pix, u.pix, u.pix))

    w = wcs.WCS(forward_transform=wcs_forward, output_frame=comp_frm,
                input_frame=detector_frame)
    w.bounding_box = tuple((0, k - 1) for k in cube_size)
    w.pixel_shape = cube_size
    w.array_shape = w.pixel_shape[::-1]

    return w, axes_order


@pytest.fixture(
    scope="function",
    params=[
        (2, 0, 1),
        (2, 1, 0),
        pytest.param((0, 2, 1), marks=pytest.mark.skip(reason="Fails round-trip for -TAB axis 2")),
        pytest.param((1, 0, 2), marks=pytest.mark.skip(reason="Fails round-trip for -TAB axis 3")),
    ]
)
def gwcs_cube_with_separable_time(request):
    """
    A mixed celestial + time WCS.
    """
    cube_size = (64, 32, 128)

    axes_order = request.param
    time_axes_order = (axes_order.index(2), )
    cel_axes_order = (axes_order.index(0), axes_order.index(1))

    detector_frame = cf.CoordinateFrame(
        name="detector", naxes=3, axes_order=(0, 1, 2),
        axes_type=("pixel", "pixel", "pixel"), unit=(u.pix, u.pix, u.pix)
    )

    # time frame:
    time_model = models.Identity(1)  # models.Linear1D(10, 0)
    time_frame = cf.TemporalFrame(Time("2010-01-01T00:00"), name='time',
                                  unit=u.s, axes_order=time_axes_order)

    # Values from data/acs.hdr:
    crpix = (12, 13)
    crval = (5.63, -72.05)
    cd = [[1.291E-05, 5.9532E-06], [5.02215E-06, -1.2645E-05]]
    aff = models.AffineTransformation2D(matrix=cd, name='rotation')
    offx = models.Shift(-crpix[0], name='x_translation')
    offy = models.Shift(-crpix[1], name='y_translation')
    wcslin = models.Mapping((1, 0)) | (offx & offy) | aff
    tan = models.Pix2Sky_TAN(name='tangent_projection')
    n2c = models.RotateNative2Celestial(*crval, 180, name='sky_rotation')
    cel_model = wcslin | tan | n2c
    icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), name='sky',
                             axes_order=cel_axes_order)

    wcs_forward = (cel_model & time_model) | models.Mapping(axes_order)

    comp_frm = cf.CompositeFrame(frames=[icrs, time_frame],
                                 name='TEST 3D FRAME WITH TIME')

    w = wcs.WCS(forward_transform=wcs_forward, output_frame=comp_frm,
                input_frame=detector_frame)

    w.bounding_box = tuple((0, k - 1) for k in cube_size)
    w.pixel_shape = cube_size
    w.array_shape = w.pixel_shape[::-1]
    return w


@pytest.fixture(scope="function")
def gwcs_7d_complex_mapping():
    """
    Useful features of this WCS (axes indices here are 0-based):
        - includes two celestial axes: input (0, 1) maps to world (2 - RA, 1 - Dec)
        - includes one separable frame with one axis: 4 -> 2
        - includes one frame with 3 input and 4 output axes (1 degenerate),
          with separable world axes (3, 5) and (0, 6).
    """
    offx = models.Shift(-64, name='x_translation')
    offy = models.Shift(-32, name='y_translation')
    cd = np.array([[1.2906, 0.59532], [0.50222, -1.2645]])
    aff = models.AffineTransformation2D(matrix=1e-5 * cd, name='rotation')
    aff2 = models.AffineTransformation2D(matrix=cd, name='rotation2')

    wcslin = (offx & offy) | aff
    tan = models.Pix2Sky_TAN(name='tangent_projection')
    n2c = models.RotateNative2Celestial(5.630568, -72.0546, 180, name='skyrot')
    icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), name='sky', axes_order=(2, 1))
    spec = cf.SpectralFrame(name='wave', unit=[u.m], axes_order=(4,), axes_names=('lambda',))
    cmplx = cf.CoordinateFrame(
        name="complex",
        naxes=4,
        axes_order=(3, 5, 0, 6),
        axis_physical_types=(['em.wl', 'em.wl', 'time', 'time']),
        axes_type=("SPATIAL", "SPATIAL", "TIME", "TIME"),
        axes_names=("x", "y", "t", 'tau'),
        unit=(u.m, u.m, u.second, u.second)
    )

    comp_frm = cf.CompositeFrame(frames=[icrs, spec, cmplx], name='TEST 7D')
    wcs_forward = ((wcslin & models.Shift(-3.14) & models.Scale(2.7) & aff2) |
                   (tan & models.Identity(1) & models.Identity(1) & models.Identity(2)) |
                   (n2c & models.Identity(1) & models.Identity(1) & models.Identity(2)) |
                   models.Mapping((3, 1, 0, 4, 2, 5, 3)))


    detector_frame = cf.CoordinateFrame(
        name="detector", naxes=6,
        axes_order=(0, 1, 2, 3, 4, 5),
        axes_type=("pixel", "pixel", "pixel", "pixel", "pixel", "pixel"),
        unit=(u.pix, u.pix, u.pix, u.pix, u.pix, u.pix)
    )

    pipeline = [('detector', wcs_forward), (comp_frm, None)]
    w = wcs.WCS(forward_transform=wcs_forward, output_frame=comp_frm,
                input_frame=detector_frame)
    w.bounding_box = ((0, 15), (0, 31), (0, 20), (0, 10), (0, 10), (0, 1))

    w.array_shape = (2, 11, 11, 21, 32, 16)
    w.pixel_shape = (16, 32, 21, 11, 11, 2)

    return w


@pytest.fixture
def spher_to_cart():
    return geometry.SphericalToCartesian()


@pytest.fixture
def cart_to_spher():
    return geometry.CartesianToSpherical()
