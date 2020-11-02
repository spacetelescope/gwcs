"""
This file contains a set of pytest fixtures which are different gwcses for testing.
"""
import pytest

import numpy as np

import astropy.units as u
from astropy import coordinates as coord
from astropy.modeling import models
from astropy.time import Time

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
    cdelt1, cdelt2, cdelt3 = -0.1, 0.5, 0.1

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


@pytest.fixture
def spher_to_cart():
    return geometry.SphericalToCartesian()


@pytest.fixture
def cart_to_spher():
    return geometry.CartesianToSpherical()
