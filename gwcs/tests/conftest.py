"""
This file contains a set of pytest fixtures which are different gwcses for testing.
"""
import pytest

import astropy.units as u
from astropy import coordinates as coord
from astropy.modeling import models
from astropy.time import Time

from .. import coordinate_frames as cf
from .. import wcs

# frames
detector_1d = cf.CoordinateFrame(name='detector', axes_order=(0,), naxes=1, axes_type="detector")
detector_2d = cf.Frame2D(name='detector', axes_order=(0, 1))
icrs_sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(),
                                   axes_order=(0, 1))

freq_frame = cf.SpectralFrame(name='freq', unit=[u.Hz], axes_order=(2, ))
wave_frame = cf.SpectralFrame(name='wave', unit=[u.m], axes_order=(2, ),
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
                models.Multiply(1*u.nm/u.pixel) & models.Multiply(1*u.nm/u.pixel))
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
