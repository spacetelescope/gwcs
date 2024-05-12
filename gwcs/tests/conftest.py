"""
This file contains a set of pytest fixtures which are different gwcses for testing.
"""
import pytest

from .. import examples
from .. import geometry
import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy import coordinates as coord
from astropy.modeling import models

from gwcs import coordinate_frames as cf
from gwcs import spectroscopy as sp
from gwcs import wcs
from gwcs import geometry

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
    return examples.gwcs_2d_quantity_shift()


@pytest.fixture
def gwcs_2d_spatial_shift():
    return examples.gwcs_2d_spatial_shift()


@pytest.fixture
def gwcs_2d_spatial_reordered():
    return examples.gwcs_2d_spatial_reordered()


@pytest.fixture
def gwcs_1d_freq():
    return examples.gwcs_1d_freq()


@pytest.fixture
def gwcs_3d_spatial_wave():
    return examples.gwcs_3d_spatial_wave()

@pytest.fixture
def gwcs_2d_shift_scale():
    return examples.gwcs_2d_shift_scale()


@pytest.fixture
def gwcs_1d_freq_quantity():
    return examples.gwcs_1d_freq_quantity()


@pytest.fixture
def gwcs_2d_shift_scale_quantity():
    return examples.gwcs_2d_shift_scale_quantity()


@pytest.fixture
def gwcs_3d_identity_units():
    return examples.gwcs_3d_identity_units()


@pytest.fixture
def gwcs_4d_identity_units():
    return examples.gwcs_4d_identity_units()


@pytest.fixture
def gwcs_simple_imaging_units():
    return examples.gwcs_simple_imaging_units()


@pytest.fixture
def gwcs_stokes_lookup():
    return examples.gwcs_stokes_lookup()


@pytest.fixture
def gwcs_3spectral_orders():
    return examples.gwcs_3spectral_orders()


@pytest.fixture
def gwcs_with_frames_strings():
    return examples.gwcs_with_frames_strings()


@pytest.fixture
def sellmeier_glass():
    return examples.sellmeier_glass()


@pytest.fixture
def sellmeier_zemax():
    return examples.sellmeier_zemax()


@pytest.fixture(scope="function")
def gwcs_3d_galactic_spectral():
    return examples.gwcs_3d_galactic_spectral()


@pytest.fixture(scope="function")
def gwcs_1d_spectral():
    return examples.gwcs_1d_spectral()


@pytest.fixture(scope="function")
def gwcs_spec_cel_time_4d():
    return examples.gwcs_spec_cel_time_4d()


@pytest.fixture(
    scope="function",
    params=[
        (2, 1, 0),
        (2, 0, 1),
        pytest.param((1, 0, 2), marks=pytest.mark.skip(reason="Fails round-trip for -TAB axis 3")),
    ]
)
def gwcs_cube_with_separable_spectral(request):
    axes_order = request.param
    return examples.gwcs_cube_with_separable_spectral(axes_order)


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
    axes_order = request.param
    return examples.gwcs_cube_with_separable_time(axes_order)


@pytest.fixture(scope="function")
def gwcs_7d_complex_mapping():
    return examples.gwcs_7d_complex_mapping()


@pytest.fixture
def spher_to_cart():
    return geometry.SphericalToCartesian()


@pytest.fixture
def cart_to_spher():
    return geometry.CartesianToSpherical()


@pytest.fixture
def gwcs_simple_imaging_no_units():
    shift_by_crpix = models.Shift(-2048) & models.Shift(-1024)
    matrix = np.array([[1.290551569736E-05, 5.9525007864732E-06],
                       [5.0226382102765E-06 , -1.2644844123757E-05]])
    rotation = models.AffineTransformation2D(matrix,
                                             translation=[0, 0])

    rotation.inverse = models.AffineTransformation2D(np.linalg.inv(matrix),
                                                     translation=[0, 0])
    tan = models.Pix2Sky_TAN()
    celestial_rotation =  models.RotateNative2Celestial(5.63056810618,
                                                        -72.05457184279,
                                                        180)
    det2sky = shift_by_crpix | rotation | tan | celestial_rotation
    det2sky.name = "linear_transform"

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
                                  unit=(u.deg, u.deg))
    pipeline = [(detector_frame, det2sky),
                (sky_frame, None)
                ]
    w = wcs.WCS(pipeline)
    w.bounding_box = ((2, 100), (5, 500))
    return w
