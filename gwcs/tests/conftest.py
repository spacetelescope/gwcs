"""
This file contains a set of pytest fixtures which are different gwcses for testing.
"""
import pytest

from .. import examples
from .. import geometry


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
def gwcs_simple_imaging():
    return examples.gwcs_simple_imaging()


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
