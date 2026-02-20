"""
This file contains a set of pytest fixtures which are different gwcses for testing.
"""

import pytest

from gwcs import examples, geometry


@pytest.fixture
def gwcs_simple_2d():
    return examples.gwcs_simple_2d()


@pytest.fixture
def gwcs_empty_output_2d():
    with pytest.warns(DeprecationWarning, match=r"The use of strings.*"):
        return examples.gwcs_empty_output_2d()


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
    with pytest.warns(DeprecationWarning, match=r"The use of strings.*"):
        return examples.gwcs_with_frames_strings()


@pytest.fixture
def sellmeier_glass():
    return examples.sellmeier_glass()


@pytest.fixture
def sellmeier_zemax():
    return examples.sellmeier_zemax()


@pytest.fixture
def gwcs_3d_galactic_spectral():
    return examples.gwcs_3d_galactic_spectral()


@pytest.fixture
def gwcs_1d_spectral():
    return examples.gwcs_1d_spectral()


@pytest.fixture
def gwcs_spec_cel_time_4d():
    return examples.gwcs_spec_cel_time_4d()


@pytest.fixture(
    params=[
        (2, 0, 1),
        (2, 1, 0),
        (0, 2, 1),
        (1, 0, 2),
    ],
)
def axes_order(request):
    return request.param


@pytest.fixture
def gwcs_cube_with_separable_spectral(axes_order):
    return examples.gwcs_cube_with_separable_spectral(axes_order)


@pytest.fixture
def gwcs_cube_with_separable_time(axes_order):
    return examples.gwcs_cube_with_separable_time(axes_order)


@pytest.fixture
def gwcs_7d_complex_mapping():
    return examples.gwcs_7d_complex_mapping()


@pytest.fixture
def spher_to_cart():
    return geometry.SphericalToCartesian()


@pytest.fixture
def cart_to_spher():
    return geometry.CartesianToSpherical()


@pytest.fixture
def gwcs_with_pipeline_celestial():
    return examples.gwcs_with_pipeline_celestial()


@pytest.fixture
def gwcs_romanisim():
    return examples.gwcs_romanisim()


@pytest.fixture(
    params=[
        (5.6, -72.4),
        (5.6, 90),
        (5.6, -90),
        (5.6, 0),
    ],
)
def fits_wcs_imaging_simple(request):
    params = request.param
    return examples.fits_wcs_imaging_simple(params)


@pytest.fixture
def gwcs_2d_spatial_shift_reverse():
    return examples.gwcs_2d_spatial_shift_reverse()


@pytest.fixture
def gwcs_multi_stage():
    return examples.gwcs_multi_stage()
