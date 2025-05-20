import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

x = [-1, 2, 4, 13]
y = [np.nan, np.nan, 4, np.nan]
y1 = [np.nan, np.nan, 4, np.nan]


@pytest.mark.parametrize(
    (("input_", "output")),
    [((2, 4), (2, 4)), ((100, 200), (np.nan, np.nan)), ((x, x), (y, y))],
)
def test_2d_spatial(gwcs_2d_spatial_shift, input_, output):
    w = gwcs_2d_spatial_shift
    w.bounding_box = ((-0.5, 21), (4, 12))

    assert_array_equal(w.invert(*w(*input_)), output)
    assert_array_equal(
        w.world_to_pixel_values(*w.pixel_to_world_values(*input_)), output
    )
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(*input_)), output)


@pytest.mark.parametrize(
    (("input_", "output")),
    [((2, 4), (2, 4)), ((100, 200), (np.nan, np.nan)), ((x, x), (y, y))],
)
def test_2d_spatial_coordinate(gwcs_2d_quantity_shift, input_, output):
    w = gwcs_2d_quantity_shift
    w.bounding_box = ((-0.5, 21), (4, 12))

    assert_array_equal(w.invert(*w(*input_)), output)
    assert_array_equal(
        w.world_to_pixel_values(*w.pixel_to_world_values(*input_)), output
    )
    assert_array_equal(w.world_to_pixel(*w.pixel_to_world(*input_)), output)


@pytest.mark.parametrize(
    (("input_", "output")),
    [((2, 4), (2, 4)), ((100, 200), (np.nan, np.nan)), ((x, x), (y, y))],
)
def test_2d_spatial_coordinate_reordered(gwcs_2d_spatial_reordered, input_, output):
    w = gwcs_2d_spatial_reordered
    w.bounding_box = ((-0.5, 21), (4, 12))

    assert_array_equal(w.invert(*w(*input_)), output)
    assert_array_equal(
        w.world_to_pixel_values(*w.pixel_to_world_values(*input_)), output
    )
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(*input_)), output)


@pytest.mark.parametrize(
    (("input_", "output")), [(2, 2), ((10, 200), (10, np.nan)), (x, (np.nan, 2, 4, 13))]
)
def test_1d_freq(gwcs_1d_freq, input_, output):
    w = gwcs_1d_freq
    w.bounding_box = (-0.5, 21)
    assert_array_equal(w.invert(w(input_)), output)
    assert_array_equal(w.world_to_pixel_values(w.pixel_to_world_values(input_)), output)
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(input_)), output)


@pytest.mark.parametrize(
    (("input_", "output")),
    [
        ((2, 4, 5), (2, 4, 5)),
        ((100, 200, 5), (np.nan, np.nan, np.nan)),
        ((x, x, x), (y1, y1, y1)),
    ],
)
def test_3d_spatial_wave(gwcs_3d_spatial_wave, input_, output):
    w = gwcs_3d_spatial_wave
    w.bounding_box = ((-0.5, 21), (4, 12), (3, 21))

    assert_array_equal(w.invert(*w(*input_)), output)
    assert_array_equal(
        w.world_to_pixel_values(*w.pixel_to_world_values(*input_)), output
    )
    assert_array_equal(w.world_to_pixel(*w.pixel_to_world(*input_)), output)


@pytest.mark.parametrize(
    (("input_", "output")),
    [
        ((1, 2, 3, 4), (1.0, 2.0, 3.0, 4.0)),
        ((100, 3, 3, 3), (np.nan, 3, 3, 3)),
        (
            (x, x, x, x),
            [
                [np.nan, 2.0, 4.0, 13.0],
                [np.nan, 2.0, 4.0, 13.0],
                [np.nan, 2.0, 4.0, 13.0],
                [np.nan, 2.0, 4.0, np.nan],
            ],
        ),
    ],
)
def test_gwcs_spec_cel_time_4d(gwcs_spec_cel_time_4d, input_, output):
    w = gwcs_spec_cel_time_4d

    assert_allclose(w.invert(*w(*input_, with_bounding_box=False)), output, atol=1e-8)


def test_scalar_out_of_bounds(gwcs_2d_spatial_shift):
    # Test for issue #591
    w = gwcs_2d_spatial_shift
    w.bounding_box = ((-0.5, 21), (4, 12))
    assert_allclose(w.invert(-1, 0), [np.nan, np.nan])
