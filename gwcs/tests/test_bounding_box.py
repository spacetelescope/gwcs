import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from astropy import units as u

import pytest


x = [-1, 2, 4, 13]
y = [np.nan, np.nan, 4, np.nan]
y1 = [np.nan, np.nan, 4, np.nan]


@pytest.mark.parametrize((("input", "output")), [((2, 4), (2, 4)),
                                                 ((100, 200), (np.nan, np.nan)),
                                                 ((x, x), (y, y))
                                                  ])
def test_2d_spatial(gwcs_2d_spatial_shift, input, output):
    w = gwcs_2d_spatial_shift
    w.bounding_box = ((-.5, 21), (4, 12))

    assert_array_equal(w.invert(*w(*input)), output)
    assert_array_equal(w.world_to_pixel_values(*w.pixel_to_world_values(*input)), output)
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(*input)), output)


@pytest.mark.parametrize((("input", "output")), [((2, 4), (2, 4)),
                                                 ((100, 200), (np.nan, np.nan)),
                                                 ((x, x), (y, y))
                                                  ])
def test_2d_spatial_coordinate(gwcs_2d_quantity_shift, input, output):
    w = gwcs_2d_quantity_shift
    w.bounding_box = ((-.5, 21), (4, 12))

    assert_array_equal(w.invert(*w(*input)), output)
    assert_array_equal(w.world_to_pixel_values(*w.pixel_to_world_values(*input)), output)
    assert_array_equal(w.world_to_pixel(*w.pixel_to_world(*input)), output)


@pytest.mark.parametrize((("input", "output")), [((2, 4), (2, 4)),
                                                 ((100, 200), (np.nan, np.nan)),
                                                 ((x, x), (y, y))
                                                  ])
def test_2d_spatial_coordinate_reordered(gwcs_2d_spatial_reordered, input, output):
    w = gwcs_2d_spatial_reordered
    w.bounding_box = ((-.5, 21), (4, 12))

    assert_array_equal(w.invert(*w(*input)), output)
    assert_array_equal(w.world_to_pixel_values(*w.pixel_to_world_values(*input)), output)
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(*input)), output)


@pytest.mark.parametrize((("input", "output")), [(2, 2),
                                                ((10, 200), (10, np.nan)),
                                                (x, (np.nan, 2, 4, 13)) 
                                                ])
def test_1d_freq(gwcs_1d_freq, input, output):
    w = gwcs_1d_freq
    w.bounding_box = (-.5, 21)
    print(f"input {input}, {output}")
    assert_array_equal(w.invert(w(input)), output)
    assert_array_equal(w.world_to_pixel_values(w.pixel_to_world_values(input)), output)
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(input)), output)


@pytest.mark.parametrize((("input", "output")), [((2, 4, 5), (2, 4, 5)),
                                                 ((100, 200, 5), (np.nan, np.nan, np.nan)),
                                                 ((x, x, x), (y1, y1, y1))
                                                  ])
def test_3d_spatial_wave(gwcs_3d_spatial_wave, input, output):
    w = gwcs_3d_spatial_wave
    w.bounding_box = ((-.5, 21), (4, 12), (3, 21))

    assert_array_equal(w.invert(*w(*input)), output)
    assert_array_equal(w.world_to_pixel_values(*w.pixel_to_world_values(*input)), output)
    assert_array_equal(w.world_to_pixel(*w.pixel_to_world(*input)), output)


@pytest.mark.parametrize((("input", "output")), [(2, 2),
                                                ((10, 200), (10, np.nan)),
                                                (x, (np.nan, 2, 4, 13)) 
                                                ])
def test_1d_freq_quantity(gwcs_1d_freq_quantity, input, output):
    w = gwcs_1d_freq_quantity
    #w.bounding_box = (-.5*u.pix, 21*u.pix)
    w.bounding_box = (-.5, 21)

    # assert_array_equal(w.invert(w(input)), output)
    # assert_array_equal(w.world_to_pixel_values(w.pixel_to_world_values(*input)), output)
    # assert_array_equal(w.world_to_pixel(w.pixel_to_world(input)), output)


@pytest.mark.parametrize((("input", "output")), [((2, 4), (2, 4)),
                                                 ((100, 200), (np.nan, np.nan)),
                                                 ((x, x), (y, y))
                                                  ])
def test_2d_shift_scale_quantity(gwcs_2d_shift_scale_quantity, input, output):
    w = gwcs_2d_shift_scale_quantity
    w.bounding_box = ((-.5, 21), (4, 12))

    assert_array_equal(w.invert(*w(*input)), output)
    assert_array_equal(w.world_to_pixel_values(*w.pixel_to_world_values(*input)), output)
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(*input)), output)


@pytest.mark.parametrize((("input", "output")), [((2, 4, 5), (2, 4, 5)),
                                                 ((100, 200, 5), (np.nan, np.nan, np.nan)),
                                                 ((x, x, x), (y1, y1, y1))
                                                  ])
def test_3d_identity_units(gwcs_3d_identity_units, input, output):
    w = gwcs_3d_identity_units
    w.bounding_box = ((-.5, 21), (4, 12), (1, 21))

    assert_array_equal(w.invert(*w(*input)), output)
    assert_array_equal(w.world_to_pixel_values(*w.pixel_to_world_values(*input)), output)
    assert_array_equal(w.world_to_pixel(w.pixel_to_world(*input)), output)


def test_4d_identity_units(gwcs_4d_identity_units, input, ooutput):
    w = gwcs_4d_identity_units
    w.bounding_box = ((-.5, 21), (4, 12), (1, 21), (5, 10))