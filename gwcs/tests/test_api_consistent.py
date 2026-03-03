# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test the API is consistent with units and quantities and follows the rules below.

WCS functions considered part of the legacy API:
wcs()
wcs.invert()
wcs.transform()

Rules:


1. Neither transforms nor inputs support units -> the output is clearly numerical
   for all functions above
2. Transforms support units but inputs do not -> return numbers/arrays
   Attach the units of the input coordinate frame to the inputs.
   Evaluate the transforms and strip the output of units
3. Both transforms and inputs support units -> return quantities
4. Transforms do not support units but inputs are quantities -> return quantities
   Strip the units from the inputs after converting to the units of the input frame.
   Evaluate the transform and attach the units of the output frame.
5. Inputs are High Level Objects - raise an error

"""

import numbers

import numpy as np
import pytest
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_allclose

x = 1
y = 2
xq = [1, 1] * u.pix
yq = 2 * u.pix


def is_numerical(args):
    return isinstance(args, numbers.Number) or all(
        isinstance(arg, numbers.Number) or arg is np.ndarray for arg in args
    )


def is_quantity(args):
    return all(isinstance(arg, u.Quantity) for arg in args)


@pytest.fixture
def wcsobj(request):
    return request.getfixturevalue(request.param)


wno_unit_1d = [
    "gwcs_1d_freq",
    "gwcs_1d_spectral",
]

wno_unit_nd = [
    "gwcs_2d_shift_scale",
    "gwcs_3d_spatial_wave",
    "gwcs_2d_spatial_shift",
    "gwcs_2d_spatial_reordered",
    "gwcs_3d_spatial_wave",
    "gwcs_simple_imaging",
    "gwcs_3spectral_orders",
    "gwcs_3d_galactic_spectral",
    "gwcs_spec_cel_time_4d",
    "gwcs_romanisim",
]

# "gwcs_7d_complex_mapping" errors in astropy - fix
# "gwcs_2d_quantity_shift" errors when inputs are quantities.
# Need to confirm if Qs are HLO

w_unit_1d = ["gwcs_stokes_lookup", "gwcs_1d_freq_quantity"]

w_unit_nd = [
    "gwcs_2d_shift_scale_quantity",
    "gwcs_3d_identity_units",
    "gwcs_3d_identity_units",
    "gwcs_4d_identity_units",
    "gwcs_simple_imaging_units",
    "gwcs_with_pipeline_celestial",
]

w_transform_test = ["gwcs_1d_freq_quantity", "gwcs_2d_quantity_shift"]

wcs_no_unit_1d = pytest.mark.parametrize(("wcsobj"), wno_unit_1d, indirect=True)
wcs_no_unit_nd = pytest.mark.parametrize(("wcsobj"), wno_unit_nd, indirect=True)
wcs_with_unit_1d = pytest.mark.parametrize(("wcsobj"), w_unit_1d, indirect=True)
wcs_with_unit_nd = pytest.mark.parametrize(("wcsobj"), w_unit_nd, indirect=True)


@wcs_no_unit_1d
def test_no_units_1d(wcsobj):
    """Transforms do not support units."""
    assert not wcsobj.forward_transform.uses_quantity

    # the case of a scalar input
    x = 1
    bbox = wcsobj.bounding_box
    if bbox is not None:
        x = np.mean(bbox.bounding_box())

    result_num = wcsobj(x)
    assert np.isscalar(result_num)

    assert_allclose(wcsobj.invert(result_num), x)

    xq = x * wcsobj.input_frame.unit[0]
    result = wcsobj(xq)
    assert_quantity_allclose(result, result_num * wcsobj.output_frame.unit[0])


@wcs_no_unit_nd
def test_no_units_nd(wcsobj):
    assert not wcsobj.forward_transform.uses_quantity

    n_inputs = wcsobj.input_frame.naxes

    inp = [1] * n_inputs
    bbox = wcsobj.bounding_box
    if bbox is not None:
        bb = bbox.bounding_box()
        inp = [np.mean(interval) for interval in bb]
    # Inputs are numbers
    result = wcsobj(*inp)
    assert is_numerical(result)
    if np.isscalar(result):
        result = [result]
    inp_new = wcsobj.invert(*result)
    _ = [assert_allclose(i, j) for i, j in zip(inp_new, inp, strict=True)]

    # Inputs are quantities; return quantities (except for pixels?)
    inpq = [coup * un for coup, un in zip(inp, wcsobj.input_frame.unit, strict=True)]
    result = wcsobj(*inpq)
    assert is_quantity(result)
    inp_new = wcsobj.invert(*result)
    _ = [assert_allclose(i, j) for i, j in zip(inp_new, inpq, strict=True)]

    sky = wcsobj.pixel_to_world(*inp)
    if not np.iterable(sky):
        sky = (sky,)
    with pytest.raises(
        TypeError, match=r"High Level objects are not supported with the native"
    ):
        wcsobj.invert(*sky)


@wcs_with_unit_1d
def test_with_units_1d(wcsobj):
    """Transform do not support units."""
    assert wcsobj.forward_transform.uses_quantity

    # the case of a scalar input
    x = 1 * wcsobj.input_frame.unit[0]

    result = wcsobj(x)
    assert isinstance(result, u.Quantity)
    assert_allclose(wcsobj.invert(result), x)

    x = 1
    result = wcsobj(x)
    assert np.isscalar(result)
    assert_allclose(wcsobj.invert(result), x)


@wcs_with_unit_nd
def test_transform_with_units(wcsobj):
    """Transforms support units."""
    assert wcsobj.forward_transform.uses_quantity

    n_inputs = wcsobj.input_frame.naxes
    xx = [x] * n_inputs

    # input is numerical, return numbers
    result_num = wcsobj(*xx)
    assert is_numerical(result_num)

    inp = wcsobj.invert(*result_num)
    assert is_numerical(inp)

    # input is quantities, return quantities
    xxq = [1 * u.pix] * n_inputs
    result = wcsobj(*xxq)
    assert all(type(res) is u.Quantity for res in result)
    assert_allclose([r.value for r in result], result_num)

    sky = wcsobj.pixel_to_world(*xxq)
    if not np.iterable(sky):
        sky = (sky,)
    with pytest.raises(
        TypeError, match=r"High Level objects are not supported with the native"
    ):
        wcsobj.invert(*sky)


@wcs_no_unit_1d
def test_add_units(wcsobj):
    if wcsobj.input_frame.naxes == 1:
        assert wcsobj.input_frame.add_units((1,)) == 1 * wcsobj.input_frame.unit[0]
        assert_allclose(
            wcsobj.input_frame.add_units(([1, 1],)),
            ([1, 1] * wcsobj.input_frame.unit[0],),
        )
    elif wcsobj.input_frame.naxes == 2:
        assert_quantity_allclose(
            wcsobj.input_frame.add_units((1, 1)), (1 * u.pix, 1 * u.pix)
        )
        assert_quantity_allclose(
            wcsobj.input_frame.add_units(([1, 1], [1, 1])),
            ([1, 1] * u.pix, [1, 1] * u.pix),
        )


@wcs_with_unit_1d
def test_remove_units(wcsobj):
    if wcsobj.input_frame.naxes == 1:
        unit = wcsobj.input_frame.unit[0]
        assert wcsobj.input_frame.remove_units(1 * unit) == (1,)
        assert_allclose(wcsobj.input_frame.remove_units(([1, 1] * unit,)), ([1, 1],))
    elif wcsobj.input_frame.naxes == 2:
        assert_quantity_allclose(
            wcsobj.input_frame.remove_units((1 * u.pix, 1 * u.pix)), (1, 1)
        )
        assert_quantity_allclose(
            wcsobj.input_frame.remove_units(([1, 1] * u.pix, [1, 1] * u.pix)),
            ([1, 1], [1, 1]),
        )


def test_transform_multistage_wcs(gwcs_with_pipeline_celestial):
    """
    Tests that the input and output types match for
    intermediate frames/transforms.
    """
    wcsobj = gwcs_with_pipeline_celestial
    frames = wcsobj.available_frames
    result = wcsobj.transform(frames[0], frames[-1], 1 * u.pix, 1 * u.pix)
    assert is_quantity(result)
    assert_allclose([r.value for r in result], wcsobj(1, 1))
    final_result = wcsobj.transform(frames[0], frames[-1], 1 * u.pix, 1 * u.pix)
    assert is_quantity(final_result)
    assert_allclose([r.value for r in final_result], wcsobj(1, 1))
    interm_result = wcsobj.transform(frames[0], frames[1], 1 * u.pix, 1 * u.pix)
    assert is_quantity(interm_result)
    tr = wcsobj.get_transform(frames[0], frames[1])
    assert_quantity_allclose(interm_result, tr(1 * u.pix, 1 * u.pix))
    ninterm_result = wcsobj.transform(frames[0], frames[1], 1, 1)
    assert_allclose([r.value for r in interm_result], ninterm_result)


def test_reverse_wcs_direction(gwcs_2d_spatial_shift_reverse):
    """Test that input quantities are converted to the units of the input frame."""
    wcsobj = gwcs_2d_spatial_shift_reverse
    assert_quantity_allclose(
        wcsobj(1 * u.arcsec, 2 * u.arcsec),
        wcsobj(1 * u.arcsec.to(u.deg) * u.deg, 2 * u.arcsec.to(u.deg) * u.deg),
    )


def test_transfrom_intermediate_1d(gwcs_multi_stage):
    wcsobj = gwcs_multi_stage
    assert wcsobj.transform("detector", "intermediate", 1) == 11.0
