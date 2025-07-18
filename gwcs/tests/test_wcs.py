# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from pathlib import Path

import asdf
import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import units as u
from astropy import wcs as astwcs
from astropy.io import fits
from astropy.modeling import bind_compound_bounding_box, models
from astropy.modeling.bounding_box import ModelBoundingBox
from astropy.time import Time
from astropy.utils.introspection import minversion
from astropy.wcs import wcsapi
from numpy.testing import assert_allclose, assert_equal

from gwcs import coordinate_frames as cf
from gwcs import wcs
from gwcs.examples import gwcs_2d_bad_bounding_box_order
from gwcs.tests import data
from gwcs.tests.utils import _gwcs_from_hst_fits_wcs
from gwcs.utils import CoordinateFrameError
from gwcs.wcstools import grid_from_bounding_box, wcs_from_fiducial, wcs_from_points

data_path = Path(data.__file__).parent.absolute()


m1 = models.Shift(12.4) & models.Shift(-2)
m2 = models.Scale(2) & models.Scale(-2)
m = m1 | m2

icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg))
detector = cf.Frame2D(name="detector", axes_order=(0, 1))
focal = cf.Frame2D(name="focal", axes_order=(0, 1), unit=(u.m, u.m))
spec = cf.SpectralFrame(
    name="wave",
    unit=[
        u.m,
    ],
    axes_order=(2,),
    axes_names=("lambda",),
)
time = cf.TemporalFrame(
    name="time",
    unit=[
        u.s,
    ],
    axes_order=(3,),
    axes_names=("time",),
    reference_frame=Time("2020-01-01"),
)
stokes = cf.StokesFrame(axes_order=(2,))

pipe = [wcs.Step(detector, m1), wcs.Step(focal, m2), wcs.Step(icrs, None)]
pipe_copy = pipe.copy()

# Create some data.
nx, ny = (5, 2)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)


def asdf_open_memory_mapping_kwarg(memmap: bool) -> dict:
    if minversion("asdf", "3.1.0"):
        return {"memmap": memmap}
    return {"copy_arrays": not memmap}


# Test initializing a WCS


def test_create_wcs():
    """
    Test initializing a WCS object.
    """
    # use only frame names
    gw1 = wcs.WCS(output_frame="icrs", input_frame="detector", forward_transform=m)
    # omit input_frame
    gw2 = wcs.WCS(output_frame="icrs", forward_transform=m)
    # use CoordinateFrame objects
    gw3 = wcs.WCS(output_frame=icrs, input_frame=detector, forward_transform=m)
    # use a pipeline to initialize
    pipe = [(detector, m1), (icrs, None)]
    gw4 = wcs.WCS(forward_transform=pipe)
    assert (
        gw1.available_frames
        == gw2.available_frames
        == gw3.available_frames
        == gw4.available_frames
        == ["detector", "icrs"]
    )
    res = m(1, 2)
    assert_allclose(gw1(1, 2), res)
    assert_allclose(gw2(1, 2), res)
    assert_allclose(gw3(1, 2), res)
    assert_allclose(gw3(1, 2), res)


def test_init_no_transform():
    """
    Test initializing a WCS object without a forward_transform.
    """
    gw = wcs.WCS(output_frame="icrs")
    assert len(gw._pipeline) == 2
    assert gw.pipeline[0].frame.name == "detector"
    with pytest.warns(
        DeprecationWarning, match="Indexing a WCS.pipeline step is deprecated."
    ):
        assert gw.pipeline[0][0].name == "detector"
    assert gw.pipeline[1].frame.name == "icrs"
    with pytest.warns(
        DeprecationWarning, match="Indexing a WCS.pipeline step is deprecated."
    ):
        assert gw.pipeline[1][0].name == "icrs"
    assert np.isin(gw.available_frames, ["detector", "icrs"]).all()
    gw = wcs.WCS(output_frame=icrs, input_frame=detector)
    assert gw._pipeline[0].frame.name == "detector"
    with pytest.warns(
        DeprecationWarning, match="Indexing a WCS.pipeline step is deprecated."
    ):
        assert gw._pipeline[0][0].name == "detector"
    assert gw._pipeline[1].frame.name == "icrs"
    with pytest.warns(
        DeprecationWarning, match="Indexing a WCS.pipeline step is deprecated."
    ):
        assert gw._pipeline[1][0].name == "icrs"
    assert np.isin(gw.available_frames, ["detector", "icrs"]).all()
    with pytest.raises(NotImplementedError):
        gw(1, 2)


def test_init_no_output_frame():
    """
    Test initializing a WCS without an output_frame raises an error.
    """
    with pytest.raises(CoordinateFrameError):
        wcs.WCS(forward_transform=m1)


def test_insert_transform():
    """Test inserting a transform."""
    gw = wcs.WCS(output_frame="icrs", forward_transform=m1)
    assert_allclose(gw.forward_transform(1, 2), m1(1, 2))
    gw.insert_transform(frame="icrs", transform=m2)
    assert_allclose(gw.forward_transform(1, 2), (m1 | m2)(1, 2))


def test_insert_frame():
    """Test inserting a frame into an existing pipeline"""
    w = wcs.WCS(pipe[:])
    original_result = w(1, 2)
    mnew = models.Shift(1) & models.Shift(1)
    new_frame = cf.Frame2D(name="new")

    # Insert at the beginning
    w.insert_frame(new_frame, mnew, w.input_frame)
    assert_allclose(w(0, 1), original_result)

    tr = w.get_transform("detector", w.output_frame)
    assert_allclose(tr(1, 2), original_result)

    # Insert at the end
    w = wcs.WCS(pipe[:])
    with pytest.raises(ValueError, match=r"New coordinate frame.*"):
        w.insert_frame("not a frame", mnew, new_frame)

    w.insert_frame("icrs", mnew, new_frame)
    assert_allclose([x - 1 for x in w(1, 2)], original_result)

    tr = w.get_transform("detector", "icrs")
    assert_allclose(tr(1, 2), original_result)

    # Force error by trying same operation
    with pytest.raises(ValueError, match=r".*both frames.*"):
        w.insert_frame("icrs", mnew, new_frame)


def test_set_transform():
    """Test setting a transform between two frames in the pipeline."""
    w = wcs.WCS(forward_transform=pipe[:])
    w.set_transform("detector", "focal", models.Identity(2))
    assert_allclose(w(1, 1), (2, -2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform("detector1", "focal", models.Identity(2))
    with pytest.raises(CoordinateFrameError):
        w.set_transform("detector", "focal1", models.Identity(2))


def test_get_transform():
    """Test getting a transform between two frames in the pipeline."""
    w = wcs.WCS(pipe[:])
    tr_forward = w.get_transform("detector", "focal")
    tr_back = w.get_transform("icrs", "detector")
    x, y = 1, 2
    fx, fy = tr_forward(1, 2)
    assert_allclose(w.pipeline[0].transform(x, y), (fx, fy))
    assert_allclose(w.pipeline[0].transform(x, y), (fx, fy))
    assert_allclose((x, y), tr_back(*w(x, y)))
    assert w.get_transform("detector", "detector") is None


def test_backward_transform():
    """
    Test backward transform raises an error when an analytical
    inverse is not available.
    """
    # Test that an error is raised when one of the models has not inverse.
    poly = models.Polynomial1D(1, c0=4)
    w = wcs.WCS(forward_transform=poly & models.Scale(2), output_frame="sky")
    with pytest.raises(NotImplementedError):
        _ = w.backward_transform

    # test backward transform
    poly.inverse = models.Shift(-4)
    w = wcs.WCS(forward_transform=poly & models.Scale(2), output_frame="sky")
    assert_allclose(w.backward_transform(1, 2), (-3, 1))


def test_backward_transform_has_inverse():
    """
    Test that backward transform has an inverse, which is the forward transform
    """
    poly = models.Polynomial1D(1, c0=4)
    poly.inverse = models.Polynomial1D(
        1, c0=-3
    )  # this is NOT the actual inverse of poly
    w = wcs.WCS(forward_transform=poly & models.Scale(2), output_frame="sky")
    assert_allclose(w.backward_transform.inverse(1, 2), w(1, 2))


def test_from_fiducial_sky():
    sky = coord.SkyCoord(1.63 * u.radian, -72.4 * u.deg, frame="fk5")
    tan = models.Pix2Sky_TAN()
    w = wcs_from_fiducial(sky, projection=tan)
    assert isinstance(w.CelestialFrame.reference_frame, coord.FK5)
    assert_allclose(w(0.1, 0.1), (93.7210280925364, -72.29972666307474))


def test_from_fiducial_composite():
    sky = coord.SkyCoord(1.63 * u.radian, -72.4 * u.deg, frame="fk5")
    tan = models.Pix2Sky_TAN()
    spec = cf.SpectralFrame(unit=(u.micron,), axes_order=(0,))
    celestial = cf.CelestialFrame(
        reference_frame=sky.frame,
        unit=(sky.spherical.lon.unit, sky.spherical.lat.unit),
        axes_order=(1, 2),
    )
    coord_frame = cf.CompositeFrame([spec, celestial], name="cube_frame")
    w = wcs_from_fiducial([0.5, sky], coord_frame, projection=tan)
    assert isinstance(w.cube_frame.frames[1].reference_frame, coord.FK5)
    assert_allclose(w(1, 1, 1), (1.5, 96.52373368309931, -71.37420187296995))
    # test returning coordinate objects with composite output_frame
    res = w.pixel_to_world(1, 2, 2)
    assert_allclose(res[0], u.Quantity(1.5 * u.micron))
    assert isinstance(res[1], coord.SkyCoord)
    assert_allclose(res[1].ra.value, 99.329496642319)
    assert_allclose(res[1].dec.value, -70.30322020351122)

    trans = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    w = wcs_from_fiducial([0.5, sky], coord_frame, projection=tan, transform=trans)
    assert_allclose(w(1, 1, 1), (11.5, 99.97738475762152, -72.29039139739766))
    # test coordinate object output

    coord_result = w.pixel_to_world(1, 1, 1)
    assert_allclose(coord_result[0], u.Quantity(11.5 * u.micron))


def test_from_fiducial_frame2d():
    fiducial = (34.5, 12.3)
    w = wcs_from_fiducial(fiducial, coordinate_frame=cf.Frame2D())
    assert w.output_frame.name == "Frame2D"
    assert_allclose(w(1, 1), (35.5, 13.3))


def test_bounding_box():
    trans3 = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    pipeline = [("detector", trans3), ("sky", None)]
    w = wcs.WCS(pipeline)
    bb = ((-1, 10), (6, 15))
    with pytest.raises(ValueError):  # noqa: PT011
        w.bounding_box = bb
    trans2 = models.Shift(10) & models.Scale(2)
    pipeline = [("detector", trans2), ("sky", None)]
    w = wcs.WCS(pipeline)
    w.bounding_box = bb
    assert w.bounding_box == w.forward_transform.bounding_box

    pipeline = [("detector", models.Shift(2)), ("sky", None)]
    w = wcs.WCS(pipeline)
    w.bounding_box = (1, 5)
    assert w.bounding_box == w.forward_transform.bounding_box
    with pytest.raises(ValueError):  # noqa: PT011
        w.bounding_box = ((1, 5), (2, 6))


def test_bounding_box_units():
    # Test that bounding_box with quantities can be assigned and evaluates
    bb = ((1 * u.pix, 5 * u.pix), (2 * u.pix, 6 * u.pix))
    trans = models.Shift(10 * u.pix) & models.Shift(2 * u.pix)
    pipeline = [("detector", trans), ("sky", None)]
    w = wcs.WCS(pipeline)
    w.bounding_box = bb
    world = w(-1 * u.pix, -1 * u.pix)
    assert_allclose(world, (np.nan, np.nan))


def test_compound_bounding_box():
    trans3 = models.Shift(10) & models.Scale(2) & models.Shift(-1)
    pipeline = [("detector", trans3), ("sky", None)]
    w = wcs.WCS(pipeline)
    cbb = {
        1: ((-1, 10), (6, 15)),
        2: ((-1, 5), (3, 17)),
        3: ((-3, 7), (1, 27)),
    }
    # Test attaching a valid bounding box (ignoring input 'x')
    w.attach_compound_bounding_box(cbb, [("x",)])
    from astropy.modeling.bounding_box import CompoundBoundingBox

    cbb = CompoundBoundingBox.validate(trans3, cbb, selector_args=[("x",)], order="F")
    assert w.bounding_box == cbb
    assert w.bounding_box is trans3.bounding_box

    # Test evaluating
    assert_allclose(w(13, 2, 1), (np.nan, np.nan, np.nan))
    assert_allclose(w(13, 2, 2), (np.nan, np.nan, np.nan))
    assert_allclose(w(13, 0, 3), (np.nan, np.nan, np.nan))
    # No bounding box for selector
    with pytest.raises(RuntimeError):
        w(13, 13, 4)

    # Test attaching a invalid bounding box (not ignoring input 'x')
    with pytest.raises(ValueError):  # noqa: PT011
        w.attach_compound_bounding_box(cbb, [("x", False)])

    # Test that bounding_box with quantities can be assigned and evaluates
    trans = models.Shift(10 * u.pix) & models.Shift(2 * u.pix)
    pipeline = [("detector", trans), ("sky", None)]
    w = wcs.WCS(pipeline)
    cbb = {1 * u.pix: (1 * u.pix, 5 * u.pix), 2 * u.pix: (2 * u.pix, 6 * u.pix)}
    w.attach_compound_bounding_box(cbb, [("x1",)])

    from astropy.modeling.bounding_box import CompoundBoundingBox

    cbb = CompoundBoundingBox.validate(trans, cbb, selector_args=[("x1",)], order="F")
    assert w.bounding_box == cbb
    assert w.bounding_box is trans.bounding_box

    assert_allclose(w(-1 * u.pix, 1 * u.pix), (np.nan, np.nan))
    assert_allclose(w(7 * u.pix, 2 * u.pix), (np.nan, np.nan))


def test_grid_from_bounding_box():
    bb = ((-1, 9.9), (6.5, 15))
    x, y = grid_from_bounding_box(bb, step=[0.1, 0.5], center=False)
    assert_allclose(x[:, 0], -1)
    assert_allclose(x[:, -1], 9.9)
    assert_allclose(y[0], 6.5)
    assert_allclose(y[-1], 15)


def test_grid_from_bounding_box_1d():
    # Test 1D case
    x = grid_from_bounding_box((-0.5, 4.5))
    assert_allclose(x, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_grid_from_bounding_box_step():
    bb = ((-0.5, 5.5), (-0.5, 4.5))
    x, y = grid_from_bounding_box(bb)
    x1, y1 = grid_from_bounding_box(bb, step=(1, 1))
    assert_allclose(x, x1)
    assert_allclose(y, y1)

    with pytest.raises(ValueError):  # noqa: PT011
        grid_from_bounding_box(bb, step=(1, 2, 1))


def test_grid_from_model_bounding_box():
    bbox = ((-1, 1), (0, 1))
    # Truth grid
    grid_truth = grid_from_bounding_box(bbox)

    # Create a bounding box
    model = models.Const2D() & models.Const1D()
    model.inputs = ("x", "y", "slit_name")
    model.bounding_box = ModelBoundingBox(
        {
            "x": bbox[0],
            "y": bbox[1],
        },
        model=model,
        ignored=["slit_name"],
        order="F",
    )
    grid = grid_from_bounding_box(model.bounding_box)

    assert np.all(grid == grid_truth)


def test_grid_from_compound_bounding_box():
    bbox = ((-1, 1), (0, 1))
    # Truth grid
    grid_truth = grid_from_bounding_box(bbox)

    # Create a compound bounding box
    model = models.Const2D() & models.Const1D()
    model.inputs = ("x", "y", "slit_name")
    bind_compound_bounding_box(
        model,
        {
            (200,): {
                "x": bbox[0],
                "y": bbox[1],
            },
            (300,): {
                "x": (-2, 2),
                "y": (0, 2),
            },
        },
        [("slit_name",)],
        order="F",
    )
    grid = grid_from_bounding_box(model.bounding_box, selector=(200,))

    assert np.all(grid == grid_truth)

    # Capture errors
    with pytest.raises(
        ValueError, match=r"Cannot use selector with a non-CompoundBoundingBox"
    ):
        grid_from_bounding_box(model.bounding_box[(300,)], selector=(300,))
    with pytest.raises(
        ValueError,
        match=r"selector must be set when bounding_box is a CompoundBoundingBox",
    ):
        grid_from_bounding_box(model.bounding_box)


def test_wcs_from_points():
    rng = np.random.default_rng(0)
    hdr = fits.Header.fromtextfile(data_path / "acs.hdr", endcard=False)
    with pytest.warns(astwcs.FITSFixedWarning) as caught_warnings:
        # this raises a warning unimportant for this testing the pix2world
        #   FITSFixedWarning(u'The WCS transformation has more axes (2) than
        #        the image it is associated with (0)')
        #   FITSFixedWarning: 'datfix' made the change
        #       'Set MJD-OBS to 53436.000000 from DATE-OBS'. [astropy.wcs.wcs]
        w = astwcs.WCS(hdr)
    assert len(caught_warnings) == 2
    y, x = np.mgrid[:2046:20j, :4023:10j]
    ra, dec = w.wcs_pix2world(x, y, 1)
    fiducial = coord.SkyCoord(ra.mean() * u.deg, dec.mean() * u.deg, frame="icrs")
    world_coords = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))
    w = wcs_from_points(xy=(x, y), world_coords=world_coords, proj_point=fiducial)
    newra, newdec = w(x, y)
    assert_allclose(newra, ra)
    assert_allclose(newdec, dec)

    w = wcs_from_points(xy=(x, y), world_coords=world_coords, proj_point="center")
    newra, newdec = w(x, y)
    assert_allclose(newra, ra)
    assert_allclose(newdec, dec)

    fiducial = fiducial.transform_to("galactic")
    w = wcs_from_points(xy=(x, y), world_coords=world_coords, proj_point=fiducial)
    newra, newdec = w(x, y)
    assert_allclose(newra, ra)
    assert_allclose(newdec, dec)

    n = rng.standard_normal(ra.size)
    n.shape = ra.shape
    nra = n * 10**-2
    ndec = n * 10**-2
    w = wcs_from_points(
        xy=(x + nra, y + ndec), world_coords=world_coords, proj_point=fiducial
    )
    newra, newdec = w(x, y)
    assert_allclose(newra, ra, atol=10**-6)
    assert_allclose(newdec, dec, atol=10**-6)

    newra, newdec = w.pixel_to_world_values(x, y)
    assert_allclose(newra, ra, atol=10**-6)
    assert_allclose(newdec, dec, atol=10**-6)

    newsky = w.pixel_to_world(x, y)
    assert_allclose(newsky.data.lon.deg, ra, atol=10**-6)
    assert_allclose(newsky.data.lat.deg, dec, atol=10**-6)


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
    pipeline = [
        (
            cf.CoordinateFrame(
                naxes=1, axes_type=("PIXEL",), axes_order=(0,), name="detector"
            ),
            trans3,
        ),
        (
            cf.CoordinateFrame(
                naxes=1, axes_type=("SPATIAL",), axes_order=(0,), name="sky"
            ),
            None,
        ),
    ]
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
    assert_allclose(
        w([1, 1], [7, 7], [3, 5]), [[np.nan, 11], [np.nan, 14], [np.nan, 4]]
    )

    # test ``transform`` method
    assert_allclose(w.transform("detector", "sky", 1, 7, 3), [np.nan, np.nan, np.nan])


def test_format_output():
    points = np.arange(5)
    values = np.array([1.5, 3.4, 6.7, 7, 32])
    t = models.Tabular1D(points, values)
    pipe = [("detector", t), ("world", None)]
    w = wcs.WCS(pipe)
    assert_allclose(w(1), 3.4)
    assert_allclose(w([1, 2]), [3.4, 6.7])
    assert np.isscalar(w(1))


def test_available_frames():
    w = wcs.WCS(pipe)
    assert w.available_frames == ["detector", "focal", "icrs"]


def test_footprint():
    icrs = cf.CelestialFrame(
        name="icrs", reference_frame=coord.ICRS(), axes_order=(0, 1)
    )
    spec = cf.SpectralFrame(
        name="freq",
        unit=[
            u.Hz,
        ],
        axes_order=(2,),
    )
    world = cf.CompositeFrame([icrs, spec])
    transform = (models.Shift(10) & models.Shift(-1)) & models.Scale(2)
    pipe = [("det", transform), (world, None)]
    w = wcs.WCS(pipe)

    with pytest.raises(TypeError):
        w.footprint()

    w.bounding_box = ((1, 5), (1, 3), (1, 6))

    assert_equal(
        w.footprint(),
        np.array(
            [
                [11, 0, 2],
                [11, 0, 12],
                [11, 2, 2],
                [11, 2, 12],
                [15, 0, 2],
                [15, 0, 12],
                [15, 2, 2],
                [15, 2, 12],
            ]
        ),
    )
    assert_equal(
        w.footprint(axis_type="spatial"),
        np.array([[11.0, 0.0], [11.0, 2.0], [15.0, 2.0], [15.0, 0.0]]),
    )

    assert_equal(w.footprint(axis_type="spectral"), np.array([2, 12]))


def test_outside_footprint_inputs(gwcs_2d_spatial_shift):
    """
    Regression test for #594
        -> When inverting the WCS, the footprint will modify the input coordinate
           variables, which is not expected.
    """
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    gwcs_2d_spatial_shift.bounding_box = ((0, 10), (0, 10))

    _ = gwcs_2d_spatial_shift.world_to_pixel_values(x, y)
    assert (x == np.linspace(-20, 20, 100)).all()
    assert (y == np.linspace(-20, 20, 100)).all()


def test_high_level_api():
    """
    Test WCS high level API.
    """
    output_frame = cf.CompositeFrame(frames=[icrs, spec, time])
    transform = m1 & models.Scale(1.5) & models.Scale(2)
    det = cf.CoordinateFrame(
        naxes=4,
        unit=(u.pix, u.pix, u.pix, u.pix),
        axes_order=(0, 1, 2, 3),
        axes_type=("length", "length", "length", "length"),
    )
    w = wcs.WCS(forward_transform=transform, output_frame=output_frame, input_frame=det)
    wrapped = wcsapi.HighLevelWCSWrapper(w)

    r, d, lam, t = w(xv, yv, xv, xv)
    world_coord = w.pixel_to_world(xv, yv, xv, xv)
    assert isinstance(world_coord[0], coord.SkyCoord)
    assert isinstance(world_coord[1], u.Quantity)
    assert isinstance(world_coord[2], Time)
    assert_allclose(world_coord[0].data.lon.value, r)
    assert_allclose(world_coord[0].data.lat.value, d)
    assert_allclose(world_coord[1].value, lam)
    assert_allclose((world_coord[2] - time.reference_frame).to(u.s).value, t)

    wrapped_world_coord = wrapped.pixel_to_world(xv, yv, xv, xv)
    assert_allclose(wrapped_world_coord[0].data.lon.value, r)
    assert_allclose(wrapped_world_coord[0].data.lat.value, d)
    assert_allclose(wrapped_world_coord[1].value, lam)
    assert_allclose((world_coord[2] - time.reference_frame).to(u.s).value, t)

    x1, y1, z1, k1 = w.world_to_pixel(*world_coord)
    assert_allclose(x1, xv)
    assert_allclose(y1, yv)
    assert_allclose(z1, xv)
    assert_allclose(k1, xv)

    x1, y1, z1, k1 = wrapped.world_to_pixel(*world_coord)
    assert_allclose(x1, xv)
    assert_allclose(y1, yv)
    assert_allclose(z1, xv)
    assert_allclose(k1, xv)


class TestImaging:
    def setup_class(self):
        hdr = fits.Header.fromtextfile(data_path / "acs.hdr", endcard=False)
        with pytest.warns(astwcs.FITSFixedWarning) as caught_warnings:
            # this raises a warning unimportant for this testing the pix2world
            #   FITSFixedWarning(u'The WCS transformation has more axes (2) than
            #        the image it is associated with (0)')
            #   FITSFixedWarning: 'datfix' made the change
            #       'Set MJD-OBS to 53436.000000 from DATE-OBS'. [astropy.wcs.wcs]
            self.fitsw = astwcs.WCS(hdr)
        assert len(caught_warnings) == 2
        a_coeff = hdr["A_*"]
        a_order = a_coeff.pop("A_ORDER")
        b_coeff = hdr["B_*"]
        b_order = b_coeff.pop("B_ORDER")

        crpix = [hdr["CRPIX1"], hdr["CRPIX2"]]
        distortion = models.SIP(
            crpix, a_order, b_order, a_coeff, b_coeff, name="sip_distorion"
        ) + models.Identity(2)

        cdmat = np.array([[hdr["CD1_1"], hdr["CD1_2"]], [hdr["CD2_1"], hdr["CD2_2"]]])
        aff = models.AffineTransformation2D(matrix=cdmat, name="rotation")

        offx = models.Shift(-hdr["CRPIX1"], name="x_translation")
        offy = models.Shift(-hdr["CRPIX2"], name="y_translation")

        wcslin = (offx & offy) | aff

        phi = hdr["CRVAL1"]
        lon = hdr["CRVAL2"]
        theta = 180
        n2c = models.RotateNative2Celestial(phi, lon, theta, name="sky_rotation")

        tan = models.Pix2Sky_TAN(name="tangent_projection")
        sky_cs = cf.CelestialFrame(reference_frame=coord.ICRS(), name="sky")
        det = cf.Frame2D(name="detector")
        focal = cf.Frame2D(name="focal")
        wcs_forward = wcslin | tan | n2c
        pipeline = [
            wcs.Step(det, distortion),
            wcs.Step(focal, wcs_forward),
            wcs.Step(sky_cs, None),
        ]

        self.wcs = wcs.WCS(
            input_frame=det, output_frame=sky_cs, forward_transform=pipeline
        )

        self.xv, self.yv = xv, yv

    def test_distortion(self):
        sipx, sipy = self.fitsw.sip_pix2foc(self.xv, self.yv, 1)
        sipx = np.array(sipx) + 2048
        sipy = np.array(sipy) + 1024
        sip_coord = self.wcs.get_transform("detector", "focal")(self.xv, self.yv)
        assert_allclose(sipx, sip_coord[0])
        assert_allclose(sipy, sip_coord[1])

    def test_wcslinear(self):
        ra, dec = self.fitsw.wcs_pix2world(self.xv, self.yv, 1)
        sky = self.wcs.get_transform("focal", "sky")(self.xv, self.yv)
        assert_allclose(ra, sky[0])
        assert_allclose(dec, sky[1])

    def test_forward(self):
        sky_coord = self.wcs(self.xv, self.yv)
        ra, dec = self.fitsw.all_pix2world(self.xv, self.yv, 1)
        assert_allclose(sky_coord[0], ra)
        assert_allclose(sky_coord[1], dec)

    def test_backward(self):
        transform = self.wcs.get_transform(
            from_frame="focal", to_frame=self.wcs.output_frame
        )
        sky_coord = self.wcs.transform("focal", self.wcs.output_frame, self.xv, self.yv)
        px_coord = transform.inverse(*sky_coord)
        assert_allclose(px_coord[0], self.xv, atol=10**-6)
        assert_allclose(px_coord[1], self.yv, atol=10**-6)

    def test_footprint(self):
        bb = ((1, 4096), (1, 2048))
        footprint = self.wcs.footprint(bb)
        fits_footprint = self.fitsw.calc_footprint(axes=(4096, 2048))
        assert_allclose(footprint, fits_footprint)

    def test_inverse(self):
        sky_coord = self.wcs(10, 20, with_units=True)
        assert np.allclose(self.wcs.invert(sky_coord), (10, 20))

    def test_back_coordinates(self):
        sky_coord = self.wcs(1, 2, with_units=True)
        res = self.wcs.transform("sky", "focal", sky_coord, with_units=False)
        assert_allclose(res, self.wcs.get_transform("detector", "focal")(1, 2))

    def test_units(self):
        assert self.wcs.unit == (u.degree, u.degree)

    def test_get_transform(self):
        with pytest.raises(CoordinateFrameError):
            assert (
                self.wcs.get_transform("x_translation", "sky_rotation").submodel_names
                == self.wcs.forward_transform[1:].submodel_names
            )

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
    fn = data_path / "miriwcs.asdf"
    with asdf.open(
        fn,
        lazy_load=False,
        ignore_missing_extensions=True,
        **asdf_open_memory_mapping_kwarg(memmap=False),
    ) as af:
        miriwcs = af.tree["wcs"]
    bounding_box = ((0, 1024), (0, 1024))
    mirisip = miriwcs.to_fits_sip(bounding_box, max_inv_pix_error=0.1, verbose=True)
    fitssip = astwcs.WCS(mirisip)
    fitsvalx, fitsvaly = fitssip.all_pix2world(xflat + 1, yflat + 1, 1)
    gwcsvalx, gwcsvaly = miriwcs(xflat, yflat)
    assert_allclose(gwcsvalx, fitsvalx, atol=1e-10, rtol=0)
    assert_allclose(gwcsvaly, fitsvaly, atol=1e-10, rtol=0)
    fits_inverse_valx, fits_inverse_valy = fitssip.all_world2pix(fitsvalx, fitsvaly, 1)
    assert_allclose(xflat, fits_inverse_valx - 1, atol=0.1, rtol=0)
    assert_allclose(yflat, fits_inverse_valy - 1, atol=0.1, rtol=0)

    mirisip = miriwcs.to_fits_sip(bounding_box=None, max_inv_pix_error=0.1)
    fitssip = astwcs.WCS(mirisip)
    fitsvalx, fitsvaly = fitssip.all_pix2world(xflat + 1, yflat + 1, 1)
    assert_allclose(gwcsvalx, fitsvalx, atol=4e-11, rtol=0)
    assert_allclose(gwcsvaly, fitsvaly, atol=4e-11, rtol=0)

    miriwcs.bounding_box = None
    with pytest.raises(ValueError):  # noqa: PT011
        _ = miriwcs.to_fits_sip(bounding_box=None, max_inv_pix_error=0.1)


@pytest.mark.parametrize(
    "matrix_type", ["CD", "PC-CDELT1", "PC-SUM1", "PC-DET1", "PC-SCALE"]
)
def test_to_fits_sip_pc_normalization(gwcs_simple_imaging_units, matrix_type):
    y, x = np.mgrid[:1024:10, :1024:10]
    xflat = np.ravel(x[1:-1, 1:-1])
    yflat = np.ravel(y[1:-1, 1:-1])
    bounding_box = ((0, 1024), (0, 1024))

    # create a simple imaging WCS without distortions:
    cdmat = np.array([[1.29e-5, 5.95e-6], [5.02e-6, -1.26e-5]])
    aff = models.AffineTransformation2D(matrix=cdmat, name="rotation")

    offx = models.Shift(-501, name="x_translation")
    offy = models.Shift(-501, name="y_translation")

    wcslin = (offx & offy) | aff

    n2c = models.RotateNative2Celestial(5.63, -72.05, 180, name="sky_rotation")
    tan = models.Pix2Sky_TAN(name="tangent_projection")

    wcs_forward = wcslin | tan | n2c

    sky_cs = cf.CelestialFrame(reference_frame=coord.ICRS(), name="sky")
    pipeline = [("detector", wcs_forward), (sky_cs, None)]

    wcs_lin = wcs.WCS(
        input_frame=cf.Frame2D(name="detector"),
        output_frame=sky_cs,
        forward_transform=pipeline,
    )

    _, _, celestial_group = wcs_lin._separable_groups(detect_celestial=True)
    fits_wcs = wcs_lin._to_fits_sip(
        celestial_group=celestial_group,
        keep_axis_position=False,
        bounding_box=bounding_box,
        max_pix_error=0.1,
        degree=None,
        max_inv_pix_error=0.1,
        inv_degree=None,
        npoints=32,
        crpix=None,
        projection="TAN",
        matrix_type=matrix_type,
        verbose=True,
    )
    fitssip = astwcs.WCS(fits_wcs)

    fitsvalx, fitsvaly = fitssip.wcs_pix2world(xflat, yflat, 0)
    inv_fitsvalx, inv_fitsvaly = fitssip.wcs_world2pix(fitsvalx, fitsvaly, 0)
    gwcsvalx, gwcsvaly = wcs_lin(xflat, yflat)

    assert_allclose(gwcsvalx, fitsvalx, atol=4e-11, rtol=0)
    assert_allclose(gwcsvaly, fitsvaly, atol=4e-11, rtol=0)

    assert_allclose(xflat, inv_fitsvalx, atol=5e-9, rtol=0)
    assert_allclose(yflat, inv_fitsvaly, atol=5e-9, rtol=0)


def test_to_fits_sip_composite_frame(gwcs_cube_with_separable_spectral):
    w, axes_order = gwcs_cube_with_separable_spectral

    dec_axis = int(axes_order.index(1) > axes_order.index(0)) + 1
    ra_axis = 3 - dec_axis

    fw_hdr = w.to_fits_sip()
    assert fw_hdr[f"CTYPE{dec_axis}"] == "DEC--TAN"
    assert fw_hdr[f"CTYPE{ra_axis}"] == "RA---TAN"
    assert fw_hdr["WCSAXES"] == 2
    assert fw_hdr["NAXIS"] == 2
    assert fw_hdr["NAXIS1"] == 128
    assert fw_hdr["NAXIS2"] == 64

    fw = astwcs.WCS(fw_hdr)
    gskyval = w.pixel_to_world(1, 60, 55)[1]
    fskyval = fw.all_pix2world(1, 60, 0)
    fskyval = [float(fskyval[ra_axis - 1]), float(fskyval[dec_axis - 1])]
    assert np.allclose([gskyval.ra.value, gskyval.dec.value], fskyval)


def test_to_fits_sip_composite_frame_galactic(gwcs_3d_galactic_spectral):
    w = gwcs_3d_galactic_spectral

    fw_hdr = w.to_fits_sip()
    assert fw_hdr["CTYPE1"] == "GLAT-TAN"

    fw = astwcs.WCS(fw_hdr)
    gskyval = w.pixel_to_world(7, 8, 9)[0]
    assert np.allclose(
        [gskyval.b.value, gskyval.l.value], fw.all_pix2world(7, 9, 0), atol=1e-3
    )


def test_to_fits_sip_composite_frame_keep_axis(gwcs_cube_with_separable_spectral):
    from inspect import Parameter, signature

    w, axes_order = gwcs_cube_with_separable_spectral
    _, _, celestial_group = w._separable_groups(detect_celestial=True)

    pars = signature(w.to_fits_sip).parameters
    kwargs = {k: v.default for k, v in pars.items() if v.default is not Parameter.empty}
    kwargs["matrix_type"] = "CD"

    fw_hdr = w._to_fits_sip(
        celestial_group=celestial_group, keep_axis_position=True, **kwargs
    )

    ra_axis = axes_order.index(0) + 1
    dec_axis = axes_order.index(1) + 1

    fw_hdr["CD1_3"] = 1
    fw_hdr["CRPIX3"] = 1

    assert fw_hdr[f"CTYPE{dec_axis}"] == "DEC--TAN"
    assert fw_hdr[f"CTYPE{ra_axis}"] == "RA---TAN"
    assert fw_hdr["WCSAXES"] == 2

    with pytest.warns(
        astwcs.FITSFixedWarning, match="The WCS transformation has more axes"
    ):
        # this raises a warning unimportant for this testing the pix2world
        #   FITSFixedWarning(u'The WCS transformation has more axes (3) than
        #        the image it is associated with (2)')
        fw = astwcs.WCS(fw_hdr)
    gskyval = w(1, 45, 55)[1:]
    assert np.allclose(gskyval, fw.all_pix2world([[1, 45, 55]], 0)[0][1:])


def test_to_fits_tab_no_bb(gwcs_3d_galactic_spectral):
    # gWCS:
    w = gwcs_3d_galactic_spectral
    w.bounding_box = None

    # FITS WCS -TAB:
    with pytest.raises(ValueError):  # noqa: PT011
        _, _ = w.to_fits_tab()


def test_to_fits_tab_cube(gwcs_3d_galactic_spectral):
    # gWCS:
    w = gwcs_3d_galactic_spectral

    # FITS WCS -TAB:
    hdr, bt = w.to_fits_tab()
    hdulist = fits.HDUList([fits.PrimaryHDU(np.ones(w.pixel_n_dim * (2,)), hdr), bt])
    fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    hdr, bt = w.to_fits_tab(bounding_box=w.bounding_box)
    hdulist = fits.HDUList([fits.PrimaryHDU(np.ones(w.pixel_n_dim * (2,)), hdr), bt])
    fits_wcs_user_bb = astwcs.WCS(hdulist[0].header, hdulist)

    # test points:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = w.bounding_box
    rng = np.random.default_rng(1)
    x = xmin + (xmax - xmin) * rng.random(100)
    y = ymin + (ymax - ymin) * rng.random(100)
    z = zmin + (zmax - zmin) * rng.random(100)

    # test:
    assert np.allclose(
        w(x, y, z), fits_wcs.wcs_pix2world(x, y, z, 0), rtol=1e-6, atol=1e-7
    )

    assert np.allclose(
        w(x, y, z), fits_wcs_user_bb.wcs_pix2world(x, y, z, 0), rtol=1e-6, atol=1e-7
    )


@pytest.mark.filterwarnings("ignore:.*The WCS transformation has more axes.*")
def test_to_fits_tab_7d(gwcs_7d_complex_mapping):
    # gWCS:
    w = gwcs_7d_complex_mapping

    # create FITS headers and -TAB headers
    hdr, bt = w.to_fits(projection="TAN")

    # create FITS WCS object:
    hdus = [fits.PrimaryHDU(np.zeros(w.array_shape), hdr)]
    hdus.extend(bt)
    hdulist = fits.HDUList(hdus)
    fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    # test points:
    rng = np.random.default_rng(1)
    npts = 100
    pts = np.zeros((len(w.bounding_box) + 1, npts))
    for k in range(len(w.bounding_box)):
        xmin, xmax = w.bounding_box[k]
        pts[k, :] = xmin + (xmax - xmin) * rng.random(npts)

    world_crds = w(*pts[:-1, :])

    # test forward transformation:
    assert np.allclose(world_crds, fits_wcs.wcs_pix2world(*pts, 0))

    # test round-tripping:
    assert np.allclose(pts, fits_wcs.wcs_world2pix(*world_crds, 0))


@pytest.mark.skip(reason="Fails round-trip for -TAB axis 4")
def test_to_fits_mixed_4d(gwcs_spec_cel_time_4d):
    # gWCS:
    w = gwcs_spec_cel_time_4d

    # create FITS headers and -TAB headers
    hdr, bt = w.to_fits()

    # create FITS WCS object:
    hdus = [fits.PrimaryHDU(np.zeros(w.array_shape), hdr)]
    hdus.extend(bt)
    hdulist = fits.HDUList(hdus)
    fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    # test points:
    rng = np.random.default_rng(1)
    npts = 100
    pts = np.zeros((len(w.bounding_box), npts))
    for k in range(len(w.bounding_box)):
        xmin, xmax = w.bounding_box[k]
        pts[k, :] = xmin + (xmax - xmin) * rng.random(npts)

    world_crds = w(*pts)

    # test forward transformation:
    assert np.allclose(world_crds, fits_wcs.wcs_pix2world(*pts, 0))

    # test round-tripping:
    pts2 = np.array(fits_wcs.wcs_world2pix(*world_crds, 0))
    assert np.allclose(pts, pts2, rtol=1e-5, atol=1e-5)


def test_to_fits_no_sip_used(gwcs_spec_cel_time_4d):
    # gWCS:
    w = gwcs_spec_cel_time_4d

    # create FITS headers and -TAB headers
    with pytest.warns(
        UserWarning, match="SIP distortion is not supported when the number"
    ):
        # UserWarning: SIP distortion is not supported when the number
        # of axes in WCS is larger than 2. Setting 'degree'
        # to 1 and 'max_inv_pix_error' to None.
        hdr, _ = w.to_fits(degree=3)

    # check that FITS WCS is not using SIP
    assert not hdr["?_ORDER"]
    assert not hdr["?P_ORDER"]
    assert not hdr["A_?_?"]
    assert not hdr["B_?_?"]
    assert not any(s.endswith("-SIP") for s in hdr["CTYPE?"].values())


def test_to_fits_1D_round_trip(gwcs_1d_spectral):
    # gWCS:
    w = gwcs_1d_spectral

    # FITS WCS -SIP (for celestial) and -TAB (for spectral):
    hdr, bt = w.to_fits()
    hdulist = fits.HDUList([fits.PrimaryHDU(np.ones(w.array_shape), hdr), bt[0]])
    fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    # test points:
    rng = np.random.default_rng(1)
    (xmin, xmax) = w.bounding_box.bounding_box()
    x = xmin + (xmax - xmin) * rng.random(100)

    # test forward transformation:
    wt = fits_wcs.wcs_pix2world(x, 0)
    assert np.allclose(w(x), wt, rtol=1e-6, atol=1e-7)

    # test inverse (round-trip):
    xinv = fits_wcs.wcs_world2pix(wt[0], 0)[0]
    assert np.allclose(x, xinv, rtol=1e-6, atol=1e-7)


def test_to_fits_sip_tab_cube(gwcs_cube_with_separable_spectral):
    # gWCS:
    w, axes_order = gwcs_cube_with_separable_spectral

    # FITS WCS -SIP (for celestial) and -TAB (for spectral):
    hdr, bt = w.to_fits(projection=models.Sky2Pix_TAN(name="TAN"))

    # create FITS WCS object:
    hdus = [fits.PrimaryHDU(np.zeros(w.array_shape), hdr)]
    hdus.extend(bt)
    hdulist = fits.HDUList(hdus)
    fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    # test points:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = w.bounding_box
    rng = np.random.default_rng(1)
    x = xmin + (xmax - xmin) * rng.random(100)
    y = ymin + (ymax - ymin) * rng.random(100)
    z = zmin + (zmax - zmin) * rng.random(100)

    world_crds = w(x, y, z)

    # test forward transformation:
    assert np.allclose(world_crds, fits_wcs.wcs_pix2world(x, y, z, 0))

    # test round-tripping:
    assert np.allclose((x, y, z), fits_wcs.wcs_world2pix(*world_crds, 0))


def test_to_fits_tab_time_cube(gwcs_cube_with_separable_time):
    # gWCS:
    w = gwcs_cube_with_separable_time

    # FITS WCS -SIP (for celestial) and -TAB (for spectral):
    hdr, bt = w.to_fits(projection=models.Sky2Pix_TAN(name="TAN"))

    # create FITS WCS object:
    hdus = [fits.PrimaryHDU(np.zeros(w.array_shape), hdr)]
    hdus.extend(bt)
    hdulist = fits.HDUList(hdus)
    fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    assert np.allclose(hdulist[1].data["coordinates"].ravel(), np.arange(128))

    # test points:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = w.bounding_box
    rng = np.random.default_rng(1)
    x = xmin + (xmax - xmin) * rng.random(5)
    y = ymin + (ymax - ymin) * rng.random(5)
    z = zmin + (zmax - zmin) * rng.random(5)

    world_crds = w(x, y, z)

    # test forward transformation:
    assert np.allclose(world_crds, fits_wcs.wcs_pix2world(x, y, z, 0))

    # test round-tripping:
    assert np.allclose(
        (x, y, z), fits_wcs.wcs_world2pix(*world_crds, 0), rtol=1e-5, atol=1e-5
    )


def test_to_fits_tab_miri_image():
    # gWCS:
    fn = data_path / "miriwcs.asdf"
    with asdf.open(
        fn,
        lazy_load=False,
        ignore_missing_extensions=True,
        **asdf_open_memory_mapping_kwarg(memmap=False),
    ) as af:
        w = af.tree["wcs"]

    # FITS WCS -TAB:
    hdr, bt = w.to_fits_tab(sampling=0.5)
    hdulist = fits.HDUList([fits.PrimaryHDU(np.ones(w.pixel_n_dim * (2,)), hdr), bt])

    fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    # test points:
    (xmin, xmax), (ymin, ymax) = w.bounding_box
    rng = np.random.default_rng(1)
    x = xmin + (xmax - xmin) * rng.random(100)
    y = ymin + (ymax - ymin) * rng.random(100)

    # test:
    assert np.allclose(w(x, y), fits_wcs.wcs_pix2world(x, y, 0), rtol=1e-6, atol=1e-7)


def test_to_fits_tab_miri_lrs():
    fn = data_path / "miri_lrs_wcs.asdf"
    with asdf.open(
        fn,
        lazy_load=False,
        ignore_missing_extensions=True,
        **asdf_open_memory_mapping_kwarg(memmap=False),
    ) as af:
        w = af.tree["wcs"]

    # FITS WCS -TAB:
    hdr, bt = w.to_fits(sampling=0.25)
    hdulist = fits.HDUList([fits.PrimaryHDU(np.ones(w.pixel_n_dim * (2,)), hdr), bt[0]])
    with pytest.warns(
        astwcs.FITSFixedWarning, match="The WCS transformation has more axes"
    ):
        # this raises a warning unimportant for this testing the pix2world
        #   FITSFixedWarning(u'The WCS transformation has more axes (3) than
        #        the image it is associated with (2)')
        fits_wcs = astwcs.WCS(hdulist[0].header, hdulist)

    # test points:
    (xmin, xmax), (ymin, ymax) = w.bounding_box
    rng = np.random.default_rng(1)
    x = xmin + (xmax - xmin) * rng.random(100)
    y = ymin + (ymax - ymin) * rng.random(100)

    # test:
    ref = np.array(w(x, y))
    tab = np.array(fits_wcs.wcs_pix2world(x, y, 0, 0))
    m = np.cumprod(np.isfinite(ref), dtype=np.bool_, axis=0)

    assert hdr["WCSAXES"] == 3
    assert np.allclose(ref[m], tab[m], rtol=5e-5, atol=5e-6, equal_nan=True)


def test_in_image():
    # create a 1-dim WCS:
    w1 = wcs.WCS(
        [
            (
                cf.SpectralFrame(name="input", axes_names=("x",), unit=(u.pix,)),
                models.Scale(2),
            ),
            (cf.SpectralFrame(name="output", axes_names=("x"), unit=(u.pix,)), None),
        ]
    )
    w1.bounding_box = (1, 5)

    assert np.isscalar(w1.in_image(4))
    assert w1.in_image(4)
    assert not w1.in_image(14)
    assert np.array_equal(
        w1.in_image([[-1, 4, 11], [2, 3, 12]]),
        [[False, True, False], [True, True, False]],
    )

    # create a 2-dim WCS:
    w2 = wcs.WCS(
        [
            (
                cf.Frame2D(name="input", axes_names=("x", "y"), unit=(u.pix, u.pix)),
                models.Scale(2) & models.Scale(1.5),
            ),
            (
                cf.Frame2D(name="output", axes_names=("x", "y"), unit=(u.pix, u.pix)),
                None,
            ),
        ]
    )
    w2.bounding_box = [(1, 100), (2, 20)]

    assert np.isscalar(w2.in_image(2, 6))
    assert not np.isscalar(w2.in_image([2], [6]))
    assert w2.in_image(4, 6)
    assert not (w2.in_image(5, 0))
    assert np.array_equal(
        w2.in_image(
            [[9, 10, 11, 15], [8, 9, 67, 98], [2, 2, np.nan, 102]],
            [[9, np.nan, 11, 15], [8, 9, 67, 98], [1, 1, np.nan, -10]],
        ),
        [
            [True, False, True, True],
            [True, True, False, False],
            [False, False, False, False],
        ],
    )


def test_iter_inv():
    fn = data_path / "nircamwcs.asdf"
    with asdf.open(
        fn,
        lazy_load=False,
        ignore_missing_extensions=True,
        **asdf_open_memory_mapping_kwarg(memmap=True),
    ) as af:
        w = af.tree["wcs"]
    # remove analytic/user-supplied inverse:
    w.pipeline[0].transform.inverse = None
    w.bounding_box = None

    # test single point
    assert np.allclose((1, 2), w.invert(*w(1, 2)))
    assert np.allclose(
        (np.nan, np.nan), w.numerical_inverse(*w(np.nan, 2)), equal_nan=True
    )

    # prepare to test a vector of points:
    rng = np.random.default_rng(10)
    x, y = 2047 * rng.random((2, 10000))  # "truth"

    # test adaptive:
    xp, yp = w.invert(
        *w(x, y),
        adaptive=True,
        detect_divergence=True,
        tolerance=1e-4,
        maxiter=50,
        quiet=False,
    )
    assert np.allclose((x, y), (xp, yp))

    with asdf.open(
        fn,
        lazy_load=False,
        ignore_missing_extensions=True,
        **asdf_open_memory_mapping_kwarg(memmap=True),
    ) as af:
        w = af.tree["wcs"]

    # test single point
    assert np.allclose((1, 2), w.numerical_inverse(*w(1, 2)))
    assert np.allclose(
        (np.nan, np.nan), w.numerical_inverse(*w(np.nan, 2)), equal_nan=True
    )

    # don't detect devergence
    xp, yp = w.numerical_inverse(
        *w(x, y),
        adaptive=True,
        tolerance=1e-5,
        maxiter=50,
        detect_divergence=False,
        quiet=False,
    )
    assert np.allclose((x, y), (xp, yp))

    with pytest.raises(wcs.NoConvergence) as e:
        w.numerical_inverse(
            *w([1, 20, 200, 2000], [200, 1000, 2000, 5]),
            adaptive=True,
            detect_divergence=True,
            maxiter=2,  # force not reaching requested accuracy
            quiet=False,
        )

    xp, yp = e.value.best_solution.T
    assert e.value.slow_conv.size == 4
    assert np.all(np.sort(e.value.slow_conv) == np.arange(4))

    # test non-adaptive:
    xp, yp = w.numerical_inverse(
        *w(x, y, with_bounding_box=False),
        adaptive=False,
        detect_divergence=True,
        quiet=False,
        with_bounding_box=False,
    )
    assert np.allclose((x, y), (xp, yp))

    # test non-adaptive:
    x[0] = 3000
    y[0] = 10000
    xp, yp = w.numerical_inverse(
        *w(x, y, with_bounding_box=False),
        adaptive=False,
        tolerance=1e-5,
        maxiter=50,
        detect_divergence=True,
        quiet=False,
        with_bounding_box=False,
    )
    assert np.allclose((x, y), (xp, yp))

    # test non-adaptive with non-recoverable divergence:
    x[0] = 300000
    y[0] = 1000000
    with pytest.raises(wcs.NoConvergence) as e:
        xp, yp = w.numerical_inverse(
            *w(x, y, with_bounding_box=False),
            adaptive=False,
            tolerance=1e-5,
            maxiter=50,
            detect_divergence=True,
            quiet=False,
            with_bounding_box=False,
        )

    xp, yp = e.value.best_solution.T
    assert np.allclose((x[1:], y[1:]), (xp[1:], yp[1:]))
    assert e.value.divergent[0] == 0


def test_tabular_2d_quantity():
    shape = (3, 3)
    data = np.arange(np.prod(shape)).reshape(shape) * u.m / u.s

    # The integer location is at the centre of the pixel.
    points_unit = u.pix
    points = [(np.arange(size) - 0) * points_unit for size in shape]

    kwargs = {
        "bounds_error": False,
        "fill_value": np.nan,
        "method": "nearest",
    }

    forward = models.Tabular2D(points, data, **kwargs)
    input_frame = cf.CoordinateFrame(
        2, ("PIXEL", "PIXEL"), (0, 1), unit=(u.pix, u.pix), name="detector"
    )
    output_frame = cf.CoordinateFrame(1, "CUSTOM", (0,), unit=(u.m / u.s,))
    w = wcs.WCS(
        forward_transform=forward, input_frame=input_frame, output_frame=output_frame
    )

    bb = w.bounding_box
    assert all(u.allclose(u.Quantity(b), [0, 2] * u.pix) for b in bb)


def test_initialize_wcs_with_list():
    # test that you can initialize a wcs with a pipeline that is a list
    # containing both Step() and (frame, transform) tuples

    # make pipeline consisting of tuples and Steps
    shift1 = models.Shift(10 * u.pix) & models.Shift(2 * u.pix)
    shift2 = models.Shift(3 * u.pix)
    pipeline = [("detector", shift1), wcs.Step("extra_step", shift2)]

    end_step = ("end_step", None)
    pipeline.append(end_step)

    # make sure no warnings occur when creating wcs with this pipeline
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wcs.WCS(pipeline)


def test_sip_roundtrip():
    hdr = fits.Header.fromtextfile(data_path / "acs.hdr", endcard=False)

    nx = ny = 1024
    hdr["naxis"] = 2
    hdr["naxis1"] = nx
    hdr["naxis2"] = ny
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=astwcs.FITSFixedWarning)
        gw = _gwcs_from_hst_fits_wcs(hdr)
    hdr_back = gw.to_fits_sip(
        max_pix_error=1e-6,
        max_inv_pix_error=None,
        npoints=64,
        crpix=(hdr["crpix1"], hdr["crpix2"]),
    )

    for k in ["naxis", "naxis1", "naxis2", "ctype1", "ctype2", "a_order", "b_order"]:
        assert hdr[k] == hdr_back[k]

    for k in ["cd1_1", "cd1_2", "cd2_1", "cd2_2"]:
        assert np.allclose(hdr[k], hdr_back[k], atol=1e-14, rtol=1e-9)

    for t in ("a", "b"):
        order = hdr[f"{t}_order"]
        for i in range(order + 1):
            for j in range(order + 1):
                if 1 < i + j <= order:
                    k = f"{t}_{i}_{j}"
                    assert np.allclose(
                        hdr[k], hdr_back[k], atol=1e-15, rtol=1.0e-8 * 10 ** (i + j)
                    )


def test_spatial_spectral_stokes():
    """Converts a FITS WCS to GWCS and compares results."""
    hdr = fits.Header.fromfile(data_path / "stokes.txt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=astwcs.FITSFixedWarning)
        aw = astwcs.WCS(hdr)
    crpix = aw.wcs.crpix
    crval = aw.wcs.crval
    cdelt = aw.wcs.cdelt

    fk5 = cf.CelestialFrame(reference_frame=coord.FK5(), name="FK5")
    detector = cf.Frame2D(name="detector", axes_order=(0, 1))
    spec = cf.SpectralFrame(
        name="FREQ",
        unit=[
            u.Hz,
        ],
        axes_order=(2,),
        axes_names=("freq",),
    )
    stokes = cf.StokesFrame(axes_order=(3,))
    world = cf.CompositeFrame(frames=[fk5, spec, stokes])

    det2sky = (
        models.Shift(-crpix[0]) & models.Shift(-crpix[1])
        | models.Scale(cdelt[0]) & models.Scale(cdelt[1])
        | models.Pix2Sky_SIN()
        | models.RotateNative2Celestial(crval[0], crval[1], 180)
    )
    det2freq = models.Shift(-crpix[2]) | models.Scale(cdelt[2]) | models.Shift(crval[2])
    det2stokes = (
        models.Shift(-crpix[3]) | models.Scale(cdelt[3]) | models.Shift(crval[3])
    )

    gw = wcs.WCS(
        [wcs.Step(detector, det2sky & det2freq & det2stokes), wcs.Step(world, None)]
    )

    x1 = np.array([0, 0, 0, 0, 0])
    x2 = np.array([0, 1, 2, 3, 4])

    gw_sky, gw_spec, gw_stokes = gw.pixel_to_world(x1 + 1, x1 + 1, x1 + 1, x2 + 1)
    aw_sky, aw_spec, aw_stokes = aw.pixel_to_world(x1, x1, x1, x2)

    assert_allclose(gw_sky.data.lon, aw_sky.data.lon)
    assert_allclose(gw_sky.data.lat, aw_sky.data.lat)
    assert_allclose(gw_spec.value, aw_spec.value)
    assert_allclose(gw_stokes.value, aw_stokes.value)


def test_wcs_str():
    w = wcs.WCS(output_frame="icrs")
    assert "icrs" in str(w)


def test_bounding_box_is_returned_F():
    bbox_tuple = ((1, 2), (3, 4))

    detector_2d_frame = cf.Frame2D(name="detector", axes_order=(0, 1))
    model_2d_shift = models.Shift(1) & models.Shift(2)

    model_2d_shift_bbox = model_2d_shift.copy()
    model_2d_shift_bbox.bounding_box = bbox_tuple

    frame = cf.CoordinateFrame(
        name="quantity",
        axes_order=(0, 1),
        naxes=2,
        axes_type=("SPATIAL", "SPATIAL"),
        unit=(u.km, u.km),
    )

    # Demonstrate that model_2d_shift does not have a bounding box
    with pytest.raises(NotImplementedError):
        _ = model_2d_shift.bounding_box

    # Demonstrate that model_2d_shift_bbox does have a bounding box
    assert model_2d_shift_bbox.bounding_box == bbox_tuple

    # Demonstrate the model_2d_shift_bbox has order "C"
    assert model_2d_shift_bbox.bounding_box.order == "C"

    # Create a WCS and then set a bounding box on it
    pipeline_bbox_after = [(detector_2d_frame, model_2d_shift), (frame, None)]
    gwcs_object_after = wcs.WCS(pipeline_bbox_after)
    gwcs_object_after.bounding_box = bbox_tuple

    assert gwcs_object_after.bounding_box == bbox_tuple
    assert gwcs_object_after.bounding_box.order == "F"

    # Create a WCS on transform with a bounding box
    pipeline_bbox_before = [(detector_2d_frame, model_2d_shift_bbox), (frame, None)]
    gwcs_object_before = wcs.WCS(pipeline_bbox_before)

    # Check that first access in this case will raise a warning
    with pytest.warns(wcs.GwcsBoundingBoxWarning):
        _ = gwcs_object_before.bounding_box

    # Check order is returned as F
    assert gwcs_object_before.bounding_box.order == "F"

    # The bounding box tuple will now be ordered differently than the original
    # Tuple due to the order change
    assert gwcs_object_before.bounding_box != bbox_tuple
    assert gwcs_object_before.bounding_box.bounding_box(order="C") == bbox_tuple

    # Show the the bounding box is different between the two WCS objects
    assert gwcs_object_after.bounding_box != gwcs_object_before.bounding_box


def test_no_bounding_box_if_read_from_file(tmp_path):
    bad_wcs = gwcs_2d_bad_bounding_box_order()

    # Check the warning is issued for the bounding box of this WCS object
    with pytest.warns(wcs.GwcsBoundingBoxWarning):
        _ = bad_wcs.bounding_box

    # Check that the warning is not issued again the second time
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = bad_wcs.bounding_box

    # Write a bad wcs bounding box to an asdf file
    asdf_file = tmp_path / "bad_wcs.asdf"
    af = asdf.AsdfFile(
        {"wcs": gwcs_2d_bad_bounding_box_order()}
    )  # re-create the bad wcs object
    af.write_to(asdf_file)

    with asdf.open(asdf_file) as af:
        wcs_from_file = af["wcs"]

    # Check that no warning is issued for the bounding box of this WCS object
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = wcs_from_file.bounding_box


def test_split_frame_wcs():
    # Setup a WCS where the pixel & world axes are (lat, wave, lon)

    # We setup a model which is pretending to be a celestial transform. Note
    # that we are pretending that this model is ordered lon, lat because that's
    # what the projections require in astropy.

    # Input is lat, wave, lon
    # lat: multiply by 20 arcsec, lon: multiply by 15 deg
    # result should be 20 arcsec, 10nm, 45 deg
    spatial = models.Multiply(20 * u.arcsec / u.pix) & models.Multiply(
        15 * u.deg / u.pix
    )
    compound = models.Linear1D(intercept=0 * u.nm, slope=10 * u.nm / u.pix) & spatial
    # This forward transforms uses mappings to be (lat, wave, lon)
    forward = models.Mapping((1, 0, 2)) | compound | models.Mapping((1, 0, 2))

    # Setup the output frame
    celestial_frame = cf.CelestialFrame(
        axes_order=(2, 0),
        unit=(u.deg, u.arcsec),
        reference_frame=coord.ICRS(),
        axes_names=("lon", "lat"),
    )
    spectral_frame = cf.SpectralFrame(axes_order=(1,), unit=u.nm, axes_names="wave")
    output_frame = cf.CompositeFrame([spectral_frame, celestial_frame])

    input_frame = cf.CoordinateFrame(
        3, ["PIXEL"] * 3, axes_order=list(range(3)), unit=[u.pix] * 3
    )

    iwcs = wcs.WCS(forward, input_frame, output_frame)
    input_pixel = [1 * u.pix, 1 * u.pix, 3 * u.pix]
    output_world = iwcs.pixel_to_world_values(*input_pixel)
    output_pixel = iwcs.world_to_pixel_values(*output_world)
    assert_allclose(output_pixel, u.Quantity(input_pixel).to_value(u.pix))

    expected_world = [20 * u.arcsec, 10 * u.nm, 45 * u.deg]
    for expected, output in zip(expected_world, output_world, strict=False):
        assert_allclose(output, expected.value)

    world_obj = iwcs.pixel_to_world(*input_pixel)
    assert isinstance(world_obj[0], coord.SkyCoord)
    assert isinstance(world_obj[1], coord.SpectralCoord)

    assert u.allclose(world_obj[0].spherical.lat, expected_world[0])
    assert u.allclose(world_obj[0].spherical.lon, expected_world[2])
    assert u.allclose(world_obj[1], expected_world[1])

    obj_pixel = iwcs.world_to_pixel(*world_obj)
    assert_allclose(obj_pixel, u.Quantity(input_pixel).to_value(u.pix))


def test_reordered_celestial():
    # This is a spatial model which is ordered lat, lon for the purposes of this test.
    # Expected lat=45 deg, lon=20 arcsec
    spatial = models.Multiply(20 * u.arcsec / u.pix) & models.Multiply(
        15 * u.deg / u.pix
    ) | models.Mapping((1, 0))

    celestial_frame = cf.CelestialFrame(
        axes_order=(1, 0), unit=(u.arcsec, u.deg), reference_frame=coord.ICRS()
    )

    input_frame = cf.CoordinateFrame(
        2, ["PIXEL"] * 2, axes_order=list(range(2)), unit=[u.pix] * 2
    )

    iwcs = wcs.WCS(spatial, input_frame, celestial_frame)

    input_pixel = [1 * u.pix, 3 * u.pix]
    output_world = iwcs.pixel_to_world_values(*input_pixel)
    output_pixel = iwcs.world_to_pixel_values(*output_world)
    assert_allclose(output_pixel, u.Quantity(input_pixel).to_value(u.pix))

    expected_world = [45 * u.deg, 20 * u.arcsec]
    assert_allclose(output_world, [e.value for e in expected_world])

    world_obj = iwcs.pixel_to_world(*input_pixel)
    assert isinstance(world_obj, coord.SkyCoord)

    assert u.allclose(world_obj.spherical.lat, expected_world[0])
    assert u.allclose(world_obj.spherical.lon, expected_world[1])

    obj_pixel = iwcs.world_to_pixel(world_obj)
    assert_allclose(obj_pixel, u.Quantity(input_pixel).to_value(u.pix))


def test_high_level_objects_in_pipeline_forward(gwcs_with_pipeline_celestial):
    """
    This test checks that high level objects still work with a multi-stage
    pipeline when doing forward transforms.
    """
    iwcs = gwcs_with_pipeline_celestial

    input_pixel = [1 * u.pix, 1 * u.pix]

    output_world = iwcs(*input_pixel)

    assert output_world[0].unit == u.deg
    assert output_world[1].unit == u.deg
    assert u.allclose(output_world[0], 20 * u.arcsec + 1 * u.deg)
    assert u.allclose(output_world[1], 15 * u.deg + 2 * u.deg)

    # with_units=True puts the result in the frame units rather than in the
    # model units.
    output_world_with_units = iwcs(*input_pixel, with_units=True)
    assert output_world_with_units[0].unit is u.arcsec
    assert output_world_with_units[1].unit is u.arcsec

    # This should be in model units of the spatial model
    intermediate_world = iwcs.transform(
        "input",
        "celestial",
        *input_pixel,
    )
    assert intermediate_world[0].unit == u.arcsec
    assert intermediate_world[1].unit == u.deg
    assert u.allclose(intermediate_world[0], 20 * u.arcsec)
    assert u.allclose(intermediate_world[1], 15 * u.deg)

    intermediate_world_with_units = iwcs.transform(
        "input", "celestial", *input_pixel, with_units=True
    )
    assert isinstance(intermediate_world_with_units, coord.SkyCoord)
    sc = intermediate_world_with_units
    assert u.allclose(sc.ra, 20 * u.arcsec)
    assert u.allclose(sc.dec, 15 * u.deg)


def test_high_level_objects_in_pipeline_backward(gwcs_with_pipeline_celestial):
    """
    This test checks that high level objects still work with a multi-stage
    pipeline when doing backward transforms.
    """
    iwcs = gwcs_with_pipeline_celestial

    input_world = [
        20 * u.arcsec + 1 * u.deg,
        15 * u.deg + 2 * u.deg,
    ]
    pixel = iwcs.invert(*input_world)

    assert all(isinstance(p, u.Quantity) for p in pixel)
    assert u.allclose(pixel, [1, 1] * u.pix)

    pixel = iwcs.invert(
        *input_world,
        with_units=True,
    )

    assert all(isinstance(p, u.Quantity) for p in pixel)
    assert u.allclose(pixel, [1, 1] * u.pix)

    intermediate_world = iwcs.transform(
        "output",
        "celestial",
        *input_world,
    )
    assert all(isinstance(p, u.Quantity) for p in intermediate_world)
    assert u.allclose(intermediate_world, [20 * u.arcsec, 15 * u.deg])

    intermediate_world = iwcs.transform(
        "output",
        "celestial",
        *input_world,
        with_units=True,
    )
    assert isinstance(intermediate_world, coord.SkyCoord)


def test_error_with_duplicate_frames():
    """
    Test that an error is raised if a frame is used more than once in the pipeline.
    """
    pipeline = [(detector, m1), (detector, m2), (focal, None)]

    with pytest.raises(ValueError, match="Frame detector is already in the pipeline."):
        wcs.WCS(pipeline)


def test_error_with_not_none_last():
    """
    Test that an error is raised if the last transform is not None
    """

    pipeline = [(detector, m1), (focal, m2)]

    with pytest.raises(
        ValueError, match="The last step in the pipeline must have a None transform."
    ):
        wcs.WCS(pipeline)


def test_bounding_box_with_units():
    """
    Test that the invert method works when a bounding box has units.
    """

    # GWCS that is adapted from its Getting Started.
    shift_by_crpix = models.Shift(-(5 - 1) * u.pix) & models.Shift(-(5 - 1) * u.pix)
    matrix = np.array(
        [
            [1.290551569736e-05, 5.9525007864732e-06],
            [5.0226382102765e-06, -1.2644844123757e-05],
        ]
    )
    rotation = models.AffineTransformation2D(matrix * u.deg, translation=[0, 0] * u.deg)
    rotation.input_units_equivalencies = {
        "x": u.pixel_scale(1 * (u.deg / u.pix)),
        "y": u.pixel_scale(1 * (u.deg / u.pix)),
    }
    rotation.inverse = models.AffineTransformation2D(
        np.linalg.inv(matrix) * u.pix, translation=[0, 0] * u.pix
    )
    rotation.inverse.input_units_equivalencies = {
        "x": u.pixel_scale(1 * (u.pix / u.deg)),
        "y": u.pixel_scale(1 * (u.pix / u.deg)),
    }
    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(
        3.581704851882 * u.deg, -30.39197867265 * u.deg, 180 * u.deg
    )
    det2sky = shift_by_crpix | rotation | tan | celestial_rotation
    det2sky.name = "linear_transform"
    detector_frame = cf.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )
    sky_frame = cf.CelestialFrame(
        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
    )
    pipeline = [(detector_frame, det2sky), (sky_frame, None)]
    w_gwcs = wcs.WCS(pipeline)
    w_gwcs.bounding_box = ((0, 8), (0, 10)) * u.pix  # x, y

    w_gwcs.invert(4 * u.deg, 5 * u.deg)
    w_gwcs.to_fits(bounding_box=([0, 100] * u.pix, [0, 100] * u.pix))


def test_direct_numerical_inverse(gwcs_romanisim):
    xy = (128, 256)
    coord = gwcs_romanisim(*xy)
    ra_dec = (np.radians(c) * u.rad for c in coord)
    out = gwcs_romanisim.numerical_inverse(*ra_dec)

    assert_allclose(xy, out)


def test_array_high_level_output():
    """
    Test that we don't loose array values when requesting a high-level output
    from a WCS object.
    """
    input_frame = cf.CoordinateFrame(
        naxes=1,
        axes_type=("SPATIAL",),
        axes_order=(0,),
        name="pixels",
        unit=(u.pix,),
        axes_names=("x",),
    )
    output_frame = cf.SpectralFrame(unit=(u.nm,), axes_names=("lambda",))
    wave_model = models.Scale(0.1) | models.Shift(500)
    gwcs = wcs.WCS([(input_frame, wave_model), (output_frame, None)])
    assert (
        gwcs(np.array([0, 1, 2]), with_units=True)
        == coord.SpectralCoord([500, 500.1, 500.2] * u.nm)
    ).all()


def test_parameterless_transform():
    """
    Test that a transform with no parameters correctly handles units.
    -> The wcs does not introduce units when evaluating the forward or backward
      transform for models with no parameters
    Regression test for #558
    """

    in_frame = cf.Frame2D(name="in_frame")
    out_frame = cf.Frame2D(name="out_frame")

    gwcs = wcs.WCS(
        [
            (in_frame, models.Identity(2)),
            (out_frame, None),
        ]
    )

    # The expectation for this wcs is that:
    # - gwcs(1, 1) has no units
    # (__call__ apparently is supposed to pass units through?)
    # - gwcs(1*u.pix, 1*u.pix) has units
    # - gwcs.invert(1, 1) has no units
    # - gwcs.invert(1*u.pix, 1*u.pix) has no units

    # No units introduced by the forward transform
    assert gwcs(1, 1) == (1, 1)
    assert gwcs(1 * u.pix, 1 * u.pix) == (1 * u.pix, 1 * u.pix)

    assert gwcs.invert(1, 1) == (1, 1)
    assert gwcs.invert(1 * u.pix, 1 * u.pix) == (1, 1)


def test_fitswcs_imaging(fits_wcs_imaging_simple):
    """Test simple FITS type imaging WCS."""
    forward_transform = fits_wcs_imaging_simple.forward_transform
    ra, dec = fits_wcs_imaging_simple(*forward_transform.crpix)
    assert_allclose((ra, dec), forward_transform.crval)
    assert_allclose(fits_wcs_imaging_simple.invert(ra, dec), forward_transform.crpix)

    sky = fits_wcs_imaging_simple.pixel_to_world(*forward_transform.crpix)
    ra, dec = sky.data.lon.value, sky.data.lat.value
    assert_allclose((ra, dec), forward_transform.crval)
    assert_allclose(
        fits_wcs_imaging_simple.world_to_pixel(sky), forward_transform.crpix
    )
