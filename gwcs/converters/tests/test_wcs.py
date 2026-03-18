# Licensed under a 3-clause BSD style license - see LICENSE.rst

import asdf
import pytest
from asdf_astropy.testing.helpers import (
    assert_model_equal,
)
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
from astropy.modeling import models

from gwcs import coordinate_frames as cf
from gwcs import wcs
from gwcs.wcs._step import _is_coordinate_frame


def _assert_frame_equal(a, b):
    __tracebackhide__ = True

    assert type(a) is type(b)

    if a is None:
        return None

    if not _is_coordinate_frame(a):
        return a == b

    assert a.name == b.name  # nosec
    if not isinstance(a, cf.EmptyFrame) or (
        isinstance(a, cf.EmptyFrame) and a._naxes is not None
    ):
        assert a.axes_order == b.axes_order  # nosec
        assert a.axes_names == b.axes_names  # nosec
        assert a.unit == b.unit  # nosec
        assert a.reference_frame == b.reference_frame  # nosec
    return None


def assert_frame_roundtrip(frame, tmp_path, version=None):
    """
    Assert that a frame can be written to an ASDF file and read back
    in without losing any of its essential properties.
    """
    path = tmp_path / "test.asdf"

    with asdf.AsdfFile({"frame": frame}, version=version) as af:
        af.write_to(path)

    with asdf.open(path) as af:
        _assert_frame_equal(frame, af["frame"])


def _assert_wcs_equal(a, b):
    assert a.name == b.name  # nosec
    assert a.pixel_shape == b.pixel_shape
    assert len(a.available_frames) == len(b.available_frames)  # nosec
    for a_step, b_step in zip(a.pipeline, b.pipeline, strict=False):
        _assert_frame_equal(a_step.frame, b_step.frame)
        assert_model_equal(a_step.transform, b_step.transform)


def assert_wcs_roundtrip(wcs, tmp_path, version=None):
    path = tmp_path / "test.asdf"

    with asdf.AsdfFile({"wcs": wcs}, version=version) as af:
        af.write_to(path)

    with asdf.open(path) as af:
        _assert_wcs_equal(wcs, af["wcs"])


def _wcs_factory():
    m1 = models.Shift(12.4) & models.Shift(-2)
    icrs = cf.CelestialFrame(name="icrs", reference_frame=coord.ICRS())
    det = cf.Frame2D(name="detector", axes_order=(0, 1))

    with pytest.warns(DeprecationWarning, match=r"The use of strings.*"):
        gw1 = wcs.WCS(output_frame="icrs", input_frame="detector", forward_transform=m1)

    gw2 = wcs.WCS(output_frame=icrs, input_frame=det, forward_transform=m1)
    gw3 = wcs.WCS(output_frame=icrs, input_frame=det, forward_transform=m1)
    gw3.pixel_shape = (100, 200)

    return [gw1, gw2, gw3]


@pytest.mark.parametrize("gw", _wcs_factory())
def test_create_wcs(tmp_path, gw):
    assert_wcs_roundtrip(gw, tmp_path)


def _composite_frame_factory():
    icrs = coord.ICRS()
    fk5 = coord.FK5()
    cel1 = cf.CelestialFrame(reference_frame=icrs)
    cel2 = cf.CelestialFrame(reference_frame=fk5)

    spec1 = cf.SpectralFrame(name="freq", unit=(u.Hz,), axes_order=(2,))
    spec2 = cf.SpectralFrame(name="wave", unit=(u.m,), axes_order=(2,))

    comp1 = cf.CompositeFrame([cel1, spec1])
    comp2 = cf.CompositeFrame([cel2, spec2])
    comp = cf.CompositeFrame([comp1, cf.SpectralFrame(axes_order=(3,), unit=(u.m,))])

    return [comp1, comp2, comp]


@pytest.mark.parametrize("frame", _composite_frame_factory())
def test_composite_frame(tmp_path, frame):
    assert_frame_roundtrip(frame, tmp_path)


def _frame_factory():
    """Creates an array of frames to be used for testing."""

    return [
        cf.CelestialFrame(reference_frame=coord.ICRS()),
        cf.CelestialFrame(reference_frame=coord.ICRS(), axes_order=(1, 0)),
        cf.CelestialFrame(reference_frame=coord.FK5(equinox=time.Time("2010-01-01"))),
        cf.CelestialFrame(
            reference_frame=coord.FK4(
                equinox=time.Time("2010-01-01"), obstime=time.Time("2015-01-01")
            )
        ),
        cf.CelestialFrame(
            reference_frame=coord.FK4NoETerms(
                equinox=time.Time("2010-01-01"), obstime=time.Time("2015-01-01")
            )
        ),
        cf.CelestialFrame(reference_frame=coord.Galactic()),
        cf.CelestialFrame(
            reference_frame=coord.Galactocentric(
                # A default galcen_coord is used since none is provided here
                galcen_distance=5.0 * u.m,
                z_sun=3 * u.pc,
                roll=3 * u.deg,
            )
        ),
        cf.CelestialFrame(
            reference_frame=coord.GCRS(
                obstime=time.Time("2010-01-01"),
                obsgeoloc=[1, 3, 2000] * u.pc,
                obsgeovel=[2, 1, 8] * (u.m / u.s),
            )
        ),
        cf.CelestialFrame(reference_frame=coord.CIRS(obstime=time.Time("2010-01-01"))),
        cf.CelestialFrame(reference_frame=coord.ITRS(obstime=time.Time("2022-01-03"))),
        cf.CelestialFrame(
            reference_frame=coord.PrecessedGeocentric(
                obstime=time.Time("2010-01-01"),
                obsgeoloc=[1, 3, 2000] * u.pc,
                obsgeovel=[2, 1, 8] * (u.m / u.s),
            )
        ),
        cf.SpectralFrame(name="freq", unit=(u.Hz,), axes_order=(2,)),
        cf.SpectralFrame(name="wave", unit=(u.m,), axes_order=(2,)),
        cf.StokesFrame(),
        cf.StokesFrame(name="polarisation", axes_order=(3,)),
        cf.TemporalFrame(time.Time("2011-01-01")),
        cf.TemporalFrame(time.Time("2011-01-01"), axes_order=(2,)),
        cf.Frame2D(),
        cf.Frame2D(name="pixels"),
        cf.Frame2D(name="detector", axes_order=(1, 0)),
        cf.Frame2D(name="world", axes_order=(0, 1), unit=(u.m, u.km)),
        cf.Frame2D(
            name="reverse_world",
            axes_order=(1, 0),
            unit=(u.m, u.km),
            axes_names=("foo", "bar"),
        ),
        cf.CoordinateFrame(
            naxes=7,
            axes_type=((cf.AxisType.SPATIAL,) * 4) + ((cf.AxisType.PIXEL,) * 3),
            axes_order=tuple(range(7)),
        ),
    ]


@pytest.mark.parametrize("frame", _frame_factory())
def test_frames(tmp_path, frame):
    assert_frame_roundtrip(frame, tmp_path)


def test_references(tmp_path):
    m1 = models.Shift(12.4) & models.Shift(-2)
    icrs = cf.CelestialFrame(name="icrs", reference_frame=coord.ICRS())
    det = cf.Frame2D(name="detector", axes_order=(0, 1))
    det2 = cf.Frame2D(name="detector2", axes_order=(0, 1))
    focal = cf.Frame2D(name="focal", axes_order=(0, 1))

    pipe1 = [(det, m1), (focal, m1), (icrs, None)]
    gw1 = wcs.WCS(pipe1)

    pipe2 = [(det, m1), (det2, m1), (icrs, None)]
    gw2 = wcs.WCS(pipe2)

    tree = {"wcs1": gw1, "wcs2": gw2}
    af = asdf.AsdfFile(tree)
    output_path = tmp_path / "test.asdf"
    af.write_to(output_path)

    with asdf.open(output_path) as af:
        gw1 = af.tree["wcs1"]
        gw2 = af.tree["wcs2"]
        assert gw1.pipeline[0].transform is gw1.pipeline[1].transform
        assert gw2.pipeline[0].transform is gw2.pipeline[1].transform
        assert gw2.pipeline[0].frame is gw1.pipeline[0].frame
