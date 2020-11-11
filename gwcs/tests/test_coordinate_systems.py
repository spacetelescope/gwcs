# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy import coordinates as coord
from astropy.tests.helper import assert_quantity_allclose
from astropy.modeling import models as m

from .. import WCS
from .. import coordinate_frames as cf

import astropy
astropy_version = astropy.__version__

coord_frames = coord.builtin_frames.__all__[:]

# Need to write a better test, using a dict {coord_frame: input_parameters}
# For now remove OffsetFrame, issue #55
try:
    coord_frames.remove("SkyOffsetFrame")
except ValueError:
    pass


icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), axes_order=(0, 1))
detector = cf.Frame2D(name='detector', axes_order=(0, 1))
focal = cf.Frame2D(name='focal', axes_order=(0, 1), unit=(u.m, u.m))

spec1 = cf.SpectralFrame(name='freq', unit=[u.Hz, ], axes_order=(2, ))
spec2 = cf.SpectralFrame(name='wave', unit=[u.m, ], axes_order=(2, ), axes_names=('lambda', ))
spec3 = cf.SpectralFrame(name='energy', unit=[u.J, ], axes_order=(2, ))
spec4 = cf.SpectralFrame(name='pixel', unit=[u.pix, ], axes_order=(2, ))
spec5 = cf.SpectralFrame(name='speed', unit=[u.m/u.s, ], axes_order=(2, ))

comp1 = cf.CompositeFrame([icrs, spec1])
comp2 = cf.CompositeFrame([focal, spec2])
comp3 = cf.CompositeFrame([icrs, spec3])
comp4 = cf.CompositeFrame([icrs, spec4])
comp5 = cf.CompositeFrame([icrs, spec5])
comp = cf.CompositeFrame([comp1, cf.SpectralFrame(axes_order=(3,), unit=(u.m,))])

xscalar = 1
yscalar = 2
xarr = np.arange(5)
yarr = np.arange(5)

inputs2 = [(xscalar, yscalar), (xarr, yarr)]
inputs1 = [xscalar, xarr]
inputs3 = [(xscalar, yscalar, xscalar), (xarr, yarr, xarr)]


def test_units():
    assert(comp1.unit == (u.deg, u.deg, u.Hz))
    assert(comp2.unit == (u.m, u.m, u.m))
    assert(comp3.unit == (u.deg, u.deg, u.J))
    assert(comp4.unit == (u.deg, u.deg, u.pix))
    assert(comp5.unit == (u.deg, u.deg, u.m/u.s))
    assert(comp.unit == (u.deg, u.deg, u.Hz, u.m))


@pytest.mark.parametrize('inputs', inputs2)
def test_coordinates_spatial(inputs):
    sky_coo = icrs.coordinates(*inputs)
    assert isinstance(sky_coo, coord.SkyCoord)
    assert_allclose((sky_coo.ra.value, sky_coo.dec.value), inputs)
    focal_coo = focal.coordinates(*inputs)
    assert_allclose([coo.value for coo in focal_coo], inputs)
    assert [coo.unit for coo in focal_coo] == [u.m, u.m]


@pytest.mark.parametrize('inputs', inputs1)
def test_coordinates_spectral(inputs):
    wave = spec2.coordinates(inputs)
    assert_allclose(wave.value, inputs)
    assert wave.unit == 'meter'
    assert isinstance(wave, u.Quantity)


@pytest.mark.parametrize('inputs', inputs3)
def test_coordinates_composite(inputs):
    frame = cf.CompositeFrame([icrs, spec2])
    result = frame.coordinates(*inputs)
    assert isinstance(result[0], coord.SkyCoord)
    assert_allclose((result[0].ra.value, result[0].dec.value), inputs[:2])
    assert_allclose(result[1].value, inputs[2])


def test_coordinates_composite_order():
    time = cf.TemporalFrame(Time("2011-01-01T00:00:00"), name='time', unit=[u.s, ], axes_order=(0, ))
    dist = cf.CoordinateFrame(name='distance', naxes=1,
                              axes_type=["SPATIAL"], unit=[u.m, ], axes_order=(1, ))
    frame = cf.CompositeFrame([time, dist])
    result = frame.coordinates(0, 0)
    assert result[0] == Time("2011-01-01T00:00:00")
    assert u.allclose(result[1], 0*u.m)


def test_bare_baseframe():
    # This is a regression test for the following call:
    frame = cf.CoordinateFrame(1, "SPATIAL", (0,), unit=(u.km,))
    assert u.allclose(frame.coordinate_to_quantity((1*u.m,)), 1*u.m)

    # Now also setup the same situation through the whole call stack to be safe.
    w = WCS(forward_transform=m.Tabular1D(points=np.arange(10)*u.pix,
                                          lookup_table=np.arange(10)*u.km),
            output_frame=frame,
            input_frame=cf.CoordinateFrame(1, "PIXEL", (0,), unit=(u.pix,), name="detector_frame")
            )
    assert u.allclose(w.world_to_pixel(0*u.km), 0)


@pytest.mark.parametrize(('frame'), coord_frames)
def test_celestial_attributes_length(frame):
    """
    Test getting default values for
    CelestialFrame attributes from reference_frame.
    """
    fr = getattr(coord, frame)
    if issubclass(fr.__class__, coord.BaseCoordinateFrame):
        cel = cf.CelestialFrame(reference_frame=fr())
        assert(len(cel.axes_names) == len(cel.axes_type) == len(cel.unit) == \
               len(cel.axes_order) == cel.naxes)


def test_axes_type():
    assert(icrs.axes_type == ('SPATIAL', 'SPATIAL'))
    assert(spec1.axes_type == ('SPECTRAL',))
    assert(detector.axes_type == ('SPATIAL', 'SPATIAL'))
    assert(focal.axes_type == ('SPATIAL', 'SPATIAL'))


def test_length_attributes():
    with pytest.raises(ValueError):
        cf.CoordinateFrame(naxes=2, unit=(u.deg),
                           axes_type=("SPATIAL", "SPATIAL"),
                           axes_order=(0, 1))

    with pytest.raises(ValueError):
        cf.CoordinateFrame(naxes=2, unit=(u.deg, u.deg),
                           axes_type=("SPATIAL",),
                           axes_order=(0, 1))

    with pytest.raises(ValueError):
        cf.CoordinateFrame(naxes=2, unit=(u.deg, u.deg),
                           axes_type=("SPATIAL", "SPATIAL"),
                           axes_order=(0,))


def test_base_coordinate():
    frame = cf.CoordinateFrame(naxes=2, axes_type=("SPATIAL", "SPATIAL"),
                               axes_order=(0, 1))
    assert frame.name == 'CoordinateFrame'
    frame = cf.CoordinateFrame(name="CustomFrame", naxes=2,
                               axes_type=("SPATIAL", "SPATIAL"),
                               axes_order=(0, 1))
    assert frame.name == 'CustomFrame'
    frame.name = "DeLorean"
    assert frame.name == 'DeLorean'

    q1, q2 = frame.coordinate_to_quantity(12 * u.deg, 3 * u.arcsec)
    assert_quantity_allclose(q1, 12 * u.deg)
    assert_quantity_allclose(q2, 3 * u.arcsec)

    q1, q2 = frame.coordinate_to_quantity((12 * u.deg, 3 * u.arcsec))
    assert_quantity_allclose(q1, 12 * u.deg)
    assert_quantity_allclose(q2, 3 * u.arcsec)


def test_temporal_relative():
    t = cf.TemporalFrame(reference_frame=Time("2018-01-01T00:00:00"), unit=u.s)
    assert t.coordinates(10) == Time("2018-01-01T00:00:00") + 10 * u.s
    assert t.coordinates(10 * u.s) == Time("2018-01-01T00:00:00") + 10 * u.s

    a = t.coordinates((10, 20))
    assert a[0] == Time("2018-01-01T00:00:00") + 10 * u.s
    assert a[1] == Time("2018-01-01T00:00:00") + 20 * u.s

    t = cf.TemporalFrame(reference_frame=Time("2018-01-01T00:00:00"))
    assert t.coordinates(10 * u.s) == Time("2018-01-01T00:00:00") + 10 * u.s
    assert t.coordinates(TimeDelta(10, format='sec')) == Time("2018-01-01T00:00:00") + 10 * u.s

    a = t.coordinates((10, 20) * u.s)
    assert a[0] == Time("2018-01-01T00:00:00") + 10 * u.s
    assert a[1] == Time("2018-01-01T00:00:00") + 20 * u.s


@pytest.mark.skipif(astropy_version<"4", reason="Requires astropy 4.0 or higher")
def test_temporal_absolute():
    t = cf.TemporalFrame(reference_frame=Time([], format='isot'))
    assert t.coordinates("2018-01-01T00:00:00") == Time("2018-01-01T00:00:00")

    a = t.coordinates(("2018-01-01T00:00:00", "2018-01-01T00:10:00"))
    assert a[0] == Time("2018-01-01T00:00:00")
    assert a[1] == Time("2018-01-01T00:10:00")

    t = cf.TemporalFrame(reference_frame=Time([], scale='tai', format='isot'))
    assert t.coordinates("2018-01-01T00:00:00") == Time("2018-01-01T00:00:00", scale='tai')


@pytest.mark.parametrize('inp', [
    (10 * u.deg, 20 * u.deg),
    ((10 * u.deg, 20 * u.deg),),
    (u.Quantity([10, 20], u.deg),),
    (coord.SkyCoord(10 * u.deg, 20 * u.deg, frame=coord.ICRS),),
    # This is the same as 10,20 in ICRS
    (coord.SkyCoord(119.26936774, -42.79039286, unit=u.deg, frame='galactic'),)
])
def test_coordinate_to_quantity_celestial(inp):
    cel = cf.CelestialFrame(reference_frame=coord.ICRS(), axes_order=(0, 1))

    lon, lat = cel.coordinate_to_quantity(*inp)
    assert_quantity_allclose(lon, 10 * u.deg)
    assert_quantity_allclose(lat, 20 * u.deg)

    with pytest.raises(ValueError):
        cel.coordinate_to_quantity(10 * u.deg, 2 * u.deg, 3 * u.deg)

    with pytest.raises(ValueError):
        cel.coordinate_to_quantity((1, 2))


@pytest.mark.parametrize('inp', [
    (100,),
    (100 * u.nm,),
    (0.1 * u.um,),
])
def test_coordinate_to_quantity_spectral(inp):
    spec = cf.SpectralFrame(unit=u.nm, axes_order=(1, ))
    wav = spec.coordinate_to_quantity(*inp)
    assert_quantity_allclose(wav, 100 * u.nm)


@pytest.mark.parametrize('inp', [
    (Time("2011-01-01T00:00:10"),),
    (10 * u.s,)
])
@pytest.mark.skipif(astropy_version<"4", reason="Requires astropy 4.0 or higher.")
def test_coordinate_to_quantity_temporal(inp):
    temp = cf.TemporalFrame(reference_frame=Time("2011-01-01T00:00:00"), unit=u.s)

    t = temp.coordinate_to_quantity(*inp)

    assert_quantity_allclose(t, 10 * u.s)

    temp2 = cf.TemporalFrame(reference_frame=Time([], format='isot'), unit=u.s)

    tt = Time("2011-01-01T00:00:00")
    t = temp2.coordinate_to_quantity(tt)

    assert t is tt


@pytest.mark.parametrize('inp', [
    (211 * u.AA, 0 * u.s, 0 * u.arcsec, 0 * u.arcsec),
    (211 * u.AA, 0 * u.s, (0 * u.arcsec, 0 * u.arcsec)),
    (211 * u.AA, 0 * u.s, (0, 0) * u.arcsec),
    (211 * u.AA, Time("2011-01-01T00:00:00"), (0, 0) * u.arcsec),
    (211 * u.AA, Time("2011-01-01T00:00:00"), coord.SkyCoord(0, 0, unit=u.arcsec)),
])
def test_coordinate_to_quantity_composite(inp):
    # Composite
    wave_frame = cf.SpectralFrame(axes_order=(0, ), unit=u.AA)
    time_frame = cf.TemporalFrame(
        axes_order=(1, ), unit=u.s, reference_frame=Time("2011-01-01T00:00:00"))
    sky_frame = cf.CelestialFrame(axes_order=(2, 3), reference_frame=coord.ICRS())

    comp = cf.CompositeFrame([wave_frame, time_frame, sky_frame])

    coords = comp.coordinate_to_quantity(*inp)

    expected = (211 * u.AA, 0 * u.s, 0 * u.arcsec, 0 * u.arcsec)
    for output, exp in zip(coords, expected):
        assert_quantity_allclose(output, exp)


def test_stokes_frame():
    sf = cf.StokesFrame()

    assert sf.coordinates(0) == 'I'
    assert sf.coordinates(0 * u.pix) == 'I'
    assert sf.coordinate_to_quantity('I') == 0 * u.one
    assert sf.coordinate_to_quantity(0) == 0


@pytest.mark.parametrize('inp', [
    (211 * u.AA, 0 * u.s, 0 * u.one, 0 * u.one),
    (211 * u.AA, 0 * u.s, (0 * u.one, 0 * u.one)),
    (211 * u.AA, 0 * u.s, (0, 0) * u.one),
    (211 * u.AA, Time("2011-01-01T00:00:00"), (0, 0) * u.one)
])
def test_coordinate_to_quantity_frame2d_composite(inp):
    wave_frame = cf.SpectralFrame(axes_order=(0, ), unit=u.AA)
    time_frame = cf.TemporalFrame(
        axes_order=(1, ), unit=u.s, reference_frame=Time("2011-01-01T00:00:00"))

    frame2d = cf.Frame2D(name="intermediate", axes_order=(2, 3), unit=(u.one, u.one))

    comp = cf.CompositeFrame([wave_frame, time_frame, frame2d])

    coords = comp.coordinate_to_quantity(*inp)

    expected = (211 * u.AA, 0 * u.s, 0 * u.one, 0 * u.one)
    for output, exp in zip(coords, expected):
        assert_quantity_allclose(output, exp)


def test_coordinate_to_quantity_frame_2d():
    frame = cf.Frame2D(unit=(u.one, u.arcsec))
    inp = (1, 2)
    expected = (1 * u.one, 2 * u.arcsec)
    result = frame.coordinate_to_quantity(*inp)
    for output, exp in zip(result, expected):
        assert_quantity_allclose(output, exp)

    inp = (1 * u.one, 2)
    expected = (1 * u.one, 2 * u.arcsec)
    result = frame.coordinate_to_quantity(*inp)
    for output, exp in zip(result, expected):
        assert_quantity_allclose(output, exp)


@pytest.mark.skipif(astropy_version<"4", reason="Requires astropy 4.0 or higher.")
def test_coordinate_to_quantity_error():
    frame = cf.Frame2D(unit=(u.one, u.arcsec))
    with pytest.raises(ValueError):
        frame.coordinate_to_quantity(1)

    with pytest.raises(ValueError):
        comp1.coordinate_to_quantity((1, 1), 2)

    frame = cf.TemporalFrame(reference_frame=Time([], format='isot'), unit=u.s)
    with pytest.raises(ValueError):
        frame.coordinate_to_quantity(1)


def test_axis_physical_type():
    assert icrs.axis_physical_types == ("pos.eq.ra", "pos.eq.dec")
    assert spec1.axis_physical_types == ("em.freq",)
    assert spec2.axis_physical_types == ("em.wl",)
    assert spec3.axis_physical_types == ("em.energy",)
    assert spec4.axis_physical_types == ("custom:unknown",)
    assert spec5.axis_physical_types == ("spect.dopplerVeloc",)
    assert comp1.axis_physical_types == ("pos.eq.ra", "pos.eq.dec", "em.freq")
    assert comp2.axis_physical_types == ("custom:x", "custom:y", "em.wl")
    assert comp3.axis_physical_types == ("pos.eq.ra", "pos.eq.dec", "em.energy")
    assert comp.axis_physical_types == ('pos.eq.ra', 'pos.eq.dec', 'em.freq', 'em.wl')

    spec6 = cf.SpectralFrame(name='waven', axes_order=(1,),
                             axis_physical_types='em.wavenumber')
    assert spec6.axis_physical_types == ('em.wavenumber',)

    t = cf.TemporalFrame(reference_frame=Time("2018-01-01T00:00:00"), unit=u.s)
    assert t.axis_physical_types == ('time',)

    fr2d = cf.Frame2D(name='d', axes_names=("x", "y"))
    assert fr2d.axis_physical_types == ('custom:x', 'custom:y')

    fr2d = cf.Frame2D(name='d', axes_names=None)
    assert fr2d.axis_physical_types == ('custom:SPATIAL', 'custom:SPATIAL')

    fr2d = cf.Frame2D(name='d', axis_physical_types=("pos.x", "pos.y"))
    assert fr2d.axis_physical_types == ('custom:pos.x', 'custom:pos.y')

    with pytest.raises(ValueError):
        cf.CelestialFrame(reference_frame=coord.ICRS(), axis_physical_types=("pos.eq.ra",))

    fr = cf.CelestialFrame(reference_frame=coord.ICRS(), axis_physical_types=("ra", "dec"))
    assert fr.axis_physical_types == ("custom:ra", "custom:dec")

    fr = cf.CelestialFrame(reference_frame=coord.BarycentricTrueEcliptic())
    assert fr.axis_physical_types == ('pos.ecliptic.lon', 'pos.ecliptic.lat')

    frame = cf.CoordinateFrame(name='custom_frame', axes_type=("SPATIAL",),
                               axes_order=(0,), axis_physical_types="length",
                               axes_names="x", naxes=1)
    assert frame.axis_physical_types == ("custom:length",)
    frame = cf.CoordinateFrame(name='custom_frame', axes_type=("SPATIAL",),
                               axes_order=(0,), axis_physical_types=("length",),
                               axes_names="x", naxes=1)
    assert frame.axis_physical_types == ("custom:length",)
    with pytest.raises(ValueError):
        cf.CoordinateFrame(name='custom_frame', axes_type=("SPATIAL",),
                           axes_order=(0,),
                           axis_physical_types=("length", "length"), naxes=1)


def test_base_frame():
    with pytest.raises(ValueError):
        cf.CoordinateFrame(name='custom_frame',
                           axes_type=("SPATIAL",),
                           naxes=1, axes_order=(0,),
                           axes_names=("x", "y"))
    frame = cf.CoordinateFrame(name='custom_frame', axes_type=("SPATIAL",), axes_order=(0,), axes_names="x", naxes=1)
    assert frame.naxes == 1
    assert frame.axes_names == ("x",)

    frame.coordinate_to_quantity(1, 2)
