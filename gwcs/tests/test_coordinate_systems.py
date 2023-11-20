# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import logging
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy import coordinates as coord
from astropy.tests.helper import assert_quantity_allclose
from astropy.modeling import models as m
from astropy.wcs.wcsapi.fitswcs import CTYPE_TO_UCD1
from astropy.coordinates import StokesCoord, SpectralCoord

from .. import WCS
from .. import coordinate_frames as cf

from astropy.wcs.wcsapi.high_level_api import values_to_high_level_objects, high_level_objects_to_values
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
spec2 = cf.SpectralFrame(name='wave', unit=[u.m, ], axes_order=(2, ), axes_names=('lambda',))
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


@pytest.fixture(autouse=True, scope="module")
def serialized_classes():
    """
    In the rest of this test file we are passing the CoordinateFrame object to
    astropy helper functions as if they were a low level WCS object.

    This little patch means that this works.
    """
    cf.CoordinateFrame.serialized_classes = False
    yield
    del cf.CoordinateFrame.serialized_classes


def test_units():
    assert(comp1.unit == (u.deg, u.deg, u.Hz))
    assert(comp2.unit == (u.m, u.m, u.m))
    assert(comp3.unit == (u.deg, u.deg, u.J))
    assert(comp4.unit == (u.deg, u.deg, u.pix))
    assert(comp5.unit == (u.deg, u.deg, u.m/u.s))
    assert(comp.unit == (u.deg, u.deg, u.Hz, u.m))


# These two functions fake the old methods on CoordinateFrame to reduce the
# amount of refactoring that needed doing in these tests.
def coordinates(*inputs, frame):
    results = values_to_high_level_objects(*inputs, low_level_wcs=frame)
    if isinstance(results, list) and len(results) == 1:
        return results[0]
    return results


def coordinate_to_quantity(*inputs, frame):
    results = high_level_objects_to_values(*inputs, low_level_wcs=frame)
    results = [r<<unit for r, unit in zip(results, frame.unit)]
    return results


@pytest.mark.parametrize('inputs', inputs2)
def test_coordinates_spatial(inputs):
    sky_coo = coordinates(*inputs, frame=icrs)
    assert isinstance(sky_coo, coord.SkyCoord)
    assert_allclose((sky_coo.ra.value, sky_coo.dec.value), inputs)
    focal_coo = coordinates(*inputs, frame=focal)
    assert_allclose([coo.value for coo in focal_coo], inputs)
    assert [coo.unit for coo in focal_coo] == [u.m, u.m]


@pytest.mark.parametrize('inputs', inputs1)
def test_coordinates_spectral(inputs):
    wave = coordinates(inputs, frame=spec2)
    assert_allclose(wave.value, inputs)
    assert wave.unit == 'meter'
    assert isinstance(wave, u.Quantity)


@pytest.mark.parametrize('inputs', inputs3)
def test_coordinates_composite(inputs):
    frame = cf.CompositeFrame([icrs, spec2])
    result = coordinates(*inputs, frame=frame)
    assert isinstance(result[0], coord.SkyCoord)
    assert_allclose((result[0].ra.value, result[0].dec.value), inputs[:2])
    assert_allclose(result[1].value, inputs[2])


def test_coordinates_composite_order():
    time = cf.TemporalFrame(Time("2011-01-01T00:00:00"), name='time', unit=[u.s, ], axes_order=(0, ))
    dist = cf.CoordinateFrame(name='distance', naxes=1,
                              axes_type=["SPATIAL"], unit=[u.m, ], axes_order=(1, ))
    frame = cf.CompositeFrame([time, dist])
    result = coordinates(0, 0, frame=frame)
    assert result[0] == Time("2011-01-01T00:00:00")
    assert u.allclose(result[1], 0*u.m)


def test_bare_baseframe():
    # This is a regression test for the following call:
    frame = cf.CoordinateFrame(1, "SPATIAL", (0,), unit=(u.km,))
    quantity = coordinate_to_quantity(1*u.m, frame=frame)
    assert u.allclose(quantity, 1*u.m)

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
                               axes_order=(0, 1),
                               unit=(u.deg, u.arcsec))
    assert frame.name == 'CustomFrame'
    frame.name = "DeLorean"
    assert frame.name == 'DeLorean'

    q1, q2 = coordinate_to_quantity(12 * u.deg, 3 * u.arcsec, frame=frame)
    assert_quantity_allclose(q1, 12 * u.deg)
    assert_quantity_allclose(q2, 3 * u.arcsec)

    q1, q2 = coordinate_to_quantity(*(12 * u.deg, 3 * u.arcsec), frame=frame)
    assert_quantity_allclose(q1, 12 * u.deg)
    assert_quantity_allclose(q2, 3 * u.arcsec)


def test_temporal_relative():
    t = cf.TemporalFrame(reference_frame=Time("2018-01-01T00:00:00"), unit=u.s)
    assert coordinates(10, frame=t) == Time("2018-01-01T00:00:00") + 10 * u.s
    assert coordinates(10 * u.s, frame=t) == Time("2018-01-01T00:00:00") + 10 * u.s

    a = coordinates((10, 20), frame=t)
    assert a[0] == Time("2018-01-01T00:00:00") + 10 * u.s
    assert a[1] == Time("2018-01-01T00:00:00") + 20 * u.s

    t = cf.TemporalFrame(reference_frame=Time("2018-01-01T00:00:00"))
    assert coordinates(10 * u.s, frame=t) == Time("2018-01-01T00:00:00") + 10 * u.s
    assert coordinates(TimeDelta(10, format='sec'), frame=t) == Time("2018-01-01T00:00:00") + 10 * u.s

    a = coordinates((10, 20) * u.s, frame=t)
    assert a[0] == Time("2018-01-01T00:00:00") + 10 * u.s
    assert a[1] == Time("2018-01-01T00:00:00") + 20 * u.s


def test_temporal_absolute():
    t = cf.TemporalFrame(reference_frame=Time([], format='isot'))
    assert coordinates("2018-01-01T00:00:00", frame=t) == Time("2018-01-01T00:00:00")

    a = coordinates(("2018-01-01T00:00:00", "2018-01-01T00:10:00"), frame=t)
    assert a[0] == Time("2018-01-01T00:00:00")
    assert a[1] == Time("2018-01-01T00:10:00")

    t = cf.TemporalFrame(reference_frame=Time([], scale='tai', format='isot'))
    assert coordinates("2018-01-01T00:00:00", frame=t) == Time("2018-01-01T00:00:00", scale='tai')


@pytest.mark.parametrize('inp', [
    (coord.SkyCoord(10 * u.deg, 20 * u.deg, frame=coord.ICRS),),
    # This is the same as 10,20 in ICRS
    (coord.SkyCoord(119.26936774, -42.79039286, unit=u.deg, frame='galactic'),)
])
def test_coordinate_to_quantity_celestial(inp):
    cel = cf.CelestialFrame(reference_frame=coord.ICRS(), axes_order=(0, 1))

    lon, lat = coordinate_to_quantity(*inp, frame=cel)
    assert_quantity_allclose(lon, 10 * u.deg)
    assert_quantity_allclose(lat, 20 * u.deg)

    with pytest.raises(ValueError):
        coordinate_to_quantity(10 * u.deg, 2 * u.deg, 3 * u.deg, frame=cel)

    with pytest.raises(ValueError):
        coordinate_to_quantity((1, 2), frame=cel)


@pytest.mark.parametrize('inp', [
    (SpectralCoord(100 * u.nm),),
    (SpectralCoord(0.1 * u.um),),
])
def test_coordinate_to_quantity_spectral(inp):
    spec = cf.SpectralFrame(unit=u.nm, axes_order=(1, ))
    wav = coordinate_to_quantity(*inp, frame=spec)
    assert_quantity_allclose(wav, 100 * u.nm)


@pytest.mark.parametrize('inp', [
    (Time("2011-01-01T00:00:10"),),
])
def test_coordinate_to_quantity_temporal(inp):
    temp = cf.TemporalFrame(reference_frame=Time("2011-01-01T00:00:00"), unit=u.s)

    t = coordinate_to_quantity(*inp, frame=temp)

    assert_quantity_allclose(t, 10 * u.s)


@pytest.mark.parametrize('inp', [
    (SpectralCoord(211 * u.AA), Time("2011-01-01T00:00:00"), coord.SkyCoord(0, 0, unit=u.arcsec)),
])
def test_coordinate_to_quantity_composite(inp):
    # Composite
    wave_frame = cf.SpectralFrame(axes_order=(0, ), unit=u.AA)
    time_frame = cf.TemporalFrame(
        axes_order=(1, ), unit=u.s, reference_frame=Time("2011-01-01T00:00:00"))
    sky_frame = cf.CelestialFrame(axes_order=(2, 3), reference_frame=coord.ICRS())

    comp = cf.CompositeFrame([wave_frame, time_frame, sky_frame])

    coords = coordinate_to_quantity(*inp, frame=comp)

    expected = (211 * u.AA, 0 * u.s, 0 * u.arcsec, 0 * u.arcsec)
    for output, exp in zip(coords, expected):
        assert_quantity_allclose(output, exp)


def test_coordinate_to_quantity_composite_split():
    inp = (
        SpectralCoord(211 * u.AA),
        coord.SkyCoord(0, 0, unit=u.arcsec),
        Time("2011-01-01T00:00:00"),
    )

    # Composite
    wave_frame = cf.SpectralFrame(axes_order=(1, ), unit=u.AA)
    sky_frame = cf.CelestialFrame(axes_order=(2, 0), reference_frame=coord.ICRS())
    time_frame = cf.TemporalFrame(
        axes_order=(3,), unit=u.s, reference_frame=Time("2011-01-01T00:00:00"))

    comp = cf.CompositeFrame([wave_frame, sky_frame, time_frame])

    coords = coordinate_to_quantity(*inp, frame=comp)

    expected = (0 * u.arcsec, 211 * u.AA, 0 * u.arcsec, 0 * u.s)
    for output, exp in zip(coords, expected):
        assert_quantity_allclose(output, exp)


def test_stokes_frame():
    sf = cf.StokesFrame()

    assert coordinates(1, frame=sf) == 'I'
    assert coordinates(1 * u.one, frame=sf) == 'I'
    assert coordinate_to_quantity(StokesCoord('I'), frame=sf) == 1 * u.one
    assert coordinate_to_quantity(StokesCoord(1), frame=sf) == 1 * u.one


def test_coordinate_to_quantity_frame2d_composite():
    inp = (SpectralCoord(211 * u.AA), Time("2011-01-01T00:00:00"), 0 * u.one, 0 * u.one)
    wave_frame = cf.SpectralFrame(axes_order=(0, ), unit=u.AA)
    time_frame = cf.TemporalFrame(
        axes_order=(1, ), unit=u.s, reference_frame=Time("2011-01-01T00:00:00"))

    frame2d = cf.Frame2D(name="intermediate", axes_order=(2, 3), unit=(u.one, u.one))

    comp = cf.CompositeFrame([wave_frame, time_frame, frame2d])

    coords = coordinate_to_quantity(*inp, frame=comp)

    expected = (211 * u.AA, 0 * u.s, 0 * u.one, 0 * u.one)
    for output, exp in zip(coords, expected):
        assert_quantity_allclose(output, exp)


def test_coordinate_to_quantity_frame_2d():
    frame = cf.Frame2D(unit=(u.one, u.arcsec))
    inp = (1 * u.one, 2 * u.arcsec)
    expected = (1 * u.one, 2 * u.arcsec)
    result = coordinate_to_quantity(*inp, frame=frame)
    for output, exp in zip(result, expected):
        assert_quantity_allclose(output, exp)


def test_coordinate_to_quantity_error():
    frame = cf.Frame2D(unit=(u.one, u.arcsec))
    with pytest.raises(ValueError):
        coordinate_to_quantity(1, frame=frame)

    with pytest.raises(ValueError):
        coordinate_to_quantity((1, 1), 2, frame=frame)

    frame = cf.TemporalFrame(reference_frame=Time([], format='isot'), unit=u.s)
    with pytest.raises(ValueError):
        coordinate_to_quantity(1, frame=frame)


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
                             axis_physical_types='em.wavenumber', unit=u.Unit(1))
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
    frame = cf.CoordinateFrame(
        name='custom_frame',
        axes_type=("SPATIAL",),
        axes_order=(0,),
        axes_names="x",
        naxes=1
    )
    assert frame.naxes == 1
    assert frame.axes_names == ("x",)

    coordinate_to_quantity(1*u.one, frame=frame)


def test_ucd1_to_ctype_not_out_of_sync(caplog):
    """
    Test that our code is not out-of-sync with ``astropy``'s definition of
    ``CTYPE_TO_UCD1`` and our dictionary of allowed duplicates.

    If this test is failing, update ``coordinate_frames._ALLOWED_UCD_DUPLICATES``
    dictionary with new types defined in ``astropy``'s ``CTYPE_TO_UCD1``.

    """
    cf._ucd1_to_ctype_name_mapping(
        ctype_to_ucd=CTYPE_TO_UCD1,
        allowed_ucd_duplicates=cf._ALLOWED_UCD_DUPLICATES
    )

    assert len(caplog.record_tuples) == 0


def test_ucd1_to_ctype(caplog):
    new_ctype_to_ucd = {
        'RPT1': 'new.repeated.type',
        'RPT2': 'new.repeated.type',
        'RPT3': 'new.repeated.type',
    }

    ctype_to_ucd = dict(**CTYPE_TO_UCD1, **new_ctype_to_ucd)

    inv_map = cf._ucd1_to_ctype_name_mapping(
        ctype_to_ucd=ctype_to_ucd,
        allowed_ucd_duplicates=cf._ALLOWED_UCD_DUPLICATES
    )

    assert caplog.record_tuples[-1][1] == logging.WARNING and \
            caplog.record_tuples[-1][2].startswith(
                "Found unsupported duplicate physical type"
            )

    for k, v in cf._ALLOWED_UCD_DUPLICATES.items():
        assert inv_map.get(k, '') == v

    for k, v in inv_map.items():
        assert ctype_to_ucd[v] == k

    assert inv_map['new.repeated.type'] in new_ctype_to_ucd
