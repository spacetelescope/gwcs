# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy import coordinates as coord
import pytest

from .. import coordinate_frames as cf


coord_frames = coord.builtin_frames.__all__[:]
# Need to write a better test, using a dict {coord_frame: input_parameters}
# For now remove OffsetFrame, issue #55
try:
    coord_frames.remove("SkyOffsetFrame")
except ValueError as ve:
    pass


icrs = cf.CelestialFrame(reference_frame=coord.ICRS(), axes_order=(0, 1))
detector = cf.Frame2D(name='detector', axes_order=(0, 1))
focal = cf.Frame2D(name='focal', axes_order=(0, 1), unit=(u.m, u.m))

spec1 = cf.SpectralFrame(name='freq', unit=[u.Hz, ], axes_order=(2, ))
spec2 = cf.SpectralFrame(name='wave', unit=[u.m, ], axes_order=(2, ), axes_names=('lambda', ))

comp1 = cf.CompositeFrame([icrs, spec1])
comp2 = cf.CompositeFrame([focal, spec2])
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


@pytest.mark.parametrize(('frame'), coord_frames)
def test_attributes(frame):
    """
    Test getting default values for  CoordinateFrame attributes from reference_frame.
    """
    cel = cf.CelestialFrame(reference_frame=getattr(coord, frame)())
    assert(len(cel.axes_names) == len(cel.axes_type) == len(cel.unit) ==
           len(cel.axes_order) == cel.naxes)
