# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

from astropy import units as u
from astropy import coordinates as coo
from .. import coordinate_frames as cf
from .. import spectral_builtin_frames

icrs = coo.ICRS()
fk5 = coo.FK5()
cel1 = cf.CelestialFrame(icrs)
cel2 = cf.CelestialFrame(fk5)

freq = spectral_builtin_frames.Frequency()
wave = spectral_builtin_frames.Wavelength()
spec1 = cf.SpectralFrame(freq)
spec2 = cf.SpectralFrame(wave)

comp1 = cf.CompositeFrame([cel1, spec1])
comp2 = cf.CompositeFrame([cel2, spec2])
comp = cf.CompositeFrame([comp1, comp2])

def test_units():
    assert(comp1.unit == [u.deg, u.deg, u.Hz])
    assert(comp2.unit == [u.deg, u.deg, u.m])
    assert(comp.unit == [u.deg, u.deg, u.Hz, u.deg, u.deg, u.m])

def test_transform_to():
    spec = cf.SpectralFrame(wave, unit=u.micron)
    q = spec.transform_to(5, freq)
    assert(isinstance(q, spectral_builtin_frames.Frequency))
    assert(q.freq == 59958491600000.01 * u.Hz)
