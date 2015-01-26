# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy import units as u
from astropy import coordinates as coo
from .. import coordinate_frames as cs
from .. import spectral_builtin_frames

icrs=coo.ICRS()
fk5 = coo.FK5()
cel1 = cs.CelestialFrame(icrs)
cel2 = cs.CelestialFrame(fk5)

freq=spectral_builtin_frames.Frequency()
wave=spectral_builtin_frames.Wavelength()
spec1 = cs.SpectralFrame(freq)
spec2 = cs.SpectralFrame(wave)

comp1 = cs.CompositeFrame([cel1, spec1])
comp2 = cs.CompositeFrame([cel2, spec2])
comp = cs.CompositeFrame([comp1, comp2])

def test_units():
    assert(comp1.unit == [u.deg, u.deg, u.Hz])
    assert(comp2.unit == [u.deg, u.deg, u.m])
    assert(comp.unit == [u.deg, u.deg, u.Hz, u.deg, u.deg, u.m])

def test_transform_to():
    spec = cs.SpectralFrame(wave, unit=u.micron)
    q = spec.transform_to(5, freq)
    assert(isinstance(q, spectral_builtin_frames.Frequency))
    assert(q.freq == 59958491600000.01 * u.Hz)