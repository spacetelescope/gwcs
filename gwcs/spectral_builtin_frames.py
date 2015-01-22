# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function

from astropy import units as u
from astropy.units import equivalencies as eq
from astropy import coordinates as coo
from astropy.coordinates import (BaseCoordinateFrame, FrameAttribute,
                                 TimeFrameAttribute, RepresentationMapping,
                                 frame_transform_graph)
from .representation import *
#from astropy import constants

__all__ = ['Wavelength', 'Frequency', 'OpticalVelocity']


class Wavelength(BaseCoordinateFrame):
    default_representation = Cartesian1DRepresentation
    reference_position = FrameAttribute(default='BARYCENTER')
    frame_specific_representation_info = {
        'cartesian1d': [RepresentationMapping('x', 'lam', 'm')]
        }


class Frequency(BaseCoordinateFrame):
    default_representation = Cartesian1DRepresentation
    reference_position = FrameAttribute(default='BARYCENTER')
    frame_specific_representation_info = {
        'cartesian1d': [RepresentationMapping('x', 'freq', 'Hz')]
        }


class OpticalVelocity(BaseCoordinateFrame):
    default_representation = Cartesian1DRepresentation
    reference_position = FrameAttribute(default='BARYCENTER')
    rest = FrameAttribute()
    frame_specific_representation_info = {
        'cartesian1d': [RepresentationMapping('x', 'v', 'm/s')]
        }


@frame_transform_graph.transform(coo.FunctionTransform, Wavelength, Frequency)
def wave_to_freq(wavecoord, freqframe):
    return Frequency(wavecoord.lam.to(u.Hz, equivalencies=eq.spectral()))


@frame_transform_graph.transform(coo.FunctionTransform, Frequency, Wavelength)
def freq_to_wave(freqcoord, waveframe):
    return Wavelength(freqcoord.f.to(u.m, equivalencies=eq.spectral()))


@frame_transform_graph.transform(coo.FunctionTransform, Wavelength, OpticalVelocity)
def wave_to_velo(wavecoord, veloframe):
    return OpticalVelocity(wavecoord.lam.to(veloframe.representation_component_units.values()[0], equivalencies=eq.doppler_optical(veloframe.rest)))


@frame_transform_graph.transform(coo.FunctionTransform, Frequency, OpticalVelocity)
def freq_to_velo(freqcoord, veloframe):
    return OpticalVelocity(freqcoord.f.to(veloframe.representation_component_units.values()[0], equivalencies=eq.doppler_optical(veloframe.rest)))


