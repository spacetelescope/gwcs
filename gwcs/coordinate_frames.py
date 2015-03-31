# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Defines coordinate frames and ties them to data axes.
"""
from __future__ import division, print_function

import numpy as np
from astropy import time
from astropy import units as u
from astropy import utils as astutil
from astropy import coordinates as coord
from astropy.coordinates import (BaseCoordinateFrame, FrameAttribute,
                                 RepresentationMapping)

from . import spectral_builtin_frames
from .spectral_builtin_frames import *
from .representation import *


__all__ = ['DetectorFrame', 'CelestialFrame', 'SpectralFrame',
           'CompositeFrame', 'FocalPlaneFrame']


STANDARD_REFERENCE_POSITION = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
                               "TOPOCENTER", "LSR", "LSRK", "LSRD",
                               "GALACTIC_CENTER", "MOON", "LOCAL_GROUP_CENTER"]


class CoordinateFrame(object):

    """
    Base class for CoordinateFrames

    Parameters
    ----------
    naxes : int
        Number of axes.
    axes_order : tuple
        A mapping of inputs to coordinate frame axes order.
    reference_frame : astropy.coordinates.builtin_frames or spectral_builtin_frames
        Reference frame (see subclasses).
    reference_position : str or tuple
        Reference position - one of `STANDARD_REFERENCE_POSITION` or a tuple of floats
    unit : list of astropy.units.Unit
        Unit for each axis.
    axes_names : list
        Names of the axes in this frame, in the order of axes_order.
    name : str
        Name of this frame.
    """

    def __init__(self, naxes, axes_order=(0, 1), reference_frame=None, reference_position=None,
                 unit=None, axes_names=None, name=None):
        """ Initialize a frame"""
        self._axes_order = axes_order
        # map data axis into frame axes - 0-based
        self._naxes = naxes

        if unit is not None:
            if astutil.isiterable(unit):
                if len(unit) != naxes:
                    raise ValueError("Number of units does not match number of axes.")
                else:
                    self._unit = [u.Unit(au) for au in unit]
            else:
                self._unit = [u.Unit(unit)]
        else:
            self._unit = reference_frame.representation_component_units.values()

        if axes_names is not None and astutil.isiterable(axes_names):
            if len(axes_names) != naxes:
                raise ValueError("Number of axes names does not match number of axes.")
        else:
            axes_names = reference_frame.representation_component_names.values()
        self._axes_names = axes_names

        self._reference_frame = reference_frame

        if name is None:
            try:
                self._name = reference_frame.name
            except AttributeError:
                self._name = repr(reference_frame)
        else:
            self._name = name

        if reference_position is not None:
            self._reference_position = reference_position
        else:
            try:
                self._reference_position = reference_frame.reference_position
            except AttributeError:
                self._reference_position = None

        super(CoordinateFrame, self).__init__()

    def __repr__(self):
        '''
        if self._name is not None:
            return self._name
        else:
            return ""
        '''
        fmt = "<{0}({1}, axes_order={2}, reference_frame={3}, reference_position{4}," \
            "unit={5}, axes_names={6}, name={7})>".format(
                self.__class__.__name__, self.naxes, self.axes_order,
                self.reference_frame, self.reference_position, self.unit,
                self.axes_names, self.name)
        return fmt

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return self.__class__.__name__

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = name

    @property
    def naxes(self):
        return self._naxes

    @property
    def unit(self):
        return self._unit

    @property
    def axes_names(self):
        return self._axes_names

    @property
    def reference_frame(self):
        return self._reference_frame

    @property
    def reference_position(self):
        try:
            return self._reference_position
        except AttributeError:
            return None

    @property
    def axes_order(self):
        return self._axes_order

    def transform_to(self, other):
        """
        Transform from the current reference system to other
        """
        raise NotImplementedError("Subclasses should implement this")

    def world_coordinates(self, *args):
        """ Create world coordinates object"""
        raise NotImplementedError("Subclasses may implement this")


class CelestialFrame(CoordinateFrame):

    """
    Celestial Frame Representation

    Parameters
    ----------
    reference_frame : astropy.coordinates.builtin_frames instance
        A reference frame.
    axes_order : tuple
        A mapping of inputs to coordinate frame axes order.
    reference_position : str
        Reference position.
    unit : str or units.Unit instance or iterable of those
        Units on axes.

    """

    def __init__(self, reference_frame, axes_order=(0, 1), reference_position=None,
                 unit=[u.degree, u.degree], name=None):
        reference_position = 'Barycenter'
        if reference_frame.name.upper() in coord.builtin_frames.__all__:
            axes_names = reference_frame.representation_component_names.keys()[:2]
        super(CelestialFrame, self).__init__(naxes=2, reference_frame=reference_frame,
                                             unit=unit, axes_order=axes_order,
                                             reference_position=reference_position,
                                             axes_names=axes_names, name=name)

    def world_coordinates(self, lon, lat):
        """
        Create a SkyCoord object.

        Parameters
        ----------
        lon, lat : float
            longitude and latitude
        """
        return coord.SkyCoord(lon, lat, unit=self.unit, frame=self._reference_frame)

    def __repr__(self):

        fmt = "<CelestialFrame(reference_frame={0}, axes_order={1}, reference_position={2}, \
        unit={3}, name={4})>".format(
            self.reference_frame, self.axes_order,
            self.reference_position, self.unit, self.name)
        return fmt

    def transform_to(self, lat, lon, other):
        """
        Transform from the current reference frame to other.

        Parameters
        ----------
        lon, lat : float
            longitude and latitude
        other : str or `BaseCoordinateFrame` class
            The frame to transform this coordinate into.
        """
        return self.world_coordinates(lon, lat).transform_to(other)


class SpectralFrame(CoordinateFrame):

    """
    Represents Spectral Frame

    Parameters
    ----------
    reference_frame : astropy.coordinates.BaseCoordinateFrame
        One of spectral_builtin_frame.
    axes_order : tuple
        A mapping of inputs to coordinate frame axes order.
    unit : str or units.Unit instance
        Spectral unit.
    axes_names : str
        Spectral axis name.
    name : str
        Name for this frame.

    """

    def __init__(self, reference_frame, axes_order=(0,), unit=None, axes_names=None, name=None):
        super(SpectralFrame, self).__init__(naxes=1, reference_frame=reference_frame,
                                            axes_order=axes_order,
                                            axes_names=axes_names,
                                            unit=unit, name=name)

    def __repr__(self):
        fmt = "<SpectralFrame(reference_frame={0}, unit={1}, axes_names={2}, axes_order={3}, name={4})>".format(
            self.reference_frame, self.unit,
            self.axes_names, self.axes_order, self.name)
        return fmt

    def world_coordinates(self, value):
        """
        Create a SkyCoord object.

        Parameters
        ----------
        value : float
            Coordinate value.
        """
        return self.reference_frame.realize_frame(Cartesian1DRepresentation(value * u.Unit(self.unit[0])))

    def transform_to(self, x, other):
        """
        Transform from the current reference frame to other.

        Parameters
        ----------
        x : float or array-like
        other : str or `BaseCoordinateFrame` class
            The frame to transform this coordinate into.
        """
        return self.world_coordinates(x).transform_to(other)


class CompositeFrame(CoordinateFrame):

    """
    Represents one or more frames.

    Parameters
    ----------
    frames : list
        List of frames (TimeFrame, CelestialFrame, SpectralFrame, CoordinateFrame).
    name : str
        Name for this frame.

    """

    def __init__(self, frames, name=""):
        """
        naxes, axes_order=(0, 1), reference_frame=None, reference_position=None,
                 unit=None, axes_names=None, name=None):


        """
        self._frames = frames[:]
        naxes = sum([frame._naxes for frame in self._frames])
        unit = []
        axes_order = []
        axes_names = []
        for frame in frames:
            unit.extend(frame.unit)
            axes_order.extend(frame.axes_order)
            axes_names.extend(frame.axes_names)
        super(CompositeFrame, self).__init__(naxes, axes_order=axes_order,
                                             unit=unit, axes_names=axes_names,
                                             name=name)

    @property
    def frames(self):
        return self._frames

    def __repr__(self):
        return repr(self.frames)

    def world_coordinates(self, *args):
        """
        Create world coordinates.
        """
        if len(args) != self.naxes:
            raise TypeError("Expected {0} arguments ({1} given)".format(len(args), self.naxes))
        result = ()
        n = 0
        for frame in self.frames:
            result += (frame.world_coordinates(*args[n: frame.naxes + n]), )
            n += frame.naxes
        return result


class Detector(BaseCoordinateFrame):
    default_representation = Cartesian2DRepresentation
    frame_specific_representation_info = {
        'cartesian2d': [RepresentationMapping('x', 'x', u.pixel),
                        RepresentationMapping('y', 'y', u.pixel)]
    }


class DetectorFrame(CoordinateFrame):

    """
    Represents a Cartesian coordinate system on a detector.
    """

    def __init__(self, axes_order=(0, 1), name='detector', reference_position='Local'):
        axes_names = ['x', 'y']
        super(DetectorFrame, self).__init__(2, name=name, axes_names=axes_names,
                                            reference_frame=Detector(),
                                            reference_position=reference_position)

    def __repr__(self):
        fmt = "<DetectorFrame(axes_order={0}, name={1}, reference_position={2}, \
        axes_names={3}".format(self.axes_order, self.name, self.reference_position, self.axes_names)
        return fmt


class Focal(BaseCoordinateFrame):
    default_representation = Cartesian2DRepresentation
    frame_specific_representation_info = {
        'cartesian2d': [RepresentationMapping('x', 'x', u.pixel),
                        RepresentationMapping('y', 'y', u.pixel)]
    }


class FocalPlaneFrame(CoordinateFrame):

    def __init__(self, reference_pixel=[0., 0.], unit=[u.pixel, u.pixel], name='focal_plane',
                 reference_position="Local", axes_names=None):
        super(FocalPlaneFrame, self).__init__(2, reference_frame=Focal(), reference_position=reference_position,
                                              unit=unit, name=name, axes_names=axes_names)
        self._reference_pixel = reference_pixel


class WorldCoordinates(object):
    pass
