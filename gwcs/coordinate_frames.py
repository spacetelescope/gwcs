# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Defines coordinate frames and ties them to data axes.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from astropy import units as u
from astropy import utils as astutil
from astropy import coordinates as coord
import six


__all__ = ['Frame2D', 'CelestialFrame', 'SpectralFrame', 'CompositeFrame',
           'CoordinateFrame']


STANDARD_REFERENCE_FRAMES = [frame.upper() for frame in coord.builtin_frames.__all__]

STANDARD_REFERENCE_POSITION = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
                               "TOPOCENTER", "LSR", "LSRK", "LSRD",
                               "GALACTIC_CENTER", "LOCAL_GROUP_CENTER"]


class CoordinateFrame(object):

    """
    Base class for CoordinateFrames.

    Parameters
    ----------
    naxes : int
        Number of axes.
    axes_type : str
        One of ["SPATIAL", "SPECTRAL", "TIME"]
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        Reference frame (usually used with output_frame to convert to world coordinate objects).
    reference_position : str
        Reference position - one of `STANDARD_REFERENCE_POSITION`
    unit : list of astropy.units.Unit
        Unit for each axis.
    axes_names : list
        Names of the axes in this frame.
    name : str
        Name of this frame.
    """

    def __init__(self, naxes, axes_type, axes_order, reference_frame=None,
                 reference_position=None, unit=None, axes_names=None,
                 name=None):
        self._naxes = naxes
        self._axes_order = tuple(axes_order)
        if isinstance(axes_type, six.string_types):
            self._axes_type = (axes_type,)
        else:
            self._axes_type = tuple(axes_type)
        self._reference_frame = reference_frame
        if unit is not None:
            if astutil.isiterable(unit):
                unit = tuple(unit)
            else:
                unit = (unit,)
            if len(unit) != naxes:
                raise ValueError("Number of units does not match number of axes.")
            else:
                self._unit = tuple([u.Unit(au) for au in unit])
        if axes_names is not None:
            if isinstance(axes_names, six.string_types):
                axes_names = (axes_names,)
            else:
                axes_names = tuple(axes_names)
            if len(axes_names) != naxes:
                raise ValueError("Number of axes names does not match number of axes.")
        else:
            axes_names = tuple([""] * naxes)
        self._axes_names = axes_names
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

        if reference_position is not None:
            self._reference_position = reference_position
        else:
            self._reference_position = None

        super(CoordinateFrame, self).__init__()

    def __repr__(self):
        fmt = '<{0}(name="{1}", unit={2}, axes_names={3}, axes_order={4}'.format(
            self.__class__.__name__, self.name,
            self.unit, self.axes_names, self.axes_order)
        if self.reference_position is not None:
            fmt += ', reference_position="{0}"'.format(self.reference_position)
        if self.reference_frame is not None:
            fmt += ", reference_frame={0}".format(self.reference_frame)
        fmt += ")>"
        return fmt

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return self.__class__.__name__

    @property
    def name(self):
        """ A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val):
        """ A custom name of this frame."""
        self._name = val

    @property
    def naxes(self):
        """ The number of axes intheis frame."""
        return self._naxes

    @property
    def unit(self):
        """The unit of this frame."""
        return self._unit

    @property
    def axes_names(self):
        """ Names of axes in the frame."""
        return self._axes_names

    @property
    def axes_order(self):
        """ A tuple of indices which map inputs to axes."""
        return self._axes_order

    @property
    def reference_frame(self):
        return self._reference_frame

    @property
    def reference_position(self):
        try:
            return self._reference_position
        except AttributeError:
            return None

    def input_axes(self, start_frame=None):
        """
        Computes which axes in `start_frame` contribute to each axis in the current frame.

        Parameters
        ----------
        start_frame : ~gwcs.coordinate_frames.CoordinateFrame
            A frame in the WCS pipeline
            The transform between start_frame and the current frame is used to compute the
            mapping inputs: outputs.
        """

        sep = self._separable(start_frame)
        inputs = []
        for ax in self.axes_order:
            inputs.append(list(sep[ax].nonzero()[0]))
        return inputs

    @property
    def axes_type(self):
        """ Type of this frame : 'SPATIAL', 'SPECTRAL', 'TIME'. """
        return self._axes_type

    def coordinates(self, *args):
        """ Create world coordinates object"""
        raise NotImplementedError("Subclasses may implement this")


class CelestialFrame(CoordinateFrame):

    """
    Celestial Frame Representation

    Parameters
    ----------
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        A reference frame.
    reference_position : str
        Reference position.
    unit : str or units.Unit instance or iterable of those
        Units on axes.
    axes_names : list
        Names of the axes in this frame.
    name : str
        Name of this frame.
    """

    def __init__(self, axes_order=None, reference_frame=None,
                 unit=None, axes_names=None,
                 name=None):
        naxes = 2
        if reference_frame is not None:
            if reference_frame.name.upper() in STANDARD_REFERENCE_FRAMES:
                _axes_names = list(reference_frame.representation_component_names.values())
                if 'distance' in _axes_names:
                    _axes_names.remove('distance')
                if axes_names is None:
                    axes_names = _axes_names
                naxes = len(_axes_names)
                _unit = list(reference_frame.representation_component_units.values())
                if unit is None and _unit:
                    unit = _unit

        if axes_order is None:
            axes_order = tuple(range(naxes))
        if unit is None:
            unit = tuple([u.degree]  * naxes)
        axes_type = ['SPATIAL'] * naxes
        super(CelestialFrame, self).__init__(naxes=naxes, axes_type=axes_type,
                                             axes_order=axes_order,
                                             reference_frame=reference_frame,
                                             unit=unit,
                                             axes_names=axes_names,
                                             name=name)

    def coordinates(self, *args):
        """
        Create a SkyCoord object.

        Parameters
        ----------
        args : float
            inputs to wcs.input_frame
        """
        # Reorder axes if necesary.
        try:
            return coord.SkyCoord(*args, unit=self.unit, frame=self._reference_frame)
        except:
            raise


class SpectralFrame(CoordinateFrame):
    """
    Represents Spectral Frame

    Parameters
    ----------
    axes_order : tuple or int
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        Reference frame (usually used with output_frame to convert to world coordinate objects).
    unit : str or units.Unit instance
        Spectral unit.
    axes_names : str
        Spectral axis name.
    name : str
        Name for this frame.
    """

    def __init__(self, axes_order=(0,), reference_frame=None, unit=None,
                 axes_names=None, name=None, reference_position=None):
        super(SpectralFrame, self).__init__(naxes=1, axes_type="SPECTRAL", axes_order=axes_order,
                                            axes_names=axes_names, reference_frame=reference_frame,
                                            unit=unit, name=name,
                                            reference_position=reference_position)

    def coordinates(self, *args):
        if np.isscalar(args):
            return args * self.unit[0]
        else:
            return args[0] * self.unit[0]


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

    def __init__(self, frames, name=None):
        self._frames = frames[:]
        naxes = sum([frame._naxes for frame in self._frames])
        axes_type = list(range(naxes))
        unit = list(range(naxes))
        axes_names = list(range(naxes))
        axes_order = []
        for frame in frames:
            axes_order.extend(frame.axes_order)
        for frame in frames:
            for ind, axtype, un, n in zip(frame.axes_order, frame.axes_type,
                                          frame.unit, frame.axes_names):
                axes_type[ind] = axtype
                axes_names[ind] = n
                unit[ind] = un
        if len(np.unique(axes_order)) != len(axes_order):
            raise ValueError("Incorrect numbering of axes, "
                             "axes_order should contain unique numbers, "
                             "got {}.".format(axes_order))
        super(CompositeFrame, self).__init__(naxes, axes_type=axes_type,
                                             axes_order=axes_order,
                                             unit=unit, axes_names=axes_names,
                                             name=name)

    @property
    def frames(self):
        return self._frames

    def __repr__(self):
        return repr(self.frames)

    def coordinates(self, *args):
        coo  = []
        for frame in self.frames:
            fargs = [args[i] for i in frame.axes_order]
            coo.append(frame.coordinates(*fargs))
        return coo


class Frame2D(CoordinateFrame):

    """
    A 2D coordinate frame.

    Parameters
    ----------
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.
    unit : list of astropy.units.Unit
        Unit for each axis.
    axes_names : list
        Names of the axes in this frame.
    name : str
        Name of this frame.
    """

    def __init__(self, axes_order=(0, 1), unit=(u.pix, u.pix), axes_names=('x', 'y'),
                 name=None):

        super(Frame2D, self).__init__(2, ["SPATIAL", "SPATIAL"], axes_order, name=name,
                                      axes_names=axes_names, unit=unit)

    def coordinates(self, *args):
        args = [args[i] for i in self.axes_order]
        coo = tuple([arg * un for arg, un in zip(args, self.unit)])
        return coo
