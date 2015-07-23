# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Defines coordinate frames and ties them to data axes.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
from astropy import time
from astropy import units as u
from astropy import utils as astutil
from astropy import coordinates as coord
from astropy.extern import six

from . import utils as gwutils


__all__ = ['Frame2D', 'CelestialFrame', 'SpectralFrame', 'CompositeFrame',
           'CoordinateFrame']


##STANDARD_REFERENCE_POSITION = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
##"TOPOCENTER", "LSR", "LSRK", "LSRD",
##"GALACTIC_CENTER", "MOON", "LOCAL_GROUP_CENTER"]


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
    wcsobj : ~gwcs.WCS
        Reference to the WCS object to which this frame belongs.
    """

    def __init__(self, naxes, axes_type, axes_order, reference_frame=None,
                 reference_position=None, unit=None, axes_names=None,
                 name=None, wcsobj=None):
        self._naxes = naxes
        self._axes_order = axes_order
        if isinstance(axes_type, six.string_types):
            self._axes_type = (axes_type,)
        else:
            self._axes_type = axes_type
        self._reference_frame = reference_frame
        if unit is not None:
            if astutil.isiterable(unit):
                if len(unit) != naxes:
                    raise ValueError("Number of units does not match number of axes.")
                else:
                    self._unit = [u.Unit(au) for au in unit]
            else:
                self._unit = [u.Unit(unit)]
        else:
            if self.reference_frame is not None:
                self._unit = list(reference_frame.representation_component_units.values())

        if axes_names is not None:
            if astutil.isiterable(axes_names):
                if len(axes_names) != naxes:
                    raise ValueError("Number of axes names does not match number of axes.")
        else:
            if self.reference_frame is not None:
                axes_names = list(reference_frame.representation_component_names.values())
            else:
                axes_names = [None] * self._naxes
        self._axes_names = axes_names

        if name is None:
            self._name = self.__class__.__name__
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
        fmt = "<{0}({1}, reference_frame={2}, reference_position{3}," \
            "unit={4}, axes_names={5}, name={6})>".format(
                self.__class__.__name__, self.naxes,
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
        """ A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val):
        """ A custom name of this frame."""
        self._name = name

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

    def _set_wcsobj(self, obj):
        self._wcsobj = obj

    def _separable(self, start=None):
        """
        Computes the relationship of i nput and output axes.
        Returns an array of shape (transform.n_outputs, transform.n_inputs).
        Rows represent outputs, columns - inputs.
        Non-zero elements mean dependence of the output axis on the corresponding input axis.

        """
        if self._wcsobj is not None:
            if start is None:
                start = self._wcsobj.input_frame
            else:
                if not start in self._wcsobj.available_frames:
                    raise ValueError("Unrecognized frame {0}".format(start))
            transform = self._wcsobj.get_transform(start, self)
            sep_matrix = gwutils._separable(transform)
            return sep_matrix
        else:
            raise ValueError("A starting frame is needed to determine axes.")

    def is_separable(self, start_frame=None):
        """
        Computes the separability of axes.

        Returns a 1D boolean array of size frame.naxes where True means
        the axis is completely separable and False means the axis is nonseparable
        from at least one other axis.

        Parameters
        ----------
        start_frame : ~gwcs.coordinate_frames.CoordinateFrame
            A frame in the WCS pipeline
            The transform between start_frame and the current frame is used to compute the
            mapping inputs: outputs.
            If None the input_frame is used as start_frame.

        See Also
        --------
        input_axes : For each output axis return the input axes contributing to it.

        """
        if self._wcsobj is not None:
            if start_frame is None:
                start_frame = self._wcsobj.input_frame
            else:
                if not start_frame in self._wcsobj.available_frames:
                    raise ValueError("Unrecognized frame {0}".format(start))
            transform = self._wcsobj.get_transform(start_frame, self)
        else:
            raise ValueError("A starting frame is needed to determine separability of axes.")

        sep = gwutils.is_separable(transform)
        return [sep[ax] for ax in self.axes_order]


    def transform_to(self, other):
        """
        Transform from the current reference system to other
        """
        raise NotImplementedError("Subclasses should implement this")

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
    wcsobj : ~gwcs.WCS
        Reference to the WCS object to which this frame belongs.

    """

    def __init__(self, axes_order=(0, 1), reference_frame=None, reference_position=None,
                 unit=(u.degree, u.degree), axes_names=None,  name=None, wcsobj=None):
        reference_position = 'Barycenter'
        if reference_frame is not None:
            if reference_frame.name.upper() in coord.builtin_frames.__all__:
                axes_names = list(reference_frame.representation_component_names.keys())[:2]
        super(CelestialFrame, self).__init__(naxes=2, axes_type=["SPATIAL", "SPATIAL"],
                                             axes_order=axes_order,
                                             reference_frame=reference_frame,
                                             unit=unit,
                                             reference_position=reference_position,
                                             axes_names=axes_names, name=name, wcsobj=wcsobj)

    def coordinates(self, *args):
        """
        Create a SkyCoord object.

        Parameters
        ----------
        args : float
            inputs to wcs.input_frame
        """
        args = self._wcsobj(*args)
        args = [args[i] for i in self.axes_order]
        try:
            return coord.SkyCoord(*args, unit=self.unit, frame=self._reference_frame)
        except:
            raise

    def __repr__(self):

        fmt = "<CelestialFrame(reference_frame={0}, \
        unit={1}, name={2})>".format(self.reference_frame, self.unit, self.name)
        return fmt

    def transform_to(self, other, *args):
        """
        Transform from the current reference frame to other.

        Parameters
        ----------
        lon, lat : float
            longitude and latitude
        other : str or `BaseCoordinateFrame` class
            The frame to transform this coordinate into.
        """
        return self.coordinates(*args).transform_to(other)


class SpectralFrame(CoordinateFrame):
    """
    Represents Spectral Frame

    Parameters
    ----------
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        Reference frame (usually used with output_frame to convert to world coordinate objects).
    unit : str or units.Unit instance
        Spectral unit.
    axes_names : str
        Spectral axis name.
    name : str
        Name for this frame.
    wcsobj : ~gwcs.WCS
        Reference to the WCS object to which this frame belongs.
    """

    def __init__(self, axes_order=(0,), reference_frame=None, unit=None, axes_names=None, name=None, wcsobj=None):
        super(SpectralFrame, self).__init__(naxes=1, axes_type="SPECTRAL", axes_order=axes_order,
                                            axes_names=axes_names, reference_frame=reference_frame,
                                            unit=unit, name=name, wcsobj=wcsobj)

    def coordinates(self, *args):
        args = self._wcsobj(*args)

        if np.isscalar(args):
            return args * self.unit[0]
        else:
            if len(getattr(self._wcsobj, self._wcsobj.output_frame).axes_order) > 1:
                args = [args[i] for i in self.axes_order]
                return args[0] * self.unit[0]
            else:
                return args * self.unit[0]


    def transform_to(self, x, other_unit):
        return self.coordinates(x).to(other_unit, equivalencies=u.spectral())


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
            for ind, type, u, n in zip(frame.axes_order, frame.axes_type,
                                          frame.unit, frame.axes_names):
                axes_type[ind] = type
                axes_names[ind] = n
                unit[ind] = u
        super(CompositeFrame, self).__init__(naxes, axes_type=axes_type, axes_order=axes_order,
                                             unit=unit, axes_names=axes_names,
                                             name=name)

    @property
    def frames(self):
        return self._frames

    def __repr__(self):
        return repr(self.frames)

    def _set_wcsobj(self, obj):
        for frame in self.frames:
            frame._set_wcsobj(obj)
        self._wcsobj = obj

    def coordinates(self, *args):
        """
        Return the output of the forwrd_transform as quantities.

        Parameters
        ----------
        args : float
            inputs to the WCS in the input_coordinate_frame.
        """
        naxes = getattr(self._wcsobj, self._wcsobj.input_frame).naxes
        if len(args) != naxes:
            raise TypeError("Expected {0} arguments ({1} given)".format(naxes, len(args)))

        result = tuple([frame.coordinates(*args) for frame in self.frames])
        return result


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
    wcsobj : ~gwcs.WCS
        Reference to the WCS object to which this frame belongs.
    """

    def __init__(self, axes_order=(0, 1), unit=(u.pix, u.pix), axes_names=('x', 'y'),
                 name=None, wcsobj=None):

        super(Frame2D, self).__init__(2, "SPATIAL", axes_order, name=name,
                                      axes_names=axes_names,unit=unit, wcsobj=None)

    def coordinates(self, *args):
        args = self._wcsobj.get_transform(self._wcsobj.input_frame, self)(*args)
        args = [args[i] for i in self.axes_order]
        coo = tuple([arg * un for arg, un in zip(args, self.unit)])
        return coo
