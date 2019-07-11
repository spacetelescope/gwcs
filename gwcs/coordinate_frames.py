# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Defines coordinate frames and ties them to data axes.
"""
import logging
import numpy as np

from astropy.utils.misc import isiterable
import astropy.time
from astropy import units as u
from astropy import utils as astutil
from astropy import coordinates as coord
from astropy.wcs.wcsapi.low_level_api import (validate_physical_types,
                                              VALID_UCDS)


__all__ = ['Frame2D', 'CelestialFrame', 'SpectralFrame', 'CompositeFrame',
           'CoordinateFrame', 'TemporalFrame']


STANDARD_REFERENCE_FRAMES = [frame.upper() for frame in coord.builtin_frames.__all__]

STANDARD_REFERENCE_POSITION = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
                               "TOPOCENTER", "LSR", "LSRK", "LSRD",
                               "GALACTIC_CENTER", "LOCAL_GROUP_CENTER"]


class CoordinateFrame:
    """
    Base class for Coordinate Frames.

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
                 name=None, axis_physical_types=None):
        self._naxes = naxes
        self._axes_order = tuple(axes_order)
        if isinstance(axes_type, str):
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
        else:
            self._unit = tuple(u.Unit("") for na in range(naxes))
        if axes_names is not None:
            if isinstance(axes_names, str):
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

        self._reference_position = reference_position

        if len(self._axes_type) != naxes:
            raise ValueError("Length of axes_type does not match number of axes.")
        if len(self._axes_order) != naxes:
            raise ValueError("Length of axes_order does not match number of axes.")

        super(CoordinateFrame, self).__init__()
        self._axis_physical_types = self._set_axis_physical_types(axis_physical_types)

    def _set_axis_physical_types(self, pht=None):
        """
        Set the physical type of the coordinate axes using VO UCD1+ v1.23 definitions.

        """
        if pht is not None:
            if isinstance(pht, str):
                pht = (pht,)
            elif not isiterable(pht):
                raise TypeError("axis_physical_types must be of type string or iterable of strings")
            if len(pht) != self.naxes:
                raise ValueError('"axis_physical_types" must be of length {}'.format(self.naxes))
            ph_type = []
            for axt in pht:
                if axt not in VALID_UCDS and not axt.startswith("custom:"):
                    ph_type.append("custom:{}".format(axt))
                else:
                    ph_type.append(axt)
            validate_physical_types(ph_type)
            return tuple(ph_type)

        if isinstance(self, CelestialFrame):
            if isinstance(self.reference_frame, coord.Galactic):
                ph_type = "pos.galactic.lon", "pos.galactic.lat"
            elif isinstance(self.reference_frame, (coord.GeocentricTrueEcliptic,
                                                   coord.GCRS,
                                                   coord.PrecessedGeocentric)):
                ph_type = "pos.bodyrc.lon", "pos.bodyrc.lat"
            elif isinstance(self.reference_frame, coord.builtin_frames.BaseRADecFrame):
                ph_type = "pos.eq.ra", "pos.eq.dec"
            elif isinstance(self.reference_frame, coord.builtin_frames.BaseEclipticFrame):
                ph_type = "pos.ecliptic.lon", "pos.ecliptic.lat"
            else:
                ph_type = tuple("custom:{}".format(t) for t in self.axes_names)
        elif isinstance(self, SpectralFrame):
            if self.unit[0].physical_type == "frequency":
                ph_type = ("em.freq",)
            elif self.unit[0].physical_type == "length":
                ph_type = ("em.wl",)
            elif self.unit[0].physical_type == "energy":
                ph_type = ("em.energy",)
            elif self.unit[0].physical_type == "speed":
                ph_type = ("spect.dopplerVeloc",)
                logging.warning("Physical type may be ambiguous. Consider "
                                "setting the physical type explicitly as "
                                "either 'spect.dopplerVeloc.optical' or "
                                "'spect.dopplerVeloc.radio'.")
            else:
                ph_type = ("custom:{}".format(self.unit[0].physical_type),)
        elif isinstance(self, TemporalFrame):
            ph_type = ("time",)
        elif isinstance(self, Frame2D):
            if all(self.axes_names):
                ph_type = self.axes_names
            else:
                ph_type = self.axes_type
            ph_type = tuple("custom:{}".format(t) for t in ph_type)
        else:
            ph_type = tuple("custom:{}".format(t) for t in self.axes_type)
        validate_physical_types(ph_type)
        return ph_type


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
        """ The number of axes in this frame."""
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
        """ Reference frame, used to convert to world coordinate objects. """
        return self._reference_frame

    @property
    def reference_position(self):
        """ Reference Position. """
        return getattr(self, "_reference_position", None)

    @property
    def axes_type(self):
        """ Type of this frame : 'SPATIAL', 'SPECTRAL', 'TIME'. """
        return self._axes_type

    def coordinates(self, *args):
        """ Create world coordinates object"""
        args = [args[i] for i in self.axes_order]
        coo = tuple([arg * un if not hasattr(arg, "to") else arg.to(un) for arg, un in zip(args, self.unit)])
        return coo

    def coordinate_to_quantity(self, *coords):
        """
        Given a rich coordinate object return an astropy quantity object.
        """
        # NoOp leaves it to the model to handle
        return coords

    @property
    def axis_physical_types(self):
        return self._axis_physical_types


class CelestialFrame(CoordinateFrame):
    """
    Celestial Frame Representation

    Parameters
    ----------
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        A reference frame.
    unit : str or units.Unit instance or iterable of those
        Units on axes.
    axes_names : list
        Names of the axes in this frame.
    name : str
        Name of this frame.
    """

    def __init__(self, axes_order=None, reference_frame=None,
                 unit=None, axes_names=None,
                 name=None, axis_physical_types=None):
        naxes = 2
        if reference_frame is not None:
            if not isinstance(reference_frame, str):
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
            unit = tuple([u.degree] * naxes)
        axes_type = ['SPATIAL'] * naxes

        super(CelestialFrame, self).__init__(naxes=naxes, axes_type=axes_type,
                                             axes_order=axes_order,
                                             reference_frame=reference_frame,
                                             unit=unit,
                                             axes_names=axes_names,
                                             name=name, axis_physical_types = axis_physical_types)

    def coordinates(self, *args):
        """
        Create a SkyCoord object.

        Parameters
        ----------
        args : float
            inputs to wcs.input_frame
        """
        if isinstance(args[0], coord.SkyCoord):
            return args[0].transform_to(self.reference_frame)
        else:
            return coord.SkyCoord(*args, unit=self.unit, frame=self.reference_frame)

    def coordinate_to_quantity(self, *coords):
        """ Convert a ``SkyCoord`` object to quantities."""
        if len(coords) == 2:
            arg = coords
        elif len(coords) == 1:
            arg = coords[0]
        else:
            raise ValueError("Unexpected number of coordinates in "
                             "input to frame {} : "
                             "expected 2, got  {}".format(self.name, len(coords)))

        if isinstance(arg, coord.SkyCoord):
            arg = arg.transform_to(self._reference_frame)
            try:
                lon = arg.data.lon
                lat = arg.data.lat
            except AttributeError:
                lon = arg.spherical.lon
                lat = arg.spherical.lat

            return lon, lat

        elif all(isinstance(a, u.Quantity) for a in arg):
            return tuple(arg)

        else:
            raise ValueError("Could not convert input {} to lon and lat quantities.".format(arg))


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
    reference_position : str
        Reference position - one of `STANDARD_REFERENCE_POSITION`

    """

    def __init__(self, axes_order=(0,), reference_frame=None, unit=None,
                 axes_names=None, name=None, axis_physical_types=None,
                 reference_position=None):
        super(SpectralFrame, self).__init__(naxes=1, axes_type="SPECTRAL", axes_order=axes_order,
                                            axes_names=axes_names, reference_frame=reference_frame,
                                            unit=unit, name=name,
                                            reference_position=reference_position,
                                            axis_physical_types=axis_physical_types)

    def coordinates(self, *args, equivalencies=[]):
        if hasattr(args[0], 'unit'):
            return args[0].to(self.unit[0], equivalencies=equivalencies)
        if np.isscalar(args):
            return args * self.unit[0]
        else:
            return args[0] * self.unit[0]

    def coordinate_to_quantity(self, *coords):
        if hasattr(coords[0], 'unit'):
            return coords[0]
        else:
            return coords[0] * self.unit[0]


class TemporalFrame(CoordinateFrame):
    """
    A coordinate frame for time axes.

    Parameters
    ----------
    axes_order : tuple or int
        A dimension in the input data that corresponds to this axis.
    reference_frame : `object`
        The object to instantiate to represent the time coordinate. Defaults to
        `astropy.time.Time`. Use partial functions to customise the
        `~astropy.time.Time` instance.
    reference_time : `astropy.time.Time` or `None`
        Reference time, the time of the 0th coordinate. If none the values of
        the axis are assumed to be valid times.
    unit : str or units.Unit instance
        Spectral unit.
    axes_names : str
        Spectral axis name.
    name : str
        Name for this frame.
    """

    def __init__(self, axes_order=(0,), reference_time=None,
                 reference_frame=astropy.time.Time, unit=None,
                 axes_names=None, name=None, axis_physical_types=None):

        super().__init__(naxes=1, axes_type="TIME", axes_order=axes_order,
                         axes_names=axes_names, reference_frame=reference_frame,
                         unit=unit, name=name,
                         reference_position=reference_time, axis_physical_types=axis_physical_types)

    def coordinates(self, *args):
        if np.isscalar(args):
            dt = args
        else:
            dt = args[0]

        if self.reference_position:
            if not hasattr(dt, 'unit'):
                dt = dt * self.unit[0]

            return self.reference_position + dt

        else:
            return self.reference_frame(dt)

    def coordinate_to_quantity(self, *coords):
        if isinstance(coords[0], astropy.time.Time):
            if self.reference_position:
                return (coords[0] - self.reference_position).to(self.unit[0])
            else:
                # If we can't convert to a quantity just drop the object out
                # and hope the transform can cope.
                return coords[0]
        # Is already a quantity
        elif hasattr(coords[0], 'unit'):
            return coords[0]
        else:
            raise ValueError("Can not convert {} to Quantity".format(coords[0]))


class CompositeFrame(CoordinateFrame):
    """
    Represents one or more frames.

    Parameters
    ----------
    frames : list
        List of frames (TemporalFrame, CelestialFrame, SpectralFrame, CoordinateFrame).
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
        ph_type = list(range(naxes))
        for frame in frames:
            axes_order.extend(frame.axes_order)
        for frame in frames:
            for ind, axtype, un, n, pht in zip(frame.axes_order, frame.axes_type,
                                               frame.unit, frame.axes_names, frame.axis_physical_types):
                axes_type[ind] = axtype
                axes_names[ind] = n
                unit[ind] = un
                ph_type[ind] = pht
        if len(np.unique(axes_order)) != len(axes_order):
            raise ValueError("Incorrect numbering of axes, "
                             "axes_order should contain unique numbers, "
                             "got {}.".format(axes_order))

        super(CompositeFrame, self).__init__(naxes, axes_type=axes_type,
                                             axes_order=axes_order,
                                             unit=unit, axes_names=axes_names,
                                             name=name)
        self._axis_physical_types = tuple(ph_type)

    @property
    def frames(self):
        return self._frames

    def __repr__(self):
        return repr(self.frames)

    def coordinates(self, *args):
        coo = []
        if len(args) == len(self.frames):
            for frame, arg in zip(self.frames, args):
                coo.append(frame.coordinates(arg))
        else:
            for frame in self.frames:
                fargs = [args[i] for i in frame.axes_order]
                coo.append(frame.coordinates(*fargs))
        return coo

    def coordinate_to_quantity(self, *coords):
        if len(coords) == len(self.frames):
            args = coords
        elif len(coords) == self.naxes:
            args = []
            for _frame in self.frames:
                if _frame.naxes > 1:
                    # Collect the arguments for this frame based on axes_order
                    args.append([coords[i] for i in _frame.axes_order])
                else:
                    args.append(coords[_frame.axes_order[0]])
        else:
            raise ValueError("Incorrect number of arguments")

        qs = []
        for _frame, arg in zip(self.frames, args):
            ret = _frame.coordinate_to_quantity(arg)
            if isinstance(ret, tuple):
                qs += list(ret)
            else:
                qs.append(ret)
        return qs


class StokesFrame(CoordinateFrame):
    """
    A coordinate frame for representing stokes polarisation states

    Parameters
    ----------
    name : str
        Name of this frame.
    """

    def __init__(self, axes_order=(0,), name=None):
        self._stokes_components = ['I', 'Q', 'U', 'V']
        super(StokesFrame, self).__init__(1, ["STOKES"], axes_order, name=name,
                                          axes_names=("stokes",), unit=u.one)

    def coordinates(self, *args):
        if hasattr(args[0], 'value'):
            arg = args[0].value
        else:
            arg = args[0]
        return self._stokes_components[int(arg)]

    def coordinate_to_quantity(self, *coords):
        if isinstance(coords[0], str):
            if coords[0] in self._stokes_components:
                return self._stokes_components.index(coords[0]) * u.pix
        else:
            return coords[0]


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
                 name=None, axis_physical_types=None):

        super(Frame2D, self).__init__(naxes=2, axes_type=["SPATIAL", "SPATIAL"],
                                      axes_order=axes_order, name=name,
                                      axes_names=axes_names, unit=unit,
                                      axis_physical_types=axis_physical_types)

    def coordinates(self, *args):
        args = [args[i] for i in self.axes_order]
        coo = tuple([arg * un for arg, un in zip(args, self.unit)])
        return coo

    def coordinate_to_quantity(self, *coords):
        # list or tuple
        if len(coords) == 1 and astutil.isiterable(coords[0]):
            coords = list(coords[0])
        elif len(coords) == 2:
            coords = list(coords)
        else:
            raise ValueError("Unexpected number of coordinates in "
                             "input to frame {} : "
                             "expected 2, got  {}".format(self.name, len(coords)))

        for i in range(2):
            if not hasattr(coords[i], 'unit'):
                coords[i] = coords[i] * self.unit[i]
        return tuple(coords)
