# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Defines coordinate frames and ties them to data axes.
"""
from collections import defaultdict
import logging
import numpy as np

from astropy.utils.misc import isiterable
from astropy import time
from astropy import units as u
from astropy import utils as astutil
from astropy import coordinates as coord
from astropy.wcs.wcsapi.low_level_api import (validate_physical_types,
                                              VALID_UCDS)
from astropy.wcs.wcsapi.fitswcs import CTYPE_TO_UCD1

__all__ = ['Frame2D', 'CelestialFrame', 'SpectralFrame', 'CompositeFrame',
           'CoordinateFrame', 'TemporalFrame']


def _ucd1_to_ctype_name_mapping(ctype_to_ucd, allowed_ucd_duplicates):
    inv_map = {}
    new_ucd = set()

    for kwd, ucd in ctype_to_ucd.items():
        if ucd in inv_map:
            if ucd not in allowed_ucd_duplicates:
                new_ucd.add(ucd)
            continue
        elif ucd in allowed_ucd_duplicates:
            inv_map[ucd] = allowed_ucd_duplicates[ucd]
        else:
            inv_map[ucd] = kwd

    if new_ucd:
        logging.warning(
            "Found unsupported duplicate physical type in 'astropy' mapping to CTYPE.\n"
            "Update 'gwcs' to the latest version or notify 'gwcs' developer.\n"
            "Duplicate physical types will be mapped to the following CTYPEs:\n" +
            '\n'.join([f'{repr(ucd):s} --> {repr(inv_map[ucd]):s}' for ucd in new_ucd])
        )

    return inv_map

# List below allowed physical type duplicates and a corresponding CTYPE
# to which all duplicates will be mapped to:
_ALLOWED_UCD_DUPLICATES = {
    'time': 'TIME',
    'em.wl': 'WAVE',
}

UCD1_TO_CTYPE = _ucd1_to_ctype_name_mapping(
    ctype_to_ucd=CTYPE_TO_UCD1,
    allowed_ucd_duplicates=_ALLOWED_UCD_DUPLICATES
)

STANDARD_REFERENCE_FRAMES = [frame.upper() for frame in coord.builtin_frames.__all__]

STANDARD_REFERENCE_POSITION = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
                               "TOPOCENTER", "LSR", "LSRK", "LSRD",
                               "GALACTIC_CENTER", "LOCAL_GROUP_CENTER"]


def get_ctype_from_ucd(ucd):
    """
    Return the FITS ``CTYPE`` corresponding to a UCD1 value.

    Parameters
    ----------
    ucd : str
        UCD string, for example one of ```WCS.world_axis_physical_types``.

    Returns
    -------
    CTYPE : str
        The corresponding FITS ``CTYPE`` value or an empty string.
    """
    return UCD1_TO_CTYPE.get(ucd, "")


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
        Reference position - one of ``STANDARD_REFERENCE_POSITION``
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

        elif isinstance(self, CelestialFrame):
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
        return tuple(ph_type)

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
        coo = tuple([arg * un if not hasattr(arg, "to") else arg.to(un) for arg, un in zip(args, self.unit)])
        if self.naxes == 1:
            return coo[0]
        return coo

    def coordinate_to_quantity(self, *coords):
        """
        Given a rich coordinate object return an astropy quantity object.
        """
        # NoOp leaves it to the model to handle
        # If coords is a 1-tuple of quantity then return the element of the tuple
        # This aligns the behavior with the other implementations
        if not hasattr(coords, 'unit') and len(coords) == 1:
            return coords[0]
        return coords

    @property
    def axis_physical_types(self):
        return self._axis_physical_types

    @property
    def _world_axis_object_classes(self):
        return {f"{at}{i}" if i != 0 else at: (u.Quantity,
                     (),
                     {'unit': unit})
                for i, (at, unit) in enumerate(zip(self._axes_type, self.unit))}

    @property
    def _world_axis_object_components(self):
        return [(f"{at}{i}" if i != 0 else at, 0, 'value') for i, at in enumerate(self._axes_type)]


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
                                             name=name, axis_physical_types=axis_physical_types)

    @property
    def _world_axis_object_classes(self):
        return {'celestial': (
            coord.SkyCoord,
            (),
            {'frame': self.reference_frame,
             'unit': self.unit})}

    @property
    def _world_axis_object_components(self):
        return [('celestial', 0, 'spherical.lon'),
                ('celestial', 1, 'spherical.lat')]

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
        Reference position - one of ``STANDARD_REFERENCE_POSITION``

    """

    def __init__(self, axes_order=(0,), reference_frame=None, unit=None,
                 axes_names=None, name=None, axis_physical_types=None,
                 reference_position=None):

        super(SpectralFrame, self).__init__(naxes=1, axes_type="SPECTRAL", axes_order=axes_order,
                                            axes_names=axes_names, reference_frame=reference_frame,
                                            unit=unit, name=name,
                                            reference_position=reference_position,
                                            axis_physical_types=axis_physical_types)

    @property
    def _world_axis_object_classes(self):
        return {'spectral': (
            coord.SpectralCoord,
            (),
            {'unit': self.unit[0]})}

    @property
    def _world_axis_object_components(self):
        return [('spectral', 0, 'value')]

    def coordinates(self, *args):
        # using SpectralCoord
        if isinstance(args[0], coord.SpectralCoord):
            return args[0].to(self.unit[0])
        else:
            if hasattr(args[0], 'unit'):
                return coord.SpectralCoord(*args).to(self.unit[0])
            else:
                return coord.SpectralCoord(*args, self.unit[0])

    def coordinate_to_quantity(self, *coords):
        if hasattr(coords[0], 'unit'):
            return coords[0]
        return coords[0] * self.unit[0]


class TemporalFrame(CoordinateFrame):
    """
    A coordinate frame for time axes.

    Parameters
    ----------
    reference_frame : `~astropy.time.Time`
        A Time object which holds the time scale and format.
        If data is provided, it is the time zero point.
        To not set a zero point for the frame initialize ``reference_frame``
        with an empty list.
    unit : str or `~astropy.units.Unit`
        Time unit.
    axes_names : str
        Time axis name.
    axes_order : tuple or int
        A dimension in the data that corresponds to this axis.
    name : str
        Name for this frame.
    """

    def __init__(self, reference_frame, unit=None, axes_order=(0,),
                 axes_names=None, name=None, axis_physical_types=None):
        axes_names = axes_names or "{}({}; {}".format(reference_frame.format,
                                                      reference_frame.scale,
                                                      reference_frame.location)

        super().__init__(naxes=1, axes_type="TIME", axes_order=axes_order,
                         axes_names=axes_names, reference_frame=reference_frame,
                         unit=unit, name=name, axis_physical_types=axis_physical_types)
        self._attrs = {}
        for a in self.reference_frame.info._represent_as_dict_extra_attrs:
            try:
                self._attrs[a] = getattr(self.reference_frame, a)
            except AttributeError:
                pass

    @property
    def _world_axis_object_classes(self):
        comp = (
            time.Time,
            (),
            {'unit': self.unit[0], **self._attrs},
            self._convert_to_time)

        return {'temporal': comp}

    @property
    def _world_axis_object_components(self):
        if isinstance(self.reference_frame.value, np.ndarray):
            return [('temporal', 0, 'value')]

        def offset_from_time_and_reference(time):
            return (time - self.reference_frame).sec
        return [('temporal', 0, offset_from_time_and_reference)]

    def coordinates(self, *args):
        if np.isscalar(args):
            dt = args
        else:
            dt = args[0]

        return self._convert_to_time(dt, unit=self.unit[0], **self._attrs)

    def _convert_to_time(self, dt, *, unit, **kwargs):
        if (not isinstance(dt, time.TimeDelta) and
                isinstance(dt, time.Time) or
                isinstance(self.reference_frame.value, np.ndarray)):
            return time.Time(dt, **kwargs)

        if not hasattr(dt, 'unit'):
            dt = dt * unit

        return self.reference_frame + dt

    def coordinate_to_quantity(self, *coords):
        if isinstance(coords[0], time.Time):
            ref_value = self.reference_frame.value
            if not isinstance(ref_value, np.ndarray):
                return (coords[0] - self.reference_frame).to(self.unit[0])
            else:
                # If we can't convert to a quantity just drop the object out
                # and hope the transform can cope.
                return coords[0]
        # Is already a quantity
        elif hasattr(coords[0], 'unit'):
            return coords[0]
        if isinstance(coords[0], np.ndarray):
            return coords[0] * self.unit[0]
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

    @property
    def _wao_classes_rename_map(self):
        mapper = defaultdict(dict)
        seen_names = []
        for frame in self.frames:
            # ensure the frame is in the mapper
            mapper[frame]
            for key in frame._world_axis_object_classes.keys():
                if key in seen_names:
                    new_key = f"{key}{seen_names.count(key)}"
                    mapper[frame][key] = new_key
                seen_names.append(key)
        return mapper

    @property
    def _wao_renamed_components_iter(self):
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            renamed_components = []
            for comp in frame._world_axis_object_components:
                comp = list(comp)
                rename = mapper[frame].get(comp[0])
                if rename:
                    comp[0] = rename
                renamed_components.append(tuple(comp))
            yield frame, renamed_components

    @property
    def _wao_renamed_classes_iter(self):
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            for key, value in frame._world_axis_object_classes.items():
                rename = mapper[frame].get(key)
                if rename:
                    key = rename
                yield key, value

    @property
    def _world_axis_object_components(self):
        """
        We need to generate the components respecting the axes_order.
        """
        out = [None] * self.naxes
        for frame, components in self._wao_renamed_components_iter:
            for i, ao in enumerate(frame.axes_order):
                out[ao] = components[i]

        if any([o is None for o in out]):
            raise ValueError("axes_order leads to incomplete world_axis_object_components")
        return out

    @property
    def _world_axis_object_classes(self):
        return dict(self._wao_renamed_classes_iter)


class StokesProfile(str):
    # This list of profiles in Table 7 in Greisen & Calabretta (2002)
    # modified to be 0 indexed
    profiles = {
        'I': 0,
        'Q': 1,
        'U': 2,
        'V': 3,
        'RR': -1,
        'LL': -2,
        'RL': -3,
        'LR': -4,
        'XX': -5,
        'YY': -6,
        'XY': -7,
        'YX': -8,
    }

    @classmethod
    def from_index(cls, indexes):
        """
        Construct a StokesProfile object from a numerical index.

        Parameters
        ----------
        indexes : `int`, `numpy.ndarray`
            An index or array of indices to construct StokesProfile objects from.
        """

        nans = np.isnan(indexes)
        indexes = np.asarray(indexes, dtype=int)
        out = np.empty_like(indexes, dtype=object)

        for profile, index in cls.profiles.items():
            out[indexes == index] = cls(profile)

        out[nans] = np.nan

        if out.size == 1 and not nans:
            return StokesProfile(out.item())
        elif nans.all():
            return np.array(out, dtype=float)
        return out

    def __new__(cls, content):
        content = str(content)
        if content not in cls.profiles.keys():
            raise ValueError(f"The profile name must be one of {cls.profiles.keys()} not {content}")
        return str.__new__(cls, content)

    def value(self):
        return self.profiles[self]


class StokesFrame(CoordinateFrame):
    """
    A coordinate frame for representing stokes polarisation states

    Parameters
    ----------
    name : str
        Name of this frame.
    """

    def __init__(self, axes_order=(0,), name=None):
        super(StokesFrame, self).__init__(1, ["STOKES"], axes_order, name=name,
                                          axes_names=("stokes",), unit=u.one,
                                          axis_physical_types="phys.polarization.stokes")

    @property
    def _world_axis_object_classes(self):
        return {'stokes': (
            StokesProfile,
            (),
            {},
            StokesProfile.from_index)}

    @property
    def _world_axis_object_components(self):
        return [('stokes', 0, 'value')]

    def coordinates(self, *args):
        if isinstance(args[0], u.Quantity):
            arg = args[0].value
        else:
            arg = args[0]

        return StokesProfile.from_index(arg)

    def coordinate_to_quantity(self, *coords):
        if isinstance(coords[0], str):
            if coords[0] in StokesProfile.profiles.keys():
                return StokesProfile.profiles[coords[0]] * u.one
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
