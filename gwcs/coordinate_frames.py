# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines coordinate frames for describing the inputs and/or outputs of a transform.

In the following example, we have a two stage transform, with an input frame, an
output frame and an intermediate frame.

.. code-block::

    ┌───────────────┐
    │               │
    │     Input     │
    │     Frame     │
    │               │
    └───────┬───────┘
            │
      ┌─────▼─────┐
      │ Transform │
      └─────┬─────┘
            │
    ┌───────▼───────┐
    │               │
    │  Intermediate │
    │     Frame     │
    │               │
    └───────┬───────┘
            │
      ┌─────▼─────┐
      │ Transform │
      └─────┬─────┘
            │
    ┌───────▼───────┐
    │               │
    │    Output     │
    │     Frame     │
    │               │
    └───────────────┘


Each frame instance is both metadata for the inputs/outputs of a transform and
also a converter between those inputs/outputs and richer coordinate
representations of those inputs/ouputs.

For example, an output frame of type `~gwcs.coordinate_frames.SpectralFrame`
provides metadata to the `.WCS` object such as the ``axes_type`` being
``"SPECTRAL"`` and the unit of the output etc.  The output frame also provides a
converter of the numeric output of the transform to a
`~astropy.coordinates.SpectralCoord` object, by combining this metadata with the
numerical values.

``axes_order`` and conversion between objects and arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the key concepts regarding coordinate frames is the ``axes_order`` argument.
This argument is used to map from the components of the frame to the inputs/outputs of the transform.
To illustrate this consider this situation where you have a forward transform
which outputs three coordinates ``[lat, lambda, lon]``.  These would be
represented as a `.SpectralFrame` and a `.CelestialFrame`, however, the axes of
a `.CelestialFrame` are always ``[lon, lat]``, so by specifying two frames as

.. code-block:: python

  [SpectralCoord(axes_order=(1,)), CelestialCoord(axes_order=(2, 0))]

we would map the outputs of this transform into the correct positions in the
frames. As shown below, this is also used when constructing the inputs to the inverse transform.

.. code-block::

                   lat, lambda, lon
                   │      │     │
                   └──────┼─────┼────────┐
              ┌───────────┘     └──┐     │
              │                    │     │
    ┌─────────▼────────┐    ┌──────▼─────▼─────┐
    │                  │    │                  │
    │  SpectralFrame   │    │  CelestialFrame  │
    │                  │    │                  │
    │       (1,)       │    │      (2, 0)      │
    │                  │    │                  │
    └─────────┬────────┘    └──────────┬────┬──┘
              │                        │    │
              │                        │    │
              ▼                        ▼    ▼
   SpectralCoord(lambda)    SkyCoord((lon, lat))
              │                        │    │
              └─────┐     ┌────────────┘    │
                    │     │    ┌────────────┘
                    ▼     ▼    ▼
                [lambda, lon, lat]
                    │     │    │
                    │     │    │
             ┌──────▼─────▼────▼────┐
             │                      │
             │  Sort by axes_order  │
             │                      │
             └────┬──────┬─────┬────┘
                  │      │     │
                  ▼      ▼     ▼
                 lat, lambda, lon

"""

import abc
from collections import defaultdict
import logging
import numpy as np
from dataclasses import dataclass, InitVar

from astropy.utils.misc import isiterable
from astropy import time
from astropy import units as u
from astropy import utils as astutil
from astropy import coordinates as coord
from astropy.wcs.wcsapi.low_level_api import (validate_physical_types,
                                              VALID_UCDS)
from astropy.wcs.wcsapi.fitswcs import CTYPE_TO_UCD1
from astropy.coordinates import StokesCoord

__all__ = ['BaseCoordinateFrame', 'Frame2D', 'CelestialFrame', 'SpectralFrame', 'CompositeFrame',
           'CoordinateFrame', 'TemporalFrame', 'StokesFrame']


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


class BaseCoordinateFrame(abc.ABC):
    """
    API Definition for a Coordinate frame
    """
    @property
    @abc.abstractmethod
    def naxes(self) -> int:
        """
        The number of axes described by this frame.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The name of the coordinate frame.
        """

    @property
    @abc.abstractmethod
    def unit(self) -> tuple[u.Unit, ...]:
        """
        The units of the axes in this frame.
        """

    @property
    @abc.abstractmethod
    def axes_names(self) -> tuple[str, ...]:
        """
        Names describing the axes of the frame.
        """

    @property
    @abc.abstractmethod
    def axes_order(self) -> tuple[int, ...]:
        """
        The position of the axes in the frame in the transform.
        """

    @property
    @abc.abstractmethod
    def reference_frame(self):
        """
        The reference frame of the coordinates described by this frame.

        This is usually an Astropy object such as ``SkyCoord`` or ``Time``.
        """

    @property
    @abc.abstractmethod
    def axes_type(self):
        """
        An upcase string describing the type of the axis.

        Known values are ``"SPATIAL", "TEMPORAL", "STOKES", "SPECTRAL", "PIXEL"``.
        """

    @property
    @abc.abstractmethod
    def axis_physical_types(self):
        """
        The UCD 1+ physical types for the axes, in frame order.
        """

    @property
    @abc.abstractmethod
    def world_axis_object_classes(self):
        """
        The APE 14 object classes for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_classes
        """

    @property
    @abc.abstractmethod
    def world_axis_object_components(self):
        """
        The APE 14 object components for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_components
        """


@dataclass
class FrameProperties:
    naxes: InitVar[int]
    axes_type: tuple[str]
    unit: tuple[u.Unit] = None
    axes_names: tuple[str] = None
    axis_physical_types: list[str] = None

    def __post_init__(self, naxes):
        if isinstance(self.axes_type, str):
            self.axes_type = (self.axes_type,)
        else:
            self.axes_type = tuple(self.axes_type)

        if len(self.axes_type) != naxes:
            raise ValueError("Length of axes_type does not match number of axes.")

        if self.unit is not None:
            if astutil.isiterable(self.unit):
                unit = tuple(self.unit)
            else:
                unit = (self.unit,)
            if len(unit) != naxes:
                raise ValueError("Number of units does not match number of axes.")
            else:
                self.unit = tuple(u.Unit(au) for au in unit)
        else:
            self.unit = tuple(u.dimensionless_unscaled for na in range(naxes))

        if self.axes_names is not None:
            if isinstance(self.axes_names, str):
                self.axes_names = (self.axes_names,)
            else:
                self.axes_names = tuple(self.axes_names)
            if len(self.axes_names) != naxes:
                raise ValueError("Number of axes names does not match number of axes.")
        else:
            self.axes_names = tuple([""] * naxes)

        if self.axis_physical_types is not None:
            if isinstance(self.axis_physical_types, str):
                self.axis_physical_types = (self.axis_physical_types,)
            elif not isiterable(self.axis_physical_types):
                raise TypeError("axis_physical_types must be of type string or iterable of strings")
            if len(self.axis_physical_types) != naxes:
                raise ValueError(f'"axis_physical_types" must be of length {naxes}')
            ph_type = []
            for axt in self.axis_physical_types:
                if axt not in VALID_UCDS and not axt.startswith("custom:"):
                    ph_type.append("custom:{axt}")
                else:
                    ph_type.append(axt)

            validate_physical_types(ph_type)
            self.axes_physical_types = tuple(ph_type)


class CoordinateFrame(BaseCoordinateFrame):
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
    unit : list of astropy.units.Unit
        Unit for each axis.
    axes_names : list
        Names of the axes in this frame.
    name : str
        Name of this frame.
    """

    def __init__(self, naxes, axes_type, axes_order, reference_frame=None,
                 unit=None, axes_names=None,
                 name=None, axis_physical_types=None):
        self._naxes = naxes
        self._axes_order = tuple(axes_order)
        self._reference_frame = reference_frame

        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

        if len(self._axes_order) != naxes:
            raise ValueError("Length of axes_order does not match number of axes.")

        if isinstance(axes_type, str):
            axes_type = (axes_type,)
        default_apt = tuple([f"custom:{t}" for t in axes_type])
        self._prop = FrameProperties(
            naxes,
            axes_type,
            unit,
            axes_names,
            axis_physical_types or default_apt,
        )

        super().__init__()

    def __repr__(self):
        fmt = '<{0}(name="{1}", unit={2}, axes_names={3}, axes_order={4}'.format(
            self.__class__.__name__, self.name,
            self.unit, self.axes_names, self.axes_order)
        if self.reference_frame is not None:
            fmt += ", reference_frame={0}".format(self.reference_frame)
        fmt += ")>"
        return fmt

    def __str__(self):
        if self._name is not None:
            return self._name
        return self.__class__.__name__

    def _sort_property(self, property):
        return tuple(dict(sorted(zip(property, self.axes_order),
                                 key=lambda x: x[1])).keys())

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
        return self._sort_property(self._prop.unit)

    @property
    def axes_names(self):
        """ Names of axes in the frame."""
        return self._sort_property(self._prop.axes_names)

    @property
    def axes_order(self):
        """ A tuple of indices which map inputs to axes."""
        return self._axes_order

    @property
    def reference_frame(self):
        """ Reference frame, used to convert to world coordinate objects. """
        return self._reference_frame

    @property
    def axes_type(self):
        """ Type of this frame : 'SPATIAL', 'SPECTRAL', 'TIME'. """
        return self._sort_property(self._prop.axes_type)

    @property
    def axis_physical_types(self):
        """
        The axis physical types for this frame.

        These physical types are the types in frame order, not transform order.
        """
        return self._sort_property(self._prop.axis_physical_types)

    @property
    def world_axis_object_classes(self):
        return {f"{at}{i}" if i != 0 else at: (u.Quantity,
                     (),
                     {'unit': unit})
                for i, (at, unit) in enumerate(zip(self._axes_type, self.unit))}

    @property
    def world_axis_object_components(self):
        return [(f"{at}{i}" if i != 0 else at, 0, 'value') for i, at in enumerate(self._prop.axes_type)]

    @property
    def _native_world_axis_object_components(self):
        """Defines the target component ordering (i.e. not taking into account axes_order)"""
        return self.world_axis_object_components


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

        super().__init__(naxes=naxes,
                         axes_type=axes_type,
                         axes_order=axes_order,
                         reference_frame=reference_frame,
                         unit=unit,
                         axes_names=axes_names,
                         name=name,
                         axis_physical_types=axis_physical_types)

    @property
    def _default_axis_physical_types(self):
        if isinstance(self.reference_frame, coord.Galactic):
            return "pos.galactic.lon", "pos.galactic.lat"
        elif isinstance(self.reference_frame, (coord.GeocentricTrueEcliptic,
                                               coord.GCRS,
                                               coord.PrecessedGeocentric)):
            return "pos.bodyrc.lon", "pos.bodyrc.lat"
        elif isinstance(self.reference_frame, coord.builtin_frames.BaseRADecFrame):
            return "pos.eq.ra", "pos.eq.dec"
        elif isinstance(self.reference_frame, coord.builtin_frames.BaseEclipticFrame):
            return "pos.ecliptic.lon", "pos.ecliptic.lat"
        else:
            return tuple("custom:{}".format(t) for t in self.axes_names)

    @property
    def world_axis_object_classes(self):
        return {'celestial': (
            coord.SkyCoord,
            (),
            {'frame': self.reference_frame,
             'unit': self.unit})}

    @property
    def _world_axis_object_components(self):
        return [('celestial', 0, lambda sc: sc.spherical.lon.to_value(self.unit[0])),
                ('celestial', 1, lambda sc: sc.spherical.lat.to_value(self.unit[1]))]

    @property
    def world_axis_object_components(self):
        # Sort the native waoc by the axes order. The axes order may have jumps
        # in it if there are other frames in between the components.
        ordered = np.array(self._native_world_axis_object_components,
                           dtype=object)[np.argsort(self.axes_order)]
        return list(map(tuple, ordered))


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
                 axes_names=None, name=None, axis_physical_types=None):

        super().__init__(naxes=1, axes_type="SPECTRAL", axes_order=axes_order,
                         axes_names=axes_names, reference_frame=reference_frame,
                         unit=unit, name=name,
                         axis_physical_types=axis_physical_types)

    @property
    def _default_axis_physical_types(self):
        if self.unit[0].physical_type == "frequency":
            return ("em.freq",)
        elif self.unit[0].physical_type == "length":
            return ("em.wl",)
        elif self.unit[0].physical_type == "energy":
            return ("em.energy",)
        elif self.unit[0].physical_type == "speed":
            return ("spect.dopplerVeloc",)
            logging.warning("Physical type may be ambiguous. Consider "
                            "setting the physical type explicitly as "
                            "either 'spect.dopplerVeloc.optical' or "
                            "'spect.dopplerVeloc.radio'.")
        else:
            return ("custom:{}".format(self.unit[0].physical_type),)

    @property
    def world_axis_object_classes(self):
        return {'spectral': (
            coord.SpectralCoord,
            (),
            {'unit': self.unit[0]})}

    @property
    def _world_axis_object_components(self):
        return [('spectral', 0, lambda sc: sc.to_value(self.unit[0]))]


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
    def _default_axis_physical_types(self):
        return ("time",)

    def _convert_to_time(self, dt, *, unit, **kwargs):
        if (not isinstance(dt, time.TimeDelta) and
                isinstance(dt, time.Time) or
                isinstance(self.reference_frame.value, np.ndarray)):
            return time.Time(dt, **kwargs)

        if not hasattr(dt, 'unit'):
            dt = dt * unit

        return self.reference_frame + dt

    @property
    def world_axis_object_classes(self):
        comp = (
            time.Time,
            (),
            {'unit': self.unit[0], **self._attrs},
            self._convert_to_time)

        return {'temporal': comp}

    @property
    def world_axis_object_components(self):
        if isinstance(self.reference_frame.value, np.ndarray):
            return [('temporal', 0, 'value')]

        def offset_from_time_and_reference(time):
            return (time - self.reference_frame).sec
        return [('temporal', 0, offset_from_time_and_reference)]


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
            unsorted_prop = zip(
                frame.axes_order,
                frame._prop.axes_type,
                frame._prop.unit,
                frame._prop.axes_names,
                frame._prop.axis_physical_types
            )
            for ind, axtype, un, n, pht in unsorted_prop:
                axes_type[ind] = axtype
                axes_names[ind] = n
                unit[ind] = un
                ph_type[ind] = pht
        if len(np.unique(axes_order)) != len(axes_order):
            raise ValueError("Incorrect numbering of axes, "
                             "axes_order should contain unique numbers, "
                             "got {}.".format(axes_order))

        super().__init__(naxes, axes_type=axes_type,
                         axes_order=axes_order,
                         unit=unit, axes_names=axes_names,
                         name=name)
        self._axis_physical_types = tuple(ph_type)

    @property
    def frames(self):
        return self._frames

    def __repr__(self):
        return repr(self.frames)

    @property
    def _wao_classes_rename_map(self):
        mapper = defaultdict(dict)
        seen_names = []
        for frame in self.frames:
            # ensure the frame is in the mapper
            mapper[frame]
            for key in frame.world_axis_object_classes.keys():
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
            for comp in frame._native_world_axis_object_components:
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
            for key, value in frame.world_axis_object_classes.items():
                rename = mapper[frame].get(key)
                if rename:
                    key = rename
                yield key, value

    @property
    def world_axis_object_components(self):
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
    def world_axis_object_classes(self):
        return dict(self._wao_renamed_classes_iter)


class StokesFrame(CoordinateFrame):
    """
    A coordinate frame for representing Stokes polarisation states.

    Parameters
    ----------
    name : str
        Name of this frame.
    axes_order : tuple
        A dimension in the data that corresponds to this axis.
    """

    def __init__(self, axes_order=(0,), axes_names=("stokes",), name=None, axis_physical_types=None):
        super().__init__(1, ["STOKES"], axes_order, name=name,
                         axes_names=axes_names, unit=u.one,
                         axis_physical_types=axis_physical_types)

    @property
    def _default_axis_physical_types(self):
        return ("phys.polarization.stokes",)

    @property
    def world_axis_object_classes(self):
        return {'stokes': (
            StokesCoord,
            (),
            {},
        )}

    @property
    def world_axis_object_components(self):
        return [('stokes', 0, 'value')]


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

        super().__init__(naxes=2, axes_type=["SPATIAL", "SPATIAL"],
                         axes_order=axes_order, name=name,
                         axes_names=axes_names, unit=unit,
                         axis_physical_types=axis_physical_types)

    @property
    def _default_axis_physical_types(self):
        if all(self.axes_names):
            ph_type = self.axes_names
        else:
            ph_type = self.axes_type
        return tuple("custom:{}".format(t) for t in ph_type)
