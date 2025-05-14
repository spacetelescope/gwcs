# Licensed under a 3-clause BSD style license - see LICENSE.rst

import contextlib
import logging
from collections import defaultdict

import numpy as np
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
from astropy.coordinates import StokesCoord
from astropy.wcs.wcsapi.fitswcs import CTYPE_TO_UCD1

from ._core import CoordinateFrame

__all__ = [
    "CelestialFrame",
    "CompositeFrame",
    "EmptyFrame",
    "Frame2D",
    "SpectralFrame",
    "StokesFrame",
    "TemporalFrame",
]


def _ucd1_to_ctype_name_mapping(ctype_to_ucd, allowed_ucd_duplicates):
    inv_map = {}
    new_ucd = set()

    for kwd, ucd in ctype_to_ucd.items():
        if ucd in inv_map:
            if ucd not in allowed_ucd_duplicates:
                new_ucd.add(ucd)
            continue
        inv_map[ucd] = allowed_ucd_duplicates.get(ucd, kwd)

    if new_ucd:
        logging.warning(
            "Found unsupported duplicate physical type in 'astropy' mapping to CTYPE.\n"
            "Update 'gwcs' to the latest version or notify 'gwcs' developer.\n"
            "Duplicate physical types will be mapped to the following CTYPEs:\n"
            + "\n".join([f"{ucd!r:s} --> {inv_map[ucd]!r:s}" for ucd in new_ucd])
        )

    return inv_map


# List below allowed physical type duplicates and a corresponding CTYPE
# to which all duplicates will be mapped to:
_ALLOWED_UCD_DUPLICATES = {
    "time": "TIME",
    "em.wl": "WAVE",
}

UCD1_TO_CTYPE = _ucd1_to_ctype_name_mapping(
    ctype_to_ucd=CTYPE_TO_UCD1, allowed_ucd_duplicates=_ALLOWED_UCD_DUPLICATES
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


class EmptyFrame(CoordinateFrame):
    """
    Represents a "default" detector frame. This is for use as the default value
    for input frame by the WCS object.
    """

    def __init__(self, name=None):
        self._name = "detector" if name is None else name

    def __repr__(self):
        return f'<{type(self).__name__}(name="{self.name}")>'

    def __str__(self):
        if self._name is not None:
            return self._name
        return type(self).__name__

    @property
    def name(self):
        """A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val):
        """A custom name of this frame."""
        self._name = val

    def _raise_error(self) -> None:
        msg = "EmptyFrame does not have any information"
        raise NotImplementedError(msg)

    @property
    def naxes(self):
        self._raise_error()

    @property
    def unit(self):
        return None

    @property
    def axes_names(self):
        self._raise_error()

    @property
    def axes_order(self):
        self._raise_error()

    @property
    def reference_frame(self):
        self._raise_error()

    @property
    def axes_type(self):
        self._raise_error()

    @property
    def axis_physical_types(self):
        self._raise_error()

    @property
    def world_axis_object_classes(self):
        self._raise_error()

    @property
    def _native_world_axis_object_components(self):
        self._raise_error()

    def to_high_level_coordinates(self, *values):
        self._raise_error()

    def from_high_level_coordinates(self, *high_level_coords):
        self._raise_error()


class CelestialFrame(CoordinateFrame):
    """
    Representation of a Celesital coordinate system.

    This class has a native order of longitude then latitude, meaning
    ``axes_names``, ``unit`` and ``axis_physical_types`` should be lon, lat
    ordered. If your transform is in a different order this should be specified
    with ``axes_order``.

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
    axis_physical_types : list
        The UCD 1+ physical types for the axes, in frame order (lon, lat).
    """

    def __init__(
        self,
        axes_order=None,
        reference_frame=None,
        unit=None,
        axes_names=None,
        name=None,
        axis_physical_types=None,
    ):
        naxes = 2
        if (
            reference_frame is not None
            and not isinstance(reference_frame, str)
            and reference_frame.name.upper() in STANDARD_REFERENCE_FRAMES
        ):
            _axes_names = list(reference_frame.representation_component_names.values())
            if "distance" in _axes_names:
                _axes_names.remove("distance")
            if axes_names is None:
                axes_names = _axes_names
            naxes = len(_axes_names)

        self.native_axes_order = tuple(range(naxes))
        if axes_order is None:
            axes_order = self.native_axes_order
        if unit is None:
            unit = tuple([u.degree] * naxes)
        axes_type = ["SPATIAL"] * naxes

        pht = axis_physical_types or self._default_axis_physical_types(
            reference_frame, axes_names
        )
        super().__init__(
            naxes=naxes,
            axes_type=axes_type,
            axes_order=axes_order,
            reference_frame=reference_frame,
            unit=unit,
            axes_names=axes_names,
            name=name,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self, reference_frame, axes_names):
        if isinstance(reference_frame, coord.Galactic):
            return "pos.galactic.lon", "pos.galactic.lat"
        if isinstance(
            reference_frame,
            coord.GeocentricTrueEcliptic | coord.GCRS | coord.PrecessedGeocentric,
        ):
            return "pos.bodyrc.lon", "pos.bodyrc.lat"
        if isinstance(reference_frame, coord.builtin_frames.BaseRADecFrame):
            return "pos.eq.ra", "pos.eq.dec"
        if isinstance(reference_frame, coord.builtin_frames.BaseEclipticFrame):
            return "pos.ecliptic.lon", "pos.ecliptic.lat"
        return tuple(f"custom:{t}" for t in axes_names)

    @property
    def world_axis_object_classes(self):
        return {
            "celestial": (
                coord.SkyCoord,
                (),
                {"frame": self.reference_frame, "unit": self._prop.unit},
            )
        }

    @property
    def _native_world_axis_object_components(self):
        return [
            ("celestial", 0, lambda sc: sc.spherical.lon.to_value(self._prop.unit[0])),
            ("celestial", 1, lambda sc: sc.spherical.lat.to_value(self._prop.unit[1])),
        ]


class SpectralFrame(CoordinateFrame):
    """
    Represents Spectral Frame

    Parameters
    ----------
    axes_order : tuple or int
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        Reference frame (usually used with output_frame to convert to world
        coordinate objects).
    unit : str or units.Unit instance
        Spectral unit.
    axes_names : str
        Spectral axis name.
    name : str
        Name for this frame.

    """

    def __init__(
        self,
        axes_order=(0,),
        reference_frame=None,
        unit=None,
        axes_names=None,
        name=None,
        axis_physical_types=None,
    ):
        if not np.iterable(unit):
            unit = (unit,)
        unit = [u.Unit(un) for un in unit]
        pht = axis_physical_types or self._default_axis_physical_types(unit)

        super().__init__(
            naxes=1,
            axes_type="SPECTRAL",
            axes_order=axes_order,
            axes_names=axes_names,
            reference_frame=reference_frame,
            unit=unit,
            name=name,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self, unit):
        if unit[0].physical_type == "frequency":
            return ("em.freq",)
        if unit[0].physical_type == "length":
            return ("em.wl",)
        if unit[0].physical_type == "energy":
            return ("em.energy",)
        if unit[0].physical_type == "speed":
            return ("spect.dopplerVeloc",)
            logging.warning(
                "Physical type may be ambiguous. Consider "
                "setting the physical type explicitly as "
                "either 'spect.dopplerVeloc.optical' or "
                "'spect.dopplerVeloc.radio'."
            )
        return (f"custom:{unit[0].physical_type}",)

    @property
    def world_axis_object_classes(self):
        return {"spectral": (coord.SpectralCoord, (), {"unit": self.unit[0]})}

    @property
    def _native_world_axis_object_components(self):
        return [("spectral", 0, lambda sc: sc.to_value(self.unit[0]))]


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

    def __init__(
        self,
        reference_frame,
        unit=u.s,
        axes_order=(0,),
        axes_names=None,
        name=None,
        axis_physical_types=None,
    ):
        axes_names = (
            axes_names
            or f"{reference_frame.format}({reference_frame.scale}; "
            f"{reference_frame.location}"
        )

        pht = axis_physical_types or self._default_axis_physical_types()

        super().__init__(
            naxes=1,
            axes_type="TIME",
            axes_order=axes_order,
            axes_names=axes_names,
            reference_frame=reference_frame,
            unit=unit,
            name=name,
            axis_physical_types=pht,
        )
        self._attrs = {}
        for a in self.reference_frame.info._represent_as_dict_extra_attrs:
            with contextlib.suppress(AttributeError):
                self._attrs[a] = getattr(self.reference_frame, a)

    def _default_axis_physical_types(self):
        return ("time",)

    def _convert_to_time(self, dt, *, unit, **kwargs):
        if (
            not isinstance(dt, time.TimeDelta) and isinstance(dt, time.Time)
        ) or isinstance(self.reference_frame.value, np.ndarray):
            return time.Time(dt, **kwargs)

        if not hasattr(dt, "unit"):
            dt = dt * unit

        return self.reference_frame + dt

    @property
    def world_axis_object_classes(self):
        comp = (
            time.Time,
            (),
            {"unit": self.unit[0], **self._attrs},
            self._convert_to_time,
        )

        return {"temporal": comp}

    @property
    def _native_world_axis_object_components(self):
        if isinstance(self.reference_frame.value, np.ndarray):
            return [("temporal", 0, "value")]

        def offset_from_time_and_reference(time):
            return (time - self.reference_frame).sec

        return [("temporal", 0, offset_from_time_and_reference)]


class CompositeFrame(CoordinateFrame):
    """
    Represents one or more frames.

    Parameters
    ----------
    frames : list
        List of constituient frames.
    name : str
        Name for this frame.
    """

    def __init__(self, frames, name=None):
        self._frames = frames[:]
        naxes = sum([frame._naxes for frame in self._frames])

        axes_order = []
        axes_type = []
        axes_names = []
        unit = []
        ph_type = []

        for frame in frames:
            axes_order.extend(frame.axes_order)

        # Stack the raw (not-native) ordered properties
        for frame in frames:
            axes_type += list(frame._prop.axes_type)
            axes_names += list(frame._prop.axes_names)
            unit += list(frame._prop.unit)
            ph_type += list(frame._prop.axis_physical_types)

        if len(np.unique(axes_order)) != len(axes_order):
            msg = (
                "Incorrect numbering of axes, "
                "axes_order should contain unique numbers, "
                f"got {axes_order}."
            )
            raise ValueError(msg)

        super().__init__(
            naxes,
            axes_type=axes_type,
            axes_order=axes_order,
            unit=unit,
            axes_names=axes_names,
            axis_physical_types=tuple(ph_type),
            name=name,
        )
        self._axis_physical_types = tuple(ph_type)

    @property
    def frames(self):
        """
        The constituient frames that comprise this `CompositeFrame`.
        """
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
            for key in frame.world_axis_object_classes:
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
            for component in frame._native_world_axis_object_components:
                comp = list(component)
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
                yield rename if rename else key, value

    @property
    def world_axis_object_components(self):
        out = [None] * self.naxes

        for frame, components in self._wao_renamed_components_iter:
            for i, ao in enumerate(frame.axes_order):
                out[ao] = components[i]

        if any(o is None for o in out):
            msg = "axes_order leads to incomplete world_axis_object_components"
            raise ValueError(msg)

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

    def __init__(
        self,
        axes_order=(0,),
        axes_names=("stokes",),
        name=None,
        axis_physical_types=None,
    ):
        pht = axis_physical_types or self._default_axis_physical_types()

        super().__init__(
            1,
            ["STOKES"],
            axes_order,
            name=name,
            axes_names=axes_names,
            unit=u.one,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self):
        return ("phys.polarization.stokes",)

    @property
    def world_axis_object_classes(self):
        return {
            "stokes": (
                StokesCoord,
                (),
                {},
            )
        }

    @property
    def _native_world_axis_object_components(self):
        return [("stokes", 0, "value")]


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

    def __init__(
        self,
        axes_order=(0, 1),
        unit=(u.pix, u.pix),
        axes_names=("x", "y"),
        name=None,
        axes_type=None,
        axis_physical_types=None,
    ):
        if axes_type is None:
            axes_type = ["SPATIAL", "SPATIAL"]
        pht = axis_physical_types or self._default_axis_physical_types(
            axes_names, axes_type
        )

        super().__init__(
            naxes=2,
            axes_type=axes_type,
            axes_order=axes_order,
            name=name,
            axes_names=axes_names,
            unit=unit,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self, axes_names, axes_type):
        if axes_names is not None and all(axes_names):
            ph_type = axes_names
        else:
            ph_type = axes_type

        return tuple(f"custom:{t}" for t in ph_type)
