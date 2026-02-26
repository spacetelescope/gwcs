from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Callable
from itertools import zip_longest
from typing import Any, NamedTuple, Protocol, Self, TypeAlias, runtime_checkable

import numpy as np
from astropy import units as u
from astropy.coordinates import builtin_frames
from astropy.time import Time

from ._axis import AxesType

__all__ = [
    "AstropyBuiltInFrame",
    "BaseCoordinateFrame",
    "CoordinateFrameProtocol",
    "WorldAxisObjectClass",
    "WorldAxisObjectClassConverter",
    "WorldAxisObjectComponent",
]

AstropyBuiltInFrame: TypeAlias = (
    Time
    | builtin_frames.ICRS
    | builtin_frames.FK5
    | builtin_frames.FK4
    | builtin_frames.FK4NoETerms
    | builtin_frames.Galactic
    | builtin_frames.Galactocentric
    | builtin_frames.Supergalactic
    | builtin_frames.AltAz
    | builtin_frames.HADec
    | builtin_frames.GCRS
    | builtin_frames.CIRS
    | builtin_frames.ITRS
    | builtin_frames.HCRS
    | builtin_frames.TEME
    | builtin_frames.TETE
    | builtin_frames.PrecessedGeocentric
    | builtin_frames.GeocentricMeanEcliptic
    | builtin_frames.BarycentricMeanEcliptic
    | builtin_frames.HeliocentricMeanEcliptic
    | builtin_frames.GeocentricTrueEcliptic
    | builtin_frames.BarycentricTrueEcliptic
    | builtin_frames.HeliocentricTrueEcliptic
    | builtin_frames.HeliocentricEclipticIAU76
    | builtin_frames.CustomBarycentricEcliptic
    | builtin_frames.LSR
    | builtin_frames.LSRK
    | builtin_frames.LSRD
    | builtin_frames.GalacticLSR
    | builtin_frames.SkyOffsetFrame
    | builtin_frames.BaseEclipticFrame
    | builtin_frames.BaseRADecFrame
)


class WorldAxisObjectClass(NamedTuple):
    """
    A tuple to document the individual elements of the ``world_axis_object_classes``
    of the WCS API.

    Notes
    -----
    - ``world_axis_object_classes`` will return a dictionary with the key being
        the name from ``world_axis_object_components`` and the value being an
        instance of this class.

    - To stay consistent with the APE 14 API, users should not access the elements
        of this tuple via their names, but instead should access them via their
        position in the tuple.

    Attributes
    ----------
    class_object : type | str
        The High-Level Object class for the axis or a string that is the fully
        qualified name of the class.
    arguments : tuple
        The positional arguments to be passed to the class when instantiating an
        object of this class. Note if ``world_axis_object_components`` specifies that
        the world coordinates should be passed as a positional argument, then this
        tuple will include `None` as a place holder for each of the world coordinates.
    keyword_arguments : dict
        The keyword arguments to be passed to the class when instantiating an object of
        this class.
    """

    class_object: type | str
    arguments: tuple[Any, ...]
    keyword_arguments: dict[str, Any]


class WorldAxisObjectClassConverter(NamedTuple):
    """
    Same as the `WorldAxisObjectClass` but with an additional converter field.

    Attributes
    ----------
    converter : Callable[..., Any]
        A callable that will convert the input values into the desired output
    """

    class_object: type | str
    arguments: tuple[Any, ...]
    keyword_arguments: dict[str, Any]
    converter: Callable[..., Any]


WorldAxisObjectClasses: TypeAlias = (
    dict[str, WorldAxisObjectClass]
    | dict[str, WorldAxisObjectClassConverter]
    | dict[str, WorldAxisObjectClass | WorldAxisObjectClassConverter]
)


class WorldAxisObjectComponent(NamedTuple):
    """
    A tuple to document the individual elements of the ``world_axis_object_components``
    of the WCS API.

    Notes
    -----
    - ``world_axis_object_components`` will return a list of tuples with each tuple
        being an instance of this class.

    - To stay consistent with the APE 14 API, users should not access the elements
        of this tuple via their names, but instead should access them via their
        position in the tuple.

    Attributes
    ----------
    name : str
        Name for the world object this world array corresponds to, which *must*
        match the string names used in ``world_axis_object_classes``.  Note that
        names might appear twice because two world arrays might correspond to a
        single world object (e.g. a celestial coordinate might have both “ra”
        and “dec” arrays, which correspond to a single sky coordinate object.
    position : str | int
        This is either a string keyword argument name or a positional index for
        the corresponding class from ``world_axis_object_classes``.
    property: str | Callable[[Any], str]
        This is a string giving the name of the property to access on the
        corresponding class from ``world_axis_object_classes`` in order to get
        numerical values.
    """

    name: str
    position: str | int
    property: str | Callable[[Any], str]

    @classmethod
    def from_tuple(cls, tup: tuple[str, str | int, str]) -> Self:
        return cls(*tup)


@runtime_checkable
class CoordinateFrameProtocol(Protocol):
    """
    API Definition for a Coordinate frame
    """

    @property
    @abstractmethod
    def naxes(self) -> int:
        """
        The number of axes described by this frame.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the coordinate frame.
        """

    @property
    @abstractmethod
    def unit(self) -> tuple[u.Unit, ...]:
        """
        The units of the axes in this frame.
        """

    @property
    @abstractmethod
    def axes_names(self) -> tuple[str, ...]:
        """
        Names describing the axes of the frame.
        """

    @property
    @abstractmethod
    def axes_order(self) -> tuple[int, ...]:
        """
        The position of the axes in the frame in the transform.
        """

    @property
    @abstractmethod
    def reference_frame(self) -> AstropyBuiltInFrame | None:
        """
        The reference frame of the coordinates described by this frame.

        This is usually an Astropy object such as ``SkyCoord`` or ``Time``.
        """

    @property
    @abstractmethod
    def axes_type(self) -> AxesType:
        """
        An upcase string (or tuple of strings) describing the type of the axis.

        See AxisType for the known values, but you can also use
        your own custom one.
        """

    @property
    @abstractmethod
    def axis_physical_types(self) -> tuple[str | None, ...]:
        """
        The UCD 1+ physical types for the axes, in frame order.
        """

    @property
    @abstractmethod
    def world_axis_object_classes(self) -> WorldAxisObjectClasses:
        """
        The APE 14 object classes for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_classes
        """

    @property
    @abstractmethod
    def world_axis_object_components(self) -> list[WorldAxisObjectComponent]:
        """
        The APE 14 object components for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_components
        """

    def add_units(
        self, arrays: u.Quantity | np.ndarray | list[float]
    ) -> tuple[u.Quantity, ...] | u.Quantity:
        """
        Add units to the arrays
        """
        if self.naxes == 1 and np.isscalar(arrays):
            return u.Quantity(arrays, self.unit[0])

        return tuple(
            array if unit is None else u.Quantity(array, unit=unit)
            # zip_longest is used here to support "non-coordinate" inputs/outputs
            #   This implicitly assumes that the "non-coordinate" inputs/outputs
            #   are tacked onto the end of the tuple of "coordinate" inputs/outputs.
            for array, unit in zip_longest(arrays, self.unit)
        )

    def remove_units(
        self, arrays: u.Quantity | np.ndarray | list[float]
    ) -> tuple[np.ndarray, ...]:
        """
        Remove units from the input arrays
        """
        if self.naxes == 1 and (np.isscalar(arrays) or isinstance(arrays, u.Quantity)):
            arrays = (arrays,)

        return tuple(
            array.to_value(unit) if isinstance(array, u.Quantity) else array
            # zip_longest is used here to support "non-coordinate" inputs/outputs
            #   This implicitly assumes that the "non-coordinate" inputs/outputs
            #   are tacked onto the end of the tuple of "coordinate" inputs/outputs.
            for array, unit in zip_longest(arrays, self.unit)
        )


class BaseCoordinateFrame(CoordinateFrameProtocol):
    """
    Legacy base class for coordinate frames.
    """

    def __init_subclass__(cls, *args, **kwargs):
        msg = (
            "BaseCoordinateFrame has been deprecated and will be removed in a"
            "future release. Please implement or inherit from CoordinateFrameProtocol "
            "instead."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        super().__init_subclass__(*args, **kwargs)

    @property
    def world_axis_object_components(self):
        """
        The APE 14 object components for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_components
        """
        if self.naxes == 1:
            return self._native_world_axis_object_components

        # If we have more than one axis then we should sort the native
        # components by the axes_order.
        ordered = np.array(self._native_world_axis_object_components, dtype=object)[
            np.argsort(self.axes_order)
        ]
        return list(map(tuple, ordered))

    @property
    @abstractmethod
    def _native_world_axis_object_components(self):
        """
        This property holds the "native" frame order of the components.

        The native order of the components is the order the frame assumes the
        input arrays are in. This is not necessarily the same as the order of
        the axes in the frame, which is given by axes_order.
        """
