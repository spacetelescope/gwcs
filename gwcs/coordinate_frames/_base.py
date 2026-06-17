from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Callable
from itertools import zip_longest
from numbers import Number
from typing import Any, NamedTuple, Protocol, Self, TypeAlias, runtime_checkable

import numpy as np
from astropy import units as u
from astropy.coordinates import BaseCoordinateFrame as _AstropyBaseCoordinateFrame
from astropy.time import Time, TimeDelta
from astropy.wcs.wcsapi.high_level_api import (
    high_level_objects_to_values,
    values_to_high_level_objects,
)
from numpy import typing as npt

from ._axis import AxesType

__all__ = [
    "AstropyBuiltInFrame",
    "BaseCoordinateFrame",
    "CoordinateFrameProtocol",
    "LowLevelArray",
    "LowLevelInput",
    "WorldAxisObjectClass",
    "WorldAxisObjectClassConverter",
    "WorldAxisObjectComponent",
]

AstropyBuiltInFrame: TypeAlias = Time | _AstropyBaseCoordinateFrame
LowLevelArray: TypeAlias = npt.NDArray[np.generic]
LowLevelInput: TypeAlias = LowLevelArray | u.Quantity


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
    def unit(self) -> tuple[u.Unit | None, ...]:
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
        self, arrays: tuple[LowLevelInput, ...] | LowLevelInput
    ) -> tuple[LowLevelInput, ...]:
        """
        Add units to the arrays
        """
        # Handle the case where we have a single axis input which maybe passed as a
        #    scalar rather than a tuple of length 1.
        if self.naxes == 1 and (np.isscalar(arrays) or isinstance(arrays, u.Quantity)):
            return (
                arrays if self.unit[0] is None else u.Quantity(arrays, self.unit[0]),
            )

        return tuple(
            # Add units to the array if there is a unit for the axis, otherwise
            #    just pass it through.
            array
            if unit is None or array is None
            else (
                array.to(unit)
                if isinstance(array, TimeDelta)
                else u.Quantity(array, unit=unit)
            )
            # zip_longest is used here to support "non-coordinate" inputs/outputs
            #   This implicitly assumes that the "non-coordinate" inputs/outputs
            #   are tacked onto the end of the tuple of "coordinate" inputs/outputs.
            for array, unit in zip_longest(arrays, self.unit)
        )

    def remove_units(
        self, arrays: tuple[LowLevelInput, ...] | LowLevelInput
    ) -> tuple[LowLevelArray, ...]:
        """
        Remove units from the input arrays
        """
        return tuple(
            # Strip the unit off an axis if the axis is a quantity,
            #     otherwise just pass it through.
            array.value if isinstance(array, u.Quantity) else array
            # self.add_units is used first because:
            # 1. If something is a Quantity, then it will be converted to the
            #    unit of the frame.
            # 2. If something is not a Quantity, but the frame has a unit for that
            #    axis, then we treat that as the correct magnitude but just missing
            #    the unit, so we get a Quantity with the correct unit.
            # 3. If there is no unit for the axis, then we just pass whatever it is
            #    through and hope for the best.
            # Now we have an array with the correct units, so we can safely strip
            #    the units off by accessing the .value (magnitude) of the attribute.
            for array in self.add_units(arrays)
        )

    def is_high_level(self, *args) -> bool:
        """
        Return `True` if the input coordinates are already high level objects
        described by this frame.

        This is used by the low level WCS API in Astropy to determine whether
        to call ``to_high_level_coordinates`` or not.
        """

        if (world_axis_object_classes := self.world_axis_object_classes) is None or len(
            args
        ) != len(world_axis_object_classes):
            return False

        type_match = []
        for arg, world_axis_object_class in zip(
            args, world_axis_object_classes.values(), strict=True
        ):
            if isinstance(class_object := world_axis_object_class.class_object, str):
                type_match.append(
                    type(arg).__name__ == class_object
                    and class_object != u.Quantity.__name__
                )
            else:
                type_match.append(
                    isinstance(arg, class_object) and class_object is not u.Quantity
                )

        if all(type_match):
            return True

        if any(type_match):
            types = [
                (
                    type(arg).__name__,
                    c.class_object
                    if isinstance(c.class_object, str)
                    else c.class_object.__name__,
                )
                for arg, c in zip(args, world_axis_object_classes.values(), strict=True)
            ]
            msg = (
                "Invalid types were passed, got "
                f"({', '.join(t[0] for t in types)}), but expected "
                f"({', '.join(t[1] for t in types)})."
            )
            raise TypeError(msg)

        return False

    def to_high_level_coordinates(self, *values, correct_1d=True):
        """
        Convert "values" to high level coordinate objects described by this frame.

        "values" are the coordinates in array or scalar form, and high level
        objects are things such as ``SkyCoord`` or ``Quantity``. See
        :ref:`wcsapi` for details.

        Parameters
        ----------
        values : `numbers.Number`, `numpy.ndarray`, or `~astropy.units.Quantity`
            ``naxis`` number of coordinates as scalars or arrays.

        Returns
        -------
        high_level_coordinates
            One (or more) high level object describing the coordinate.
        """
        # We allow Quantity-like objects here which values_to_high_level_objects
        # does not.
        values = self.remove_units(values)

        if not all(isinstance(v, Number) or type(v) is np.ndarray for v in values):
            msg = "All values should be a scalar number or a numpy array."
            raise TypeError(msg)

        high_level = values_to_high_level_objects(*values, low_level_wcs=self)
        if correct_1d and len(high_level) == 1:
            high_level = high_level[0]
        return high_level

    def from_high_level_coordinates(self, *high_level_coords, correct_1d=True):
        """
        Convert high level coordinate objects to "values" as described by this frame.

        "values" are the coordinates in array or scalar form, and high level
        objects are things such as ``SkyCoord`` or ``Quantity``. See
        :ref:`wcsapi` for details.

        Parameters
        ----------
        high_level_coordinates
            One (or more) high level object describing the coordinate.

        Returns
        -------
        values : `numbers.Number` or `numpy.ndarray`
            ``naxis`` number of coordinates as scalars or arrays.
        """
        values = high_level_objects_to_values(*high_level_coords, low_level_wcs=self)
        if correct_1d and self.naxes == 1:
            values = values[0]
        return values


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
