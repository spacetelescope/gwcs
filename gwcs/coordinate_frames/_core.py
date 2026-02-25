import numbers
from typing import TypeVar

import numpy as np
from astropy import units as u
from astropy.wcs.wcsapi.high_level_api import (
    high_level_objects_to_values,
    values_to_high_level_objects,
)

from ._axis import AxesType, AxisType
from ._base import (
    AstropyBuiltInFrame,
    BaseCoordinateFrame,
    WorldAxisObjectClass,
    WorldAxisObjectClassConverter,
    WorldAxisObjectComponent,
)
from ._properties import FrameProperties

__all__ = ["CoordinateFrame"]

_PropertyItem = TypeVar("_PropertyItem")


class CoordinateFrame(BaseCoordinateFrame):
    """
    Base class for Coordinate Frames.

    Parameters
    ----------
    naxes : int
        Number of axes.
    axes_type : AxesType
        One of ["SPATIAL", "SPECTRAL", "TIME"]
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        Reference frame (usually used with output_frame to convert to world
        coordinate objects).
    unit : list of astropy.units.Unit
        Unit for each axis.
    axes_names : list
        Names of the axes in this frame.
    name : str
        Name of this frame.
    """

    def __init__(
        self,
        naxes: int,
        axes_type: AxesType,
        axes_order: tuple[int, ...],
        reference_frame: AstropyBuiltInFrame | None = None,
        unit: tuple[u.Unit, ...] | None = None,
        axes_names: tuple[str, ...] | None = None,
        name: str | None = None,
        axis_physical_types: tuple[str | None, ...] | None = None,
    ) -> None:
        self._naxes = naxes
        self._axes_order = tuple(axes_order)
        self._reference_frame = reference_frame

        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

        if len(self._axes_order) != naxes:
            msg = "Length of axes_order does not match number of axes."
            raise ValueError(msg)

        if isinstance(axes_type, str):
            axes_type = (axes_type,)

        self._prop = FrameProperties.from_frame(
            naxes,
            axes_type,
            unit,
            axes_names,
            axis_physical_types or self._default_axis_physical_types(axes_type),
        )

        super().__init__()

    def _default_axis_physical_types(
        self, axes_type: tuple[AxisType | str, ...]
    ) -> tuple[str, ...]:
        """
        The default physical types to use for this frame if none are specified
        by the user.
        """
        return tuple(f"custom:{t}" for t in axes_type)

    def __repr__(self) -> str:
        fmt = (
            f'<{self.__class__.__name__}(name="{self.name}", unit={self.unit}, '
            f"axes_names={self.axes_names}, axes_order={self.axes_order}"
        )
        if self.reference_frame is not None:
            fmt += f", reference_frame={self.reference_frame}"
        fmt += ")>"
        return fmt

    def __str__(self) -> str:
        if self._name is not None:
            return self._name
        return self.__class__.__name__

    def _sort_property(
        self, prop: tuple[_PropertyItem, ...]
    ) -> tuple[_PropertyItem, ...]:
        """Reorder the frame properties according to 'axes_order'."""
        sorted_prop = sorted(
            zip(prop, self.axes_order, strict=False), key=lambda x: x[1]
        )
        return tuple([t[0] for t in sorted_prop])

    @property
    def name(self) -> str:
        """A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        """A custom name of this frame."""
        self._name = val

    @property
    def naxes(self) -> int:
        """The number of axes in this frame."""
        return self._naxes

    @property
    def unit(self) -> tuple[u.Unit, ...]:
        """The unit of this frame."""
        return self._sort_property(self._prop.unit)

    @property
    def axes_names(self) -> tuple[str, ...]:
        """Names of axes in the frame."""
        return self._sort_property(self._prop.axes_names)

    @property
    def axes_order(self) -> tuple[int, ...]:
        """A tuple of indices which map inputs to axes."""
        return self._axes_order

    @property
    def reference_frame(self) -> AstropyBuiltInFrame | None:
        """Reference frame, used to convert to world coordinate objects."""
        return self._reference_frame

    @property
    def axes_type(self) -> tuple[AxisType | str, ...]:
        """Type of this frame : 'SPATIAL', 'SPECTRAL', 'TIME'."""
        return self._sort_property(self._prop.axes_type)

    @property
    def axis_physical_types(self) -> tuple[str | None, ...]:
        """
        The axis physical types for this frame.

        These physical types are the types in frame order, not transform order.
        """
        return self._sort_property(self._prop.axis_physical_types)

    @property
    def world_axis_object_classes(
        self,
    ) -> (
        dict[str, WorldAxisObjectClass]
        | dict[str, WorldAxisObjectClassConverter]
        | dict[str, WorldAxisObjectClass | WorldAxisObjectClassConverter]
    ):
        return {
            f"{at}{i}" if i != 0 else at: WorldAxisObjectClass(
                u.Quantity, (), {"unit": unit}
            )
            for i, (at, unit) in enumerate(zip(self.axes_type, self.unit, strict=True))
        }

    @property
    def _native_world_axis_object_components(self) -> list[WorldAxisObjectComponent]:
        return [
            WorldAxisObjectComponent(f"{at}{i}" if i != 0 else at, 0, "value")
            for i, at in enumerate(self._prop.axes_type)
        ]

    @property
    def serialized_classes(self) -> bool:
        """
        This property is used by the low level WCS API in Astropy.

        By providing it we can duck type as a low level WCS object.
        """
        return False

    @property
    def raw_properties(self) -> FrameProperties:
        """The raw FrameProperties object for this frame."""
        return self._prop

    def to_high_level_coordinates(self, *values):
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
        values = [
            v.to_value(unit) if hasattr(v, "to_value") else v
            for v, unit in zip(values, self.unit, strict=False)
        ]

        if not all(
            isinstance(v, numbers.Number) or type(v) is np.ndarray for v in values
        ):
            msg = "All values should be a scalar number or a numpy array."
            raise TypeError(msg)

        high_level = values_to_high_level_objects(*values, low_level_wcs=self)
        if len(high_level) == 1:
            high_level = high_level[0]
        return high_level

    def from_high_level_coordinates(self, *high_level_coords):
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
        if len(values) == 1:
            values = values[0]
        return values
