from astropy import units as u
from astropy.coordinates import BaseCoordinateFrame as _BaseCoordinateFrame

from gwcs._typing import AxisPhysicalTypes
from gwcs.api import WorldAxisClasses, WorldAxisComponents

from ._axis import AxesType
from ._core import CoordinateFrame

__all__ = ["EmptyFrame"]


class EmptyFrame(CoordinateFrame):
    """
    Represents a "default" detector frame. This is for use as the default value
    for input frame by the WCS object.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = "detector" if name is None else name

    def __repr__(self) -> str:
        return f'<{type(self).__name__}(name="{self.name}")>'

    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        """A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        """A custom name of this frame."""
        self._name = val

    def _raise_error(self) -> None:
        msg = "EmptyFrame does not have any information"
        raise NotImplementedError(msg)

    @property
    def naxes(self) -> int:  # type: ignore[return]
        self._raise_error()

    @property
    def unit(self) -> tuple[u.Unit, ...]:  # type: ignore[return]
        self._raise_error()

    @property
    def axes_names(self) -> tuple[str, ...]:  # type: ignore[return]
        self._raise_error()

    @property
    def axes_order(self) -> tuple[int, ...]:  # type: ignore[return]
        self._raise_error()

    @property
    def reference_frame(self) -> _BaseCoordinateFrame | None:  # type: ignore[return]
        self._raise_error()

    @property
    def axes_type(self) -> AxesType:  # type: ignore[return]
        self._raise_error()

    @property
    def axis_physical_types(self) -> AxisPhysicalTypes:  # type: ignore[return]
        self._raise_error()

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:  # type: ignore[return]
        self._raise_error()

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:  # type: ignore[return]
        self._raise_error()
