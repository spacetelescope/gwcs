import warnings

from ._axis import AxesType, AxisType
from ._base import (
    CoordinateFrameProtocol,
    WorldAxisObjectClass,
    WorldAxisObjectClasses,
    WorldAxisObjectComponent,
)

__all__ = ["EmptyFrame"]


class EmptyFrameDeprecationWarning(DeprecationWarning):
    pass


class EmptyFrame(CoordinateFrameProtocol):
    """
    Represents a "default" detector frame. This is for use as the default value
    for input frame by the WCS object.
    """

    def __init__(self, name: str | None = None, naxes: int | None = None) -> None:
        self._name = "detector" if name is None else name
        self._naxes = naxes
        msg = (
            "The use of strings in place of a proper CoordinateFrame has been "
            "deprecated."
        )
        warnings.warn(msg, EmptyFrameDeprecationWarning, stacklevel=2)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}(name="{self.name}")>'

    def __str__(self) -> str:
        if self._name is not None:
            return self._name
        return type(self).__name__

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
    def naxes(self) -> int:
        if self._naxes is None:
            msg = "The number of axes for this EmptyFrame has not  been set."
            raise NotImplementedError(msg)

        return self._naxes

    @property
    def unit(self) -> tuple[None, ...]:
        return (None,) * self.naxes

    @property
    def axes_names(self) -> tuple[str, ...]:
        return ("None",) * self.naxes

    @property
    def axes_order(self) -> tuple[int, ...]:
        return tuple(range(self.naxes))

    @property
    def reference_frame(self) -> None:
        return None

    @property
    def axes_type(self) -> AxesType:
        return (AxisType.UNKNOWN,) * self.naxes

    @property
    def axis_physical_types(self):
        return tuple(f"custom:{t}" for t in self.axes_type)

    @property
    def world_axis_object_classes(self) -> WorldAxisObjectClasses:
        return {
            f"{at}{i}" if i != 0 else at: WorldAxisObjectClass(
                "None", (), {"unit": unit}
            )
            for i, (at, unit) in enumerate(zip(self.axes_type, self.unit, strict=False))
        }

    @property
    def world_axis_object_components(self) -> list[WorldAxisObjectComponent]:
        return [
            WorldAxisObjectComponent(f"{at}{i}" if i != 0 else at, 0, "value")
            for i, at in enumerate(self.axes_type)
        ]

    def to_high_level_coordinates(self, *values):
        self._raise_error()

    def from_high_level_coordinates(self, *high_level_coords):
        self._raise_error()
