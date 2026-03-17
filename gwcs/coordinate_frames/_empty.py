from __future__ import annotations

import warnings
from typing import Self, TypeAlias, Union

from astropy import units as u
from astropy.modeling import Model

from ._axis import AxesType, AxisType
from ._base import (
    CoordinateFrameProtocol,
    LowLevelInput,
    WorldAxisObjectClass,
    WorldAxisObjectClasses,
    WorldAxisObjectComponent,
)

__all__ = ["EmptyFrame", "EmptyFrameDeprecationWarning", "EmptyFrameUnitsWarning"]

_Mdl: TypeAlias = Union[Model, None]  # noqa: UP007


class EmptyFrameDeprecationWarning(DeprecationWarning):
    pass


class EmptyFrameUnitsWarning(UserWarning):
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

    @classmethod
    def from_transform(cls, name: str | None = None, transform: _Mdl = None) -> Self:
        """
        Class method constructor to allow for an EmptyFrame to be created using data
        from a transform.

        Notes
        -----
        - This assumes that the number of inputs to the transform corresponds to the
            number of axes in the frame. Therefore, this assumes that there are NO
            non-coordinate inputs present.

        Parameters
        ----------
        name : str or None
            Name of the frame. If None, defaults to "detector".
        transform : astropy.modeling.Model or None
            A transform from this step's frame to next step's frame. The transform of
            the last step should be None.

        Returns
        -------
        EmptyFrame
            An EmptyFrame object with the number of axes set based on the transform.
        """
        return cls(name=name, naxes=None if transform is None else transform.n_inputs)

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

    @naxes.setter
    def naxes(self, val: int | None) -> None:
        self._naxes = val

    @property
    def unit(self) -> tuple[None, ...]:
        return (None,) * self.naxes

    @property
    def axes_names(self) -> tuple[str, ...]:
        return ("",) * self.naxes

    @property
    def axes_order(self) -> tuple[int, ...]:
        return tuple(range(self.naxes))

    @property
    def reference_frame(self) -> None:
        return None

    @property
    def axes_type(self) -> AxesType:
        return (AxisType.UNDEFINED,) * self.naxes

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

    def add_units(
        self, arrays: tuple[LowLevelInput, ...] | LowLevelInput
    ) -> tuple[LowLevelInput, ...]:
        msg = (
            "EmptyFrame (string frame) does not have any unit information. "
            "Therefore, GWCS cannot ensure that units are being properly handled. "
            "If an error occurs due to units, please provide a proper CoordinateFrame "
            "and/or ensure that the input values you are providing have the correct "
            "units attached (or removed) in order to be compatible with the transform."
        )
        warnings.warn(msg, EmptyFrameUnitsWarning, stacklevel=2)

        return super().add_units(arrays)

    def remove_units(
        self, arrays: tuple[LowLevelInput, ...] | LowLevelInput
    ) -> tuple[LowLevelInput, ...]:
        with warnings.catch_warnings():
            # Filter unit warning if there is no unit information to remove. This
            #   is because there is nothing for GWCS to be concerned about in this
            #   case and the only reason it is being triggered is that we attempt
            #   to add the unit information in order to force a unit conversion if
            #   needed.
            if (self.naxes == 1 and not isinstance(arrays, u.Quantity)) or not any(
                isinstance(arr, u.Quantity) for arr in arrays
            ):
                warnings.filterwarnings("ignore", category=EmptyFrameUnitsWarning)

            return super().remove_units(arrays)
