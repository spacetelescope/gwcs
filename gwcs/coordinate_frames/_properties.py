from __future__ import annotations

from collections.abc import Callable

from astropy.units import Unit, dimensionless_unscaled
from astropy.utils.misc import isiterable
from astropy.wcs.wcsapi.low_level_api import VALID_UCDS, validate_physical_types

from gwcs._typing import AxisPhysicalType, AxisPhysicalTypes

from ._axis import AxesType

__all__ = ["FrameProperties"]


class FrameProperties:
    def __init__(
        self,
        naxes: int,
        axes_type: AxesType,
        unit: tuple[Unit, ...] | None = None,
        axes_names: str | tuple[str, ...] | None = None,
        axis_physical_types: AxisPhysicalType | AxisPhysicalTypes | None = None,
        default_axis_physical_types: Callable[[FrameProperties], AxisPhysicalTypes]
        | None = None,
    ) -> None:
        self.naxes = naxes

        self.axes_type = (
            (axes_type,) if isinstance(axes_type, str) else tuple(axes_type)
        )

        if len(self.axes_type) != naxes:
            msg = "Length of axes_type does not match number of axes."
            raise ValueError(msg)

        if unit is None:
            self.unit = tuple(dimensionless_unscaled for _ in range(naxes))
        else:
            unit_ = tuple(unit) if isiterable(unit) else (unit,)  # type: ignore[no-untyped-call]
            if len(unit_) != naxes:
                msg = "Number of units does not match number of axes."
                raise ValueError(msg)
            self.unit = tuple(Unit(au) for au in unit_)  # type: ignore[no-untyped-call, misc]

        if axes_names is None:
            self.axes_names = tuple([""] * naxes)
        else:
            self.axes_names = (
                (axes_names,) if isinstance(axes_names, str) else tuple(axes_names)
            )
            if len(self.axes_names) != naxes:
                msg = "Number of axes names does not match number of axes."
                raise ValueError(msg)

        if axis_physical_types is None:
            if default_axis_physical_types is None:
                default_axis_physical_types = self._default_axis_physical_types

            self.axis_physical_types: AxisPhysicalTypes = default_axis_physical_types(
                self
            )
        else:
            self.axis_physical_types = (
                (axis_physical_types,)
                if isinstance(axis_physical_types, str)
                else tuple(axis_physical_types)
            )

            if len(self.axis_physical_types) != naxes:
                msg = f'"axis_physical_types" must be of length {naxes}'
                raise ValueError(msg)

            self.axis_physical_types = tuple(
                f"custom:{axt}"
                if axt not in VALID_UCDS and not axt.startswith("custom:")
                else axt
                for axt in self.axis_physical_types
            )
            validate_physical_types(self.axis_physical_types)  # type: ignore[no-untyped-call]

    @staticmethod
    def _default_axis_physical_types(properties: FrameProperties) -> AxisPhysicalTypes:
        """
        The default physical types to use for this frame if none are specified
        by the user.
        """
        return tuple(f"custom:{t}" for t in properties.axes_type)
