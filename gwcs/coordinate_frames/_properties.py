from dataclasses import dataclass
from typing import Self

import numpy as np
from astropy import units as u
from astropy.wcs.wcsapi.low_level_api import VALID_UCDS, validate_physical_types

from ._axis import AxesType, AxisType

__all__ = ["FrameProperties"]


@dataclass(frozen=True)
class FrameProperties:
    axes_type: tuple[AxisType | str, ...]
    unit: tuple[u.Unit, ...]
    axes_names: tuple[str, ...]
    axis_physical_types: tuple[str | None, ...]

    @classmethod
    def from_frame(
        cls,
        naxes: int,
        axes_type: AxesType,
        unit: tuple[u.Unit, ...] | None,
        axes_names: tuple[str, ...] | None,
        axis_physical_types: tuple[str | None, ...] | str,
    ) -> Self:
        """Class method constructor to allow FrameProperties to be Frozen."""
        axes_type = (axes_type,) if isinstance(axes_type, str) else tuple(axes_type)

        if len(axes_type) != naxes:
            msg = "Length of axes_type does not match number of axes."
            raise ValueError(msg)

        if unit is None:
            unit = tuple(u.dimensionless_unscaled for _ in range(naxes))
        else:
            unit = tuple(unit) if np.iterable(unit) else (unit,)
            if len(unit) != naxes:
                msg = "Number of units does not match number of axes."
                raise ValueError(msg)
            unit = tuple(au if au is None else u.Unit(au) for au in unit)

        if axes_names is None:
            axes_names = ("",) * naxes
        else:
            axes_names = (
                (axes_names,) if isinstance(axes_names, str) else tuple(axes_names)
            )
            if len(axes_names) != naxes:
                msg = "Number of axes names does not match number of axes."
                raise ValueError(msg)

        if isinstance(axis_physical_types, str):
            axis_physical_types = (axis_physical_types,)

        elif not np.iterable(axis_physical_types):
            msg = "axis_physical_types must be of type string or iterable of strings"
            raise TypeError(msg)

        if len(axis_physical_types) != naxes:
            msg = f'"axis_physical_types" must be of length {naxes}'
            raise ValueError(msg)

        ph_type: list[str | None] = []
        for axt in axis_physical_types:
            if (
                axt is not None
                and axt not in VALID_UCDS
                and not axt.startswith("custom:")
            ):
                ph_type.append(f"custom:{axt}")
            else:
                ph_type.append(axt)

        validate_physical_types(ph_type)
        axis_physical_types = tuple(ph_type)

        return cls(
            axes_type=axes_type,
            unit=unit,
            axes_names=axes_names,
            axis_physical_types=axis_physical_types,
        )
