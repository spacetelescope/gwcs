from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

__all__ = ["AxesType", "AxisType"]


class AxisType(StrEnum):
    """
    Enumeration of the Axis types
    """

    SPATIAL = "SPATIAL"
    SPECTRAL = "SPECTRAL"
    TIME = "TIME"
    STOKES = "STOKES"
    PIXEL = "PIXEL"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def from_input(cls, axis_type: str) -> AxisType | str:
        """
        Convert a string to an AxisType.
        """
        if (upper := axis_type.upper()) in cls.__members__:
            return cls[upper]

        return axis_type


AxesType: TypeAlias = tuple[AxisType | str, ...] | AxisType | str
