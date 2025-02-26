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

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


AxesType: TypeAlias = tuple[AxisType | str, ...] | AxisType | str
