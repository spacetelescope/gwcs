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


AxesType: TypeAlias = tuple[AxisType, ...] | AxisType
