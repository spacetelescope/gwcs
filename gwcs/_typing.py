from __future__ import annotations

from fractions import Fraction
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    SpectralCoord,
    StokesCoord,
)
from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox
from astropy.time import Time
from astropy.units import Quantity

__all__ = [
    "AxisPhysicalType",
    "AxisPhysicalTypes",
    "BoundingBox",
    "Bounds",
    "HighLevelObject",
    "HighLevelObjects",
    "Interval",
    "LowLevelArrays",
    "LowLevelUnitArrays",
    "LowLevelUnitValue",
    "LowLevelValue",
    "OutputLowLevelArray",
    "Real",
]

Real: TypeAlias = int | float | Fraction | np.integer | np.floating

Interval: TypeAlias = tuple[Real, Real]
Bounds: TypeAlias = tuple[Interval, ...] | None

BoundingBox: TypeAlias = ModelBoundingBox | CompoundBoundingBox | None

# This is to represent a single  value from a low-level function.
LowLevelValue: TypeAlias = Real | npt.NDArray[np.number]
# Handle when units are a possibility. Not all functions allow units in/out
LowLevelUnitValue: TypeAlias = LowLevelValue | Quantity

# This is to represent all the values together for a single low-level function.
LowLevelArrays: TypeAlias = tuple[LowLevelValue, ...] | LowLevelValue
LowLevelUnitArrays: TypeAlias = tuple[LowLevelUnitValue, ...]

# This is to represent a general array output from a low-level function.
# Due to the fact 1D outputs are returned as a single value, rather than a tuple.
OutputLowLevelArray: TypeAlias = LowLevelValue | LowLevelArrays

HighLevelObject: TypeAlias = Time | SkyCoord | SpectralCoord | StokesCoord | Quantity
HighLevelObjects: TypeAlias = tuple[HighLevelObject, ...] | HighLevelObject

AxisPhysicalType: TypeAlias = str | BaseCoordinateFrame
AxisPhysicalTypes: TypeAlias = tuple[str | BaseCoordinateFrame, ...]
