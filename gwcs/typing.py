"""Type aliases and classes for public GWCS API.

This module serves as a single source of truth for GWCS type definitions,
making them available for users and improving Sphinx documentation resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeVar, Union

from astropy.coordinates import BaseCoordinateFrame
from astropy.modeling import Model
from astropy.time import Time
from astropy.units import Quantity
from numpy import dtype, generic, ndarray

from .coordinate_frames import (
    AxisType,
    CoordinateFrameProtocol,
    WorldAxisObjectClass,
    WorldAxisObjectClassConverter,
)
from .wcs import Step
from .wcs._pipeline import _BasePipeline

__all__ = [
    "AstropyBuiltInFrame",
    "AxesType",
    "ForwardTransform",
    "LowLevelArray",
    "LowLevelInput",
    "Mdl",
    "StepTuple",
    "WorldAxisObjectClasses",
]

_DtypeGeneric = TypeVar("_DtypeGeneric", bound=generic)

AstropyBuiltInFrame: TypeAlias = Time | BaseCoordinateFrame
LowLevelArray: TypeAlias = ndarray[tuple[int, ...], dtype[_DtypeGeneric]]
LowLevelInput: TypeAlias = LowLevelArray | Quantity


WorldAxisObjectClasses: TypeAlias = (
    dict[str, WorldAxisObjectClass]
    | dict[str, WorldAxisObjectClassConverter]
    | dict[str, WorldAxisObjectClass | WorldAxisObjectClassConverter]
)


AxesType: TypeAlias = tuple[AxisType | str, ...] | AxisType | str


# Type aliases due to the use of the `|` for type hints not working with Model
Mdl: TypeAlias = Union[Model, None]  # noqa: UP007
StepTuple: TypeAlias = tuple[CoordinateFrameProtocol, Mdl]
ForwardTransform: TypeAlias = Union[Model, Sequence[Step | StepTuple] | _BasePipeline]  # noqa: UP007
