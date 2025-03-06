from collections import defaultdict
from collections.abc import Generator
from typing import Any, cast

import numpy as np
from astropy import units as u

from gwcs._typing import AxisPhysicalType
from gwcs.api import (
    WorldAxisClass,
    WorldAxisClasses,
    WorldAxisComponent,
    WorldAxisComponents,
    WorldAxisConverterClass,
)

from ._axis import AxisType
from ._base import BaseCoordinateFrame
from ._core import CoordinateFrame

__all__ = ["CompositeFrame"]


class CompositeFrame(CoordinateFrame):
    """
    Represents one or more frames.

    Parameters
    ----------
    frames
        List of constituient frames.
    name
        Name for this frame.
    """

    def __init__(
        self, frames: list[BaseCoordinateFrame], name: str | None = None
    ) -> None:
        self._frames = frames[:]
        naxes = sum([frame.naxes for frame in self._frames])

        axes_order: list[int] = []
        axes_type: list[AxisType | str] = []
        axes_names: list[str] = []
        unit: list[u.Unit] = []
        ph_type: list[AxisPhysicalType] = []

        for frame in frames:
            axes_order.extend(frame.axes_order)

        # Stack the raw (not-native) ordered properties
        for frame in frames:
            axes_type += list(frame._prop.axes_type)
            axes_names += list(frame._prop.axes_names)
            # no common base class in astropy.units for all units
            unit += list(frame._prop.unit)  # type: ignore[arg-type]
            ph_type += list(frame._prop.axis_physical_types)

        if len(np.unique(axes_order)) != len(axes_order):
            msg = (
                "Incorrect numbering of axes, "
                "axes_order should contain unique numbers, "
                f"got {axes_order}."
            )
            raise ValueError(msg)

        super().__init__(
            naxes,
            axes_type=tuple(axes_type),
            axes_order=tuple(axes_order),
            unit=tuple(unit),
            axes_names=tuple(axes_names),
            axis_physical_types=tuple(ph_type),
            name=name,
        )
        self._axis_physical_types = tuple(ph_type)

    @property
    def frames(self) -> list[BaseCoordinateFrame]:
        """
        The constituient frames that comprise this `CompositeFrame`.
        """
        return self._frames

    def __repr__(self) -> str:
        return repr(self.frames)

    @property
    def _wao_classes_rename_map(self) -> dict[Any, Any]:
        mapper: dict[Any, Any] = defaultdict(dict)
        seen_names: list[str] = []
        for frame in self.frames:
            # ensure the frame is in the mapper
            mapper[frame]
            for key in frame.world_axis_object_classes:
                key = cast(str, key)
                if key in seen_names:
                    new_key = f"{key}{seen_names.count(key)}"
                    mapper[frame][key] = new_key
                seen_names.append(key)
        return mapper

    @property
    def _wao_renamed_components_iter(
        self,
    ) -> Generator[tuple[BaseCoordinateFrame, list[WorldAxisComponent]], None, None]:
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            renamed_components: list[WorldAxisComponent] = []
            for component in frame._native_world_axis_object_components:
                rename: str = mapper[frame].get(component[0], component[0])
                renamed_components.append(
                    WorldAxisComponent(rename, component[1], component[2])
                )

            yield frame, renamed_components

    @property
    def _wao_renamed_classes_iter(
        self,
    ) -> Generator[tuple[str, WorldAxisClass | WorldAxisConverterClass], None, None]:
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            for key, value in frame.world_axis_object_classes.items():
                rename: str = mapper[frame].get(key, key)
                yield rename, value

    @property
    def world_axis_object_components(self) -> WorldAxisComponents:
        """
        Object components for this frame.
        """
        out: list[WorldAxisComponent | None] = [None] * self.naxes

        for frame, components in self._wao_renamed_components_iter:
            for i, ao in enumerate(frame.axes_order):
                out[ao] = components[i]

        if any(o is None for o in out):
            msg = "axes_order leads to incomplete world_axis_object_components"
            raise ValueError(msg)

        # There can be None in the list here, but this is unique to this and
        # annoying otherwise, so we ignore MyPy
        return out  # type: ignore[return-value]

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        return dict(self._wao_renamed_classes_iter)
