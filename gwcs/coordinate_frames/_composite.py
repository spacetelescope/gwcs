from collections.abc import Generator

import numpy as np
from astropy import units as u

from ._axis import AxisType
from ._base import (
    WorldAxisObjectClass,
    WorldAxisObjectClassConverter,
    WorldAxisObjectClasses,
    WorldAxisObjectComponent,
)
from ._core import CoordinateFrame

__all__ = ["CompositeFrame"]


class CompositeFrame(CoordinateFrame):
    """
    Represents one or more frames.

    Parameters
    ----------
    frames : list
        List of constituient frames.
    name : str
        Name for this frame.
    """

    def __init__(self, frames: list[CoordinateFrame], name: str | None = None) -> None:
        self._frames = frames[:]
        naxes = sum([frame._naxes for frame in self._frames])

        axes_order: list[int] = []
        axes_type: list[AxisType | str] = []
        axes_names: list[str] = []
        unit: list[u.Unit] = []
        ph_type: list[str | None] = []

        for frame in frames:
            axes_order.extend(frame.axes_order)

        # Stack the raw (not-native) ordered properties
        for frame in frames:
            axes_type += list(frame._prop.axes_type)
            axes_names += list(frame._prop.axes_names)
            unit += list(frame._prop.unit)
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
        # Reset after the super init which may have messed this up
        self._axis_physical_types = tuple(ph_type)

    @property
    def frames(self) -> list[CoordinateFrame]:
        """
        The constituient frames that comprise this `CompositeFrame`.
        """
        return self._frames

    def __repr__(self) -> str:
        return repr(self.frames)

    @property
    def _wao_classes_rename_map(self) -> dict[CoordinateFrame, dict[str, str]]:
        mapper: dict[CoordinateFrame, dict[str, str]] = {}
        seen_names: list[str] = []
        for frame in self.frames:
            if frame not in mapper:
                mapper[frame] = {}

            for key in frame.world_axis_object_classes:
                if key in seen_names:
                    new_key = f"{key}{seen_names.count(key)}"
                    mapper[frame][key] = new_key
                seen_names.append(key)
        return mapper

    @property
    def _wao_renamed_components_iter(
        self,
    ) -> Generator[tuple[CoordinateFrame, list[WorldAxisObjectComponent]], None, None]:
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            yield (
                frame,
                [
                    WorldAxisObjectComponent(
                        mapper[frame].get(component[0], component[0]),
                        component[1],
                        component[2],
                    )
                    for component in frame._native_world_axis_object_components
                ],
            )

    @property
    def _wao_renamed_classes_iter(
        self,
    ) -> Generator[
        tuple[str, WorldAxisObjectClass | WorldAxisObjectClassConverter], None, None
    ]:
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            for key, value in frame.world_axis_object_classes.items():
                yield mapper[frame].get(key, key), value

    @property
    def world_axis_object_components(self) -> list[WorldAxisObjectComponent]:
        out_dict: dict[int, WorldAxisObjectComponent] = {}

        for frame, components in self._wao_renamed_components_iter:
            for i, ao in enumerate(frame.axes_order):
                out_dict[ao] = components[i]

        if len(out_dict) != self.naxes:
            msg = "axes_order leads to incomplete world_axis_object_components"
            raise ValueError(msg)

        return [out_dict[i] for i in range(self.naxes)]

    @property
    def world_axis_object_classes(self) -> WorldAxisObjectClasses:
        return dict(self._wao_renamed_classes_iter)
