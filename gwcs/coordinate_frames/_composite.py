from collections import defaultdict

import numpy as np

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

    def __init__(self, frames, name=None):
        self._frames = frames[:]
        naxes = sum([frame._naxes for frame in self._frames])

        axes_order = []
        axes_type = []
        axes_names = []
        unit = []
        ph_type = []

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
            axes_order=axes_order,
            unit=unit,
            axes_names=axes_names,
            axis_physical_types=tuple(ph_type),
            name=name,
        )
        self._axis_physical_types = tuple(ph_type)

    @property
    def frames(self):
        """
        The constituient frames that comprise this `CompositeFrame`.
        """
        return self._frames

    def __repr__(self):
        return repr(self.frames)

    @property
    def _wao_classes_rename_map(self):
        mapper = defaultdict(dict)
        seen_names = []
        for frame in self.frames:
            # ensure the frame is in the mapper
            mapper[frame]
            for key in frame.world_axis_object_classes:
                if key in seen_names:
                    new_key = f"{key}{seen_names.count(key)}"
                    mapper[frame][key] = new_key
                seen_names.append(key)
        return mapper

    @property
    def _wao_renamed_components_iter(self):
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            renamed_components = []
            for component in frame._native_world_axis_object_components:
                comp = list(component)
                rename = mapper[frame].get(comp[0])
                if rename:
                    comp[0] = rename
                renamed_components.append(tuple(comp))
            yield frame, renamed_components

    @property
    def _wao_renamed_classes_iter(self):
        mapper = self._wao_classes_rename_map
        for frame in self.frames:
            for key, value in frame.world_axis_object_classes.items():
                rename = mapper[frame].get(key)
                yield rename if rename else key, value

    @property
    def world_axis_object_components(self):
        out = [None] * self.naxes

        for frame, components in self._wao_renamed_components_iter:
            for i, ao in enumerate(frame.axes_order):
                out[ao] = components[i]

        if any(o is None for o in out):
            msg = "axes_order leads to incomplete world_axis_object_components"
            raise ValueError(msg)

        return out

    @property
    def world_axis_object_classes(self):
        return dict(self._wao_renamed_classes_iter)
