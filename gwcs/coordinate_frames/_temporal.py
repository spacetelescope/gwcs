import contextlib
from typing import Any, cast

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from gwcs._typing import AxisPhysicalTypes
from gwcs.api import (
    WorldAxisClasses,
    WorldAxisComponent,
    WorldAxisComponents,
    WorldAxisConverterClass,
)

from ._axis import AxisType
from ._core import CoordinateFrame
from ._properties import FrameProperties

__all__ = ["TemporalFrame"]


class TemporalFrame(CoordinateFrame):
    """
    A coordinate frame for time axes.

    Parameters
    ----------
    reference_frame
        A Time object which holds the time scale and format.
        If data is provided, it is the time zero point.
        To not set a zero point for the frame initialize ``reference_frame``
        with an empty list.
    unit
        Time unit.
    axes_names
        Time axis name.
    axes_order
        A dimension in the data that corresponds to this axis.
    name
        Name for this frame.
    """

    def __init__(
        self,
        reference_frame: Time,
        # Astropy dynamically builds these types at runtime, so MyPy can't find them
        unit: tuple[u.Unit, ...] = (u.s,),  # type: ignore[attr-defined]
        axes_order: tuple[int, ...] = (0,),
        axes_names: tuple[str, ...] | None = None,
        name: str | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        _axes_names = (
            (
                f"{reference_frame.format}({reference_frame.scale}; "
                f"{reference_frame.location}",
            )
            if axes_names is None
            else axes_names
        )

        super().__init__(
            naxes=1,
            axes_type=AxisType.TIME,
            axes_order=axes_order,
            axes_names=_axes_names,
            reference_frame=reference_frame,
            unit=unit,
            name=name,
            axis_physical_types=axis_physical_types,
        )
        self._attrs = {}
        for a in self.reference_frame.info._represent_as_dict_extra_attrs:
            with contextlib.suppress(AttributeError):
                self._attrs[a] = getattr(self.reference_frame, a)

    @property
    def reference_frame(self) -> Time:
        return cast(Time, self._reference_frame)

    def _default_axis_physical_types(
        self, properties: FrameProperties
    ) -> AxisPhysicalTypes:
        return ("time",)

    def _convert_to_time(self, dt: Any, *, unit: u.Unit, **kwargs: Any) -> Time:
        if (not isinstance(dt, TimeDelta) and isinstance(dt, Time)) or isinstance(
            self.reference_frame.value, np.ndarray
        ):
            return Time(dt, **kwargs)  # type: ignore[no-untyped-call]

        if not hasattr(dt, "unit"):
            dt = dt * unit

        return cast(Time, self.reference_frame + dt)

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        return {
            "temporal": WorldAxisConverterClass(
                Time,
                (),
                {"unit": self.unit[0], **self._attrs},
                self._convert_to_time,
            )
        }

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        if isinstance(self.reference_frame.value, np.ndarray):
            return [WorldAxisComponent("temporal", 0, "value")]

        def offset_from_time_and_reference(time: Time) -> Any:
            return (time - self.reference_frame).sec

        return [WorldAxisComponent("temporal", 0, offset_from_time_and_reference)]
