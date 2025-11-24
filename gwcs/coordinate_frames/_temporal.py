import contextlib

import numpy as np
from astropy import time
from astropy import units as u

from ._axis import AxisType
from ._core import CoordinateFrame

__all__ = ["TemporalFrame"]


class TemporalFrame(CoordinateFrame):
    """
    A coordinate frame for time axes.

    Parameters
    ----------
    reference_frame : `~astropy.time.Time`
        A Time object which holds the time scale and format.
        If data is provided, it is the time zero point.
        To not set a zero point for the frame initialize ``reference_frame``
        with an empty list.
    unit : str or `~astropy.units.Unit`
        Time unit.
    axes_names : str
        Time axis name.
    axes_order : tuple or int
        A dimension in the data that corresponds to this axis.
    name : str
        Name for this frame.
    """

    def __init__(
        self,
        reference_frame,
        unit=u.s,
        axes_order=(0,),
        axes_names=None,
        name=None,
        axis_physical_types=None,
    ):
        axes_names = (
            axes_names
            or f"{reference_frame.format}({reference_frame.scale}; "
            f"{reference_frame.location}"
        )

        pht = axis_physical_types or self._default_axis_physical_types()

        super().__init__(
            naxes=1,
            axes_type=AxisType.TIME,
            axes_order=axes_order,
            axes_names=axes_names,
            reference_frame=reference_frame,
            unit=unit,
            name=name,
            axis_physical_types=pht,
        )
        self._attrs = {}
        for a in self.reference_frame.info._represent_as_dict_extra_attrs:
            with contextlib.suppress(AttributeError):
                self._attrs[a] = getattr(self.reference_frame, a)

    def _default_axis_physical_types(self):
        return ("time",)

    def _convert_to_time(self, dt, *, unit, **kwargs):
        if (
            not isinstance(dt, time.TimeDelta) and isinstance(dt, time.Time)
        ) or isinstance(self.reference_frame.value, np.ndarray):
            return time.Time(dt, **kwargs)

        if not hasattr(dt, "unit"):
            dt = dt * unit

        return self.reference_frame + dt

    @property
    def world_axis_object_classes(self):
        comp = (
            time.Time,
            (),
            {"unit": self.unit[0], **self._attrs},
            self._convert_to_time,
        )

        return {"temporal": comp}

    @property
    def _native_world_axis_object_components(self):
        if isinstance(self.reference_frame.value, np.ndarray):
            return [("temporal", 0, "value")]

        def offset_from_time_and_reference(time):
            return (time - self.reference_frame).sec

        return [("temporal", 0, offset_from_time_and_reference)]
