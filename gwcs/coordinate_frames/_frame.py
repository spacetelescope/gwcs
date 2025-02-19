from astropy import units as u

from gwcs._typing import AxisPhysicalTypes

from ._axis import AxesType, AxisType
from ._core import CoordinateFrame
from ._properties import FrameProperties


class Frame2D(CoordinateFrame):
    """
    A 2D coordinate frame.

    Parameters
    ----------
    axes_order
        A dimension in the input data that corresponds to this axis.
    unit
        Unit for each axis.
    axes_name
        Names of the axes in this frame.
    name
        Name of this frame.
    """

    def __init__(
        self,
        axes_order: tuple[int, ...] = (0, 1),
        # Astropy dynamically builds these types at runtime, so MyPy can't find them
        unit: tuple[u.Unit, ...] = (u.pix, u.pix),  # type: ignore[attr-defined]
        axes_names: tuple[str, ...] = ("x", "y"),
        name: str | None = None,
        axes_type: AxesType | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        if axes_type is None:
            axes_type = (AxisType.SPATIAL, AxisType.SPATIAL)

        super().__init__(
            naxes=2,
            axes_type=axes_type,
            axes_order=axes_order,
            name=name,
            axes_names=axes_names,
            unit=unit,
            axis_physical_types=axis_physical_types,
        )

    def _default_axis_physical_types(
        self, properties: FrameProperties
    ) -> AxisPhysicalTypes:
        if all(properties.axes_names):
            ph_type = properties.axes_names
        else:
            ph_type = properties.axes_type

        return tuple(f"custom:{t}" for t in ph_type)
