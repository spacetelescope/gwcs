# Licensed under a 3-clause BSD style license - see LICENSE.rst


from astropy import units as u

from ._axis import AxisType
from ._core import CoordinateFrame

__all__ = ["Frame2D"]


class Frame2D(CoordinateFrame):
    """
    A 2D coordinate frame.

    Parameters
    ----------
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.
    unit : list of astropy.units.Unit
        Unit for each axis.
    axes_names : list
        Names of the axes in this frame.
    name : str
        Name of this frame.
    """

    def __init__(
        self,
        axes_order: tuple[int, int] = (0, 1),
        unit: tuple[u.Unit, u.Unit] = (u.pix, u.pix),
        axes_names: tuple[str, str] = ("x", "y"),
        name: str | None = None,
        axes_type: tuple[AxisType | str, AxisType | str] | None = None,
        axis_physical_types: tuple[str | None, str | None] | None = None,
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
            axis_physical_types=axis_physical_types
            or (
                axes_names
                if (axes_names is not None and all(axes_names))
                else axes_type
            ),
        )
