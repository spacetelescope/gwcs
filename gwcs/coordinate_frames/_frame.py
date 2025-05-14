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
        axes_order=(0, 1),
        unit=(u.pix, u.pix),
        axes_names=("x", "y"),
        name=None,
        axes_type=None,
        axis_physical_types=None,
    ):
        if axes_type is None:
            axes_type = (AxisType.SPATIAL, AxisType.SPATIAL)
        pht = axis_physical_types or self._default_axis_physical_types(
            axes_names, axes_type
        )

        super().__init__(
            naxes=2,
            axes_type=axes_type,
            axes_order=axes_order,
            name=name,
            axes_names=axes_names,
            unit=unit,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self, axes_names, axes_type):
        if axes_names is not None and all(axes_names):
            ph_type = axes_names
        else:
            ph_type = axes_type

        return tuple(f"custom:{t}" for t in ph_type)
