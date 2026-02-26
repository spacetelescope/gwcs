from __future__ import annotations

from itertools import zip_longest
from typing import Protocol, runtime_checkable

import numpy as np
from astropy import units as u

from ._axis import AxesType

__all__ = ["CoordinateFrameProtocol"]


@runtime_checkable
class CoordinateFrameProtocol(Protocol):
    """
    API Definition for a Coordinate frame
    """

    @property
    def naxes(self) -> int:
        """
        The number of axes described by this frame.
        """

    @property
    def name(self) -> str:
        """
        The name of the coordinate frame.
        """

    @property
    def unit(self) -> tuple[u.Unit, ...]:
        """
        The units of the axes in this frame.
        """

    @property
    def axes_names(self) -> tuple[str, ...]:
        """
        Names describing the axes of the frame.
        """

    @property
    def axes_order(self) -> tuple[int, ...]:
        """
        The position of the axes in the frame in the transform.
        """

    @property
    def reference_frame(self):
        """
        The reference frame of the coordinates described by this frame.

        This is usually an Astropy object such as ``SkyCoord`` or ``Time``.
        """

    @property
    def axes_type(self) -> AxesType:
        """
        An upcase string (or tuple of strings) describing the type of the axis.

        See AxisType for the known values, but you can also use
        your own custom one.
        """

    @property
    def axis_physical_types(self):
        """
        The UCD 1+ physical types for the axes, in frame order.
        """

    @property
    def world_axis_object_classes(self):
        """
        The APE 14 object classes for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_classes
        """

    @property
    def world_axis_object_components(self):
        """
        The APE 14 object components for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_components
        """

    def add_units(
        self, arrays: u.Quantity | np.ndarray | list[float]
    ) -> tuple[u.Quantity, ...] | u.Quantity:
        """
        Add units to the arrays
        """
        if self.naxes == 1 and np.isscalar(arrays):
            return u.Quantity(arrays, self.unit[0])

        return tuple(
            array if unit is None else u.Quantity(array, unit=unit)
            # zip_longest is used here to support "non-coordinate" inputs/outputs
            #   This implicitly assumes that the "non-coordinate" inputs/outputs
            #   are tacked onto the end of the tuple of "coordinate" inputs/outputs.
            for array, unit in zip_longest(arrays, self.unit)
        )

    def remove_units(
        self, arrays: u.Quantity | np.ndarray | list[float]
    ) -> tuple[np.ndarray, ...]:
        """
        Remove units from the input arrays
        """
        if self.naxes == 1 and (np.isscalar(arrays) or isinstance(arrays, u.Quantity)):
            arrays = (arrays,)

        return tuple(
            array.to_value(unit) if isinstance(array, u.Quantity) else array
            # zip_longest is used here to support "non-coordinate" inputs/outputs
            #   This implicitly assumes that the "non-coordinate" inputs/outputs
            #   are tacked onto the end of the tuple of "coordinate" inputs/outputs.
            for array, unit in zip_longest(arrays, self.unit)
        )
