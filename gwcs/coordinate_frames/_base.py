from __future__ import annotations

import warnings
from abc import abstractmethod
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
    @abstractmethod
    def naxes(self) -> int:
        """
        The number of axes described by this frame.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the coordinate frame.
        """

    @property
    @abstractmethod
    def unit(self) -> tuple[u.Unit, ...]:
        """
        The units of the axes in this frame.
        """

    @property
    @abstractmethod
    def axes_names(self) -> tuple[str, ...]:
        """
        Names describing the axes of the frame.
        """

    @property
    @abstractmethod
    def axes_order(self) -> tuple[int, ...]:
        """
        The position of the axes in the frame in the transform.
        """

    @property
    @abstractmethod
    def reference_frame(self):
        """
        The reference frame of the coordinates described by this frame.

        This is usually an Astropy object such as ``SkyCoord`` or ``Time``.
        """

    @property
    @abstractmethod
    def axes_type(self) -> AxesType:
        """
        An upcase string (or tuple of strings) describing the type of the axis.

        See AxisType for the known values, but you can also use
        your own custom one.
        """

    @property
    @abstractmethod
    def axis_physical_types(self):
        """
        The UCD 1+ physical types for the axes, in frame order.
        """

    @property
    @abstractmethod
    def world_axis_object_classes(self):
        """
        The APE 14 object classes for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_classes
        """

    @property
    @abstractmethod
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


class BaseCoordinateFrame(CoordinateFrameProtocol):
    """
    Legacy base class for coordinate frames.
    """

    def __init_subclass__(cls, *args, **kwargs):
        msg = (
            "BaseCoordinateFrame has been deprecated and will be removed in a"
            "future release. Please implement or inherit from CoordinateFrameProtocol "
            "instead."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        super().__init_subclass__(*args, **kwargs)

    @property
    def world_axis_object_components(self):
        """
        The APE 14 object components for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_components
        """
        if self.naxes == 1:
            return self._native_world_axis_object_components

        # If we have more than one axis then we should sort the native
        # components by the axes_order.
        ordered = np.array(self._native_world_axis_object_components, dtype=object)[
            np.argsort(self.axes_order)
        ]
        return list(map(tuple, ordered))

    @property
    @abstractmethod
    def _native_world_axis_object_components(self):
        """
        This property holds the "native" frame order of the components.

        The native order of the components is the order the frame assumes the
        input arrays are in. This is not necessarily the same as the order of
        the axes in the frame, which is given by axes_order.
        """
