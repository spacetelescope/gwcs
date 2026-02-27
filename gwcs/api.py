"""
This module defines the abstract APIs for the GWCS Package:

- Native GWCS API defined by GWCS.
- WCS API defined in astropy APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from astropy import units as u
from astropy.modeling import separable
from astropy.wcs.wcsapi import BaseLowLevelWCS, HighLevelWCSMixin
from numpy import typing as npt

from gwcs import utils
from gwcs.coordinate_frames import EmptyFrame

if TYPE_CHECKING:
    from astropy.modeling import Model
    from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox

    from gwcs.coordinate_frames import (
        CoordinateFrameProtocol,
        WorldAxisObjectClasses,
        WorldAxisObjectComponent,
    )

__all__ = ["LowLevelInput", "NativeAPIMixin", "WCSAPIMixin"]

LowLevelArray: TypeAlias = npt.NDArray[np.generic]
LowLevelInput: TypeAlias = LowLevelArray | u.Quantity


class NativeAPIMixin(abc.ABC):
    """
    A mix-in class that is intended to be inherited by the
    :class:`~gwcs.wcs.WCS` class and provides the native GWCS API.
    """

    _pixel_shape: tuple[int, ...] | None

    @property
    @abc.abstractmethod
    def input_frame(self) -> CoordinateFrameProtocol:
        """The input coordinate frame."""

    @property
    @abc.abstractmethod
    def output_frame(self) -> CoordinateFrameProtocol:
        """The output coordinate frame."""

    @property
    @abc.abstractmethod
    def forward_transform(self) -> Model:
        """The forward transformation model from the input_frame to the output_frame."""

    @property
    @abc.abstractmethod
    def bounding_box(self) -> ModelBoundingBox | CompoundBoundingBox | None:
        """The input_frame's bounding box"""

    @abc.abstractmethod
    def evaluate(
        self,
        *args: LowLevelInput,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelInput, ...] | LowLevelInput:
        """
        Executes the forward transform.

        args : float or array-like
            Inputs in the input coordinate system, separate inputs
            for each dimension.
        with_bounding_box : bool, optional
            If True(default) values in the result which correspond to
            any of the inputs being outside the bounding_box are set
            to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box
            (default is np.nan).

        Other Parameters
        ----------------
        kwargs : dict
            Keyword arguments to be passed to the ``forward_transform`` model.
        """

    @abc.abstractmethod
    def invert(
        self,
        *args: LowLevelInput,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelInput, ...] | LowLevelInput:
        """
        Invert coordinates from output frame to input frame using analytical or
        user-supplied inverse. When neither analytical nor user-supplied
        inverses are defined, a numerical solution will be attempted using a
        numerical inverse algorithm.

        .. note::
            Currently numerical inverse is implemented only for 2D imaging WCS.

        Parameters
        ----------
        args : float, array like
            Coordinates to be inverted. The number of arguments must be equal
            to the number of world coordinates given by ``world_n_dim``.

        with_bounding_box : bool, optional
            If `True` (default) values in the result which correspond to any
            of the inputs being outside the bounding_box are set to
            ``fill_value``.

        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is ``np.nan``).

        Other Parameters
        ----------------
        kwargs : dict
            Keyword arguments to be passed to the ``backward_transform`` model
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : tuple or value
            Returns a tuple of scalar or array values for each axis. Unless
            ``input_frame.naxes == 1`` when it shall return the value.
            The return type will be `~astropy.units.Quantity` objects if the
            transform returns ``Quantity`` objects, else values.
        """


class WCSAPIMixin(BaseLowLevelWCS, HighLevelWCSMixin, NativeAPIMixin):
    """
    A mix-in class that is intended to be inherited by the
    :class:`~gwcs.wcs.WCS` class and provides the low- and high-level
    WCS API described in the astropy APE 14
    (https://doi.org/10.5281/zenodo.1188875).
    """

    # Low Level APE 14 API
    @property
    def pixel_n_dim(self) -> int:
        """
        The number of axes in the pixel coordinate system.
        """
        if isinstance(self.input_frame, EmptyFrame):
            # This is because astropy.modeling.Model does not type hint n_inputs
            return self.forward_transform.n_inputs  # type: ignore[no-any-return]
        return self.input_frame.naxes

    @property
    def world_n_dim(self) -> int:
        """
        The number of axes in the world coordinate system.
        """
        if isinstance(self.output_frame, EmptyFrame):
            # This is because astropy.modeling.Model does not type hint n_inputs
            return self.forward_transform.n_outputs  # type: ignore[no-any-return]
        return self.output_frame.naxes

    @property
    def world_axis_physical_types(self) -> tuple[str | None, ...] | None:
        """
        An iterable of strings describing the physical type for each world axis.
        These should be names from the VO UCD1+ controlled Vocabulary
        (http://www.ivoa.net/documents/latest/UCDlist.html). If no matching UCD
        type exists, this can instead be ``"custom:xxx"``, where ``xxx`` is an
        arbitrary string.  Alternatively, if the physical type is
        unknown/undefined, an element can be `None`.
        """
        return self.output_frame.axis_physical_types

    @property
    def world_axis_units(self) -> tuple[str, ...]:
        """
        An iterable of strings given the units of the world coordinates for each
        axis.
        The strings should follow the `IVOA VOUnit standard
        <http://ivoa.net/documents/VOUnits/>`_ (though as noted in the VOUnit
        specification document, units that do not follow this standard are still
        allowed, but just not recommended).
        """
        if isinstance(self.output_frame, EmptyFrame):
            return ()
        return tuple(unit.to_string(format="vounit") for unit in self.output_frame.unit)

    def _remove_quantity_frame(
        self,
        result: tuple[LowLevelInput, ...] | LowLevelInput,
        frame: CoordinateFrameProtocol,
    ) -> tuple[LowLevelArray, ...] | LowLevelArray:
        if isinstance(frame, EmptyFrame):
            return result

        if frame.naxes == 1:
            result = (result,)

        output: tuple[LowLevelArray, ...] = tuple(
            # `cast` is used here for mypy as to_value isn't type hinted properly for
            # Quantity. `cast` is a no-op at runtime, so there is no performance impact.
            cast(LowLevelArray, r.to_value(unit)) if isinstance(r, u.Quantity) else r
            # ToDo: use zip_longest here to support "non-coordinate" inputs/outputs.
            for r, unit in zip(result, frame.unit, strict=False)
        )

        # If we only have one output axes, we shouldn't return a tuple.
        if frame.naxes == 1 and isinstance(output, tuple):
            return output[0]
        return output

    def pixel_to_world_values(
        self, *pixel_arrays: LowLevelInput
    ) -> tuple[LowLevelArray, ...] | LowLevelArray:
        """
        Convert pixel coordinates to world coordinates.

        This method takes ``pixel_n_dim`` scalars or arrays as input, and pixel
        coordinates should be zero-based. Returns ``world_n_dim`` scalars or
        arrays in units given by ``world_axis_units``. Note that pixel
        coordinates are assumed to be 0 at the center of the first pixel in each
        dimension. If a pixel is in a region where the WCS is not defined, NaN
        can be returned. The coordinates should be specified in the ``(x, y)``
        order, where for an image, ``x`` is the horizontal coordinate and ``y``
        is the vertical coordinate.
        """
        return self._remove_quantity_frame(
            self.evaluate(*pixel_arrays), self.output_frame
        )

    def array_index_to_world_values(
        self, *index_arrays: LowLevelInput
    ) -> tuple[LowLevelArray, ...] | LowLevelArray:
        """
        Convert array indices to world coordinates.
        This is the same as `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values`
        except that the indices should be given in ``(i, j)`` order, where for an image
        ``i`` is the row and ``j`` is the column (i.e. the opposite order to
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values`).
        """
        return self.pixel_to_world_values(*index_arrays[::-1])

    def world_to_pixel_values(
        self, *world_arrays: LowLevelInput
    ) -> tuple[LowLevelArray, ...] | LowLevelArray:
        """
        Convert world coordinates to pixel coordinates.

        This method takes ``world_n_dim`` scalars or arrays as input in units
        given by ``world_axis_units``. Returns ``pixel_n_dim`` scalars or
        arrays. Note that pixel coordinates are assumed to be 0 at the center of
        the first pixel in each dimension. If a world coordinate does not have a
        matching pixel coordinate, NaN can be returned.  The coordinates should
        be returned in the ``(x, y)`` order, where for an image, ``x`` is the
        horizontal coordinate and ``y`` is the vertical coordinate.
        """
        return self._remove_quantity_frame(self.invert(*world_arrays), self.input_frame)

    def world_to_array_index_values(
        self, *world_arrays: LowLevelInput
    ) -> tuple[LowLevelArray, ...] | LowLevelArray:
        """
        Convert world coordinates to array indices.
        This is the same as `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_to_pixel_values`
        except that the indices should be returned in ``(i, j)`` order, where for an
        image ``i`` is the row and ``j`` is the column (i.e. the opposite order to
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values`). The indices should
        be returned as rounded integers.
        """
        results = self.world_to_pixel_values(*world_arrays)
        results = results[::-1] if isinstance(results, tuple) else (results,)

        results = tuple(utils.to_index(result) for result in results)
        return results[0] if self.pixel_n_dim == 1 else results

    @property
    def array_shape(self) -> tuple[int, ...] | None:
        """
        The shape of the data that the WCS applies to as a tuple of
        length `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim`.
        If the WCS is valid in the context of a dataset with a particular
        shape, then this property can be used to store the shape of the
        data. This can be used for example if implementing slicing of WCS
        objects. This is an optional property, and it should return `None`
        if a shape is not known or relevant.
        The shape should be given in ``(row, column)`` order (the convention
        for arrays in Python).
        """
        if self._pixel_shape is None:
            return None
        return self._pixel_shape[::-1]

    @array_shape.setter
    def array_shape(self, value: tuple[int, ...] | None) -> None:
        self.pixel_shape = None if value is None else value[::-1]

    @property
    def pixel_bounds(self) -> tuple[tuple[float, float], ...] | None:
        """
        The bounds (in pixel coordinates) inside which the WCS is defined,
        as a list with `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim`
        ``(min, max)`` tuples.  The bounds should be given in
        ``[(xmin, xmax), (ymin, ymax)]`` order. WCS solutions are sometimes
        only guaranteed to be accurate within a certain range of pixel values,
        for example when defining a WCS that includes fitted distortions. This
        is an optional property, and it should return `None` if a shape is not
        known or relevant.
        """
        bounding_box = self.bounding_box
        if bounding_box is None:
            return None

        if self.pixel_n_dim == 1 and len(bounding_box) == 2:
            bounding_box = (bounding_box,)

        # Iterate over the bounding box and convert from quantity if required.
        bounding_box = list(bounding_box)
        for i, bb_axes in enumerate(bounding_box):
            bb = [lim.value if isinstance(lim, u.Quantity) else lim for lim in bb_axes]

            bounding_box[i] = tuple(bb)

        return tuple(bounding_box)

    @property
    def pixel_shape(self) -> tuple[int, ...] | None:
        """
        The shape of the data that the WCS applies to as a tuple of length
        ``pixel_n_dim`` in ``(x, y)`` order (where for an image, ``x`` is
        the horizontal coordinate and ``y`` is the vertical coordinate)
        (optional).

        If the WCS is valid in the context of a dataset with a particular
        shape, then this property can be used to store the shape of the
        data. This can be used for example if implementing slicing of WCS
        objects. This is an optional property, and it should return `None`
        if a shape is neither known nor relevant.
        """
        return self._pixel_shape

    @pixel_shape.setter
    def pixel_shape(self, value: tuple[int, ...] | None) -> None:
        if value is None:
            self._pixel_shape = None
        else:
            if len(value) != self.pixel_n_dim:
                msg = (
                    "The number of data axes, "
                    f"{self.pixel_n_dim}, does not equal the "
                    f"shape {len(value)}."
                )
                raise ValueError(msg)

            self._pixel_shape = tuple(value)

    @property
    def axis_correlation_matrix(self):
        """
        Returns an (`~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim`,
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim`) matrix that indicates
        using booleans whether a given world coordinate depends on a given pixel
        coordinate.  This defaults to a matrix where all elements are `True` in
        the absence of any further information. For completely independent axes,
        the diagonal would be `True` and all other entries `False`.
        """
        return separable.separability_matrix(self.forward_transform)

    @property
    def serialized_classes(self) -> bool:
        """
        Indicates whether Python objects are given in serialized form or as
        actual Python objects.
        """
        return False

    @property
    def world_axis_object_classes(self) -> WorldAxisObjectClasses | None:
        if isinstance(self.output_frame, EmptyFrame):
            return None

        return self.output_frame.world_axis_object_classes

    @property
    def world_axis_object_components(self) -> list[WorldAxisObjectComponent] | None:
        if isinstance(self.output_frame, EmptyFrame):
            return None

        return self.output_frame.world_axis_object_components

    @property
    def pixel_axis_names(self) -> tuple[str, ...]:
        """
        An iterable of strings describing the name for each pixel axis.
        """
        if isinstance(self.input_frame, EmptyFrame):
            return tuple([""] * self.pixel_n_dim)

        return self.input_frame.axes_names

    @property
    def world_axis_names(self) -> tuple[str, ...]:
        """
        An iterable of strings describing the name for each world axis.
        """
        if isinstance(self.output_frame, EmptyFrame):
            return tuple([""] * self.world_n_dim)

        return self.output_frame.axes_names
