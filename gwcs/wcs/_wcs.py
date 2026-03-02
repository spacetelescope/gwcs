# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import annotations

import functools
import itertools
import sys
import warnings
from copy import copy
from typing import Self, overload

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.modeling import Model, fix_inputs, projections
from astropy.modeling.bounding_box import ModelBoundingBox as Bbox
from astropy.modeling.models import (
    Mapping,
    RotateCelestial2Native,
    Shift,
    Sky2Pix_TAN,
    Tabular1D,
    Tabular2D,
)
from astropy.modeling.parameters import _tofloat
from astropy.wcs.utils import celestial_frame_to_wcs, proj_plane_pixel_scales
from astropy.wcs.wcsapi.high_level_api import (
    high_level_objects_to_values,
    values_to_high_level_objects,
)
from numpy import typing as npt
from scipy import optimize

from gwcs.api import WCSAPIMixin
from gwcs.coordinate_frames import (
    AxisType,
    CelestialFrame,
    CompositeFrame,
    CoordinateFrameProtocol,
    EmptyFrame,
    LowLevelArray,
    LowLevelInput,
    get_ctype_from_ucd,
)
from gwcs.utils import _compute_lon_pole, is_high_level, to_index

from ._exception import NoConvergence
from ._pipeline import ForwardTransform, Pipeline
from ._step import Step, StepTuple
from ._utils import (
    fit_2D_poly,
    fix_transform_inputs,
    make_sampling_grid,
    reform_poly_coefficients,
    store_2D_coefficients,
)

__all__ = ["WCS"]

_ITER_INV_KWARGS = ["tolerance", "maxiter", "adaptive", "detect_divergence", "quiet"]


class _WorldAxisInfo:
    def __init__(self, axis, frame, world_axis_order, cunit, ctype, input_axes):
        """
        A class for holding information about a world axis from an output frame.

        Parameters
        ----------
        axis : int
            Output axis number [in the forward transformation].

        frame : cf.CoordinateFrameProtocol
            Coordinate frame to which this axis belongs.

        world_axis_order : int
            Index of this axis in `gwcs.WCS.output_frame.axes_order`

        cunit : str
            Axis unit using FITS conversion (``CUNIT``).

        ctype : str
            Axis FITS type (``CTYPE``).

        input_axes : tuple of int
            Tuple of input axis indices contributing to this world axis.

        """
        self.axis = axis
        self.frame = frame
        self.world_axis_order = world_axis_order
        self.cunit = cunit
        self.ctype = ctype
        self.input_axes = input_axes


class WCS(Pipeline, WCSAPIMixin):
    """
    Basic WCS class.

    Parameters
    ----------
    forward_transform : `~astropy.modeling.Model` or a list
        The transform between ``input_frame`` and ``output_frame``.
        A list of (frame, transform) tuples where ``frame`` is the starting frame and
        ``transform`` is the transform from this frame to the next one or
        ``output_frame``.  The last tuple is (transform, None), where None indicates
        the end of the pipeline.
    input_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    output_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    name : str
        a name for this WCS

    """

    @overload
    def __init__(
        self,
        forward_transform: Model,
        input_frame: str | CoordinateFrameProtocol,
        output_frame: str | CoordinateFrameProtocol,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        forward_transform: list[Step | StepTuple],
        input_frame: None = None,
        output_frame: None = None,
        name: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        forward_transform: ForwardTransform,
        input_frame: str | CoordinateFrameProtocol | None = None,
        output_frame: str | CoordinateFrameProtocol | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            forward_transform=forward_transform,
            # mypy for some reason isn't able to infer the correct overload here
            input_frame=input_frame,  # type: ignore[arg-type]
            output_frame=output_frame,  # type: ignore[arg-type]
        )

        self._approx_inverse = None
        self._name = "" if name is None else name
        self._pixel_shape = None

    def _add_units_input(
        self, arrays: tuple[LowLevelInput, ...], frame: CoordinateFrameProtocol
    ) -> tuple[LowLevelInput, ...]:
        if not isinstance(frame, EmptyFrame):
            return frame.add_units(arrays)

        # This is a falllback that should be rarely used if ever
        return arrays  # type: ignore[return-value]

    def _remove_units_input(
        self, arrays: tuple[LowLevelInput, ...], frame: CoordinateFrameProtocol
    ) -> tuple[LowLevelArray, ...]:
        if not isinstance(frame, EmptyFrame):
            return frame.remove_units(arrays)

        return arrays

    def evaluate(
        self,
        *args: LowLevelInput,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelInput, ...] | LowLevelInput:
        # Call into variable as this involves computing the forward transform
        #   after each call to it.
        transform = self.forward_transform

        input_is_quantity, transform_uses_quantity = self._units_are_present(
            args, transform
        )
        args = self._make_input_units_consistent(
            transform,
            *args,
            frame=self.input_frame,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )

        result = transform(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )
        if not isinstance(self.output_frame, EmptyFrame):
            if self.output_frame.naxes == 1:
                result = (result,)

            result = self._make_output_units_consistent(
                transform,
                *result,
                frame=self.output_frame,
                input_is_quantity=input_is_quantity,
                transform_uses_quantity=transform_uses_quantity,
            )

            if self.output_frame.naxes == 1:
                return result[0]
        return result

    def __call__(
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
        """
        return self.evaluate(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )

    def _units_are_present(
        self, args: tuple[LowLevelInput, ...], transform: Model
    ) -> tuple[bool, bool]:
        """
        Determine if the inputs to a transform are quantities and the transform
        supports units.

        Parameters
        ----------
        args : a tuple of scalars or ndarray-like objects
            Inputs to a transform.
        transform : `~astropy.modeling.Model`
            Transform to be evaluated.

        Returns
        -------
        input_is_quantity, transform_uses_quantity : bool

        """
        # Validate that the input type matches what the transform expects
        input_is_quantity = any(isinstance(a, u.Quantity) for a in args)
        if isinstance(transform, (Tabular1D, Tabular2D)):
            transform_uses_quantity = (
                isinstance(transform.lookup_table, u.Quantity)
                or transform.input_units_equivalencies is not None
            )
        elif transform is not None and len(transform.parameters) == 0:
            transform_uses_quantity = False
        else:
            transform_uses_quantity = not (
                transform is None or not transform.uses_quantity
            )
        return input_is_quantity, transform_uses_quantity

    def _make_input_units_consistent(
        self,
        transform: Model,
        *args: LowLevelInput,
        frame: CoordinateFrameProtocol,
        input_is_quantity: bool = False,
        transform_uses_quantity: bool = False,
        **kwargs,
    ) -> tuple[LowLevelInput, ...]:
        """
        Adds or removes units from the arguments as needed so that the transform
        can be successfully evaluated.
        """
        # Validate that the input type matches what the transform expects
        if (not input_is_quantity and not transform_uses_quantity) or (
            input_is_quantity and transform_uses_quantity
        ):
            return args
        if not input_is_quantity and (
            transform_uses_quantity or transform.parameters.size
        ):
            return self._add_units_input(args, frame)
        if not transform_uses_quantity and input_is_quantity:
            return self._remove_units_input(args, frame)
        return args

    def _make_output_units_consistent(
        self,
        transform: Model,
        *args: LowLevelInput,
        frame: CoordinateFrameProtocol,
        input_is_quantity=False,
        transform_uses_quantity=False,
        **kwargs,
    ) -> tuple[LowLevelInput, ...]:
        """
        Adds or removes units from the arguments as needed so that
        the type of the output matches the input.
        """
        if not input_is_quantity and not transform_uses_quantity:
            return args

        if input_is_quantity and transform_uses_quantity:
            # make sure the output is returned in the units of the output frame
            return self._add_units_input(args, frame)
        if not input_is_quantity and (
            transform_uses_quantity or transform.parameters.size
        ):
            return self._remove_units_input(args, frame)
        if not transform_uses_quantity and input_is_quantity:
            return self._add_units_input(args, frame)
        return args

    def in_image(
        self,
        *args: LowLevelInput,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> bool | npt.NDArray[np.bool_]:
        """
        This method tests if one or more of the input world coordinates are
        contained within forward transformation's image and that it maps to
        the domain of definition of the forward transformation.
        In practical terms, this function tests
        that input world coordinate(s) can be converted to input frame and that
        it is within the forward transformation's ``bounding_box`` when
        defined.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Unit` coordinates to be inverted

        kwargs : dict
            keyword arguments to be passed either to ``backward_transform``
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : bool, numpy.ndarray
            A single boolean value or an array of boolean values with `True`
            indicating that the WCS footprint contains the coordinate
            and `False` if input is outside the footprint.

        """
        coords = self.invert(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )

        result: npt.NDArray[np.bool_] = np.isfinite(coords)
        if self.input_frame.naxes > 1:
            result = np.all(result, axis=0)

        return result

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
        inverses are defined, a numerical solution will be attempted using
        :py:meth:`numerical_inverse`.

        .. note::
            Currently numerical inverse is implemented only for 2D imaging WCS.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or `~astropy.units.Unit`
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
            Keyword arguments to be passed to :py:meth:`numerical_inverse`
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : tuple or value
            Returns a tuple of scalar or array values for each axis. Unless
            ``input_frame.naxes == 1`` when it shall return the value.
            The return type will be `~astropy.units.Quantity` objects if the
            transform returns ``Quantity`` objects, else values.

        """  # noqa: E501
        try:
            transform = self.backward_transform
        except NotImplementedError:
            transform = None

        if is_high_level(*args, low_level_wcs=self):
            msg = (
                "High Level objects are not supported with the native API. "
                "Please use the `world_to_pixel` method."
            )
            raise TypeError(msg)

        if with_bounding_box and self.bounding_box is not None:
            args = self.outside_footprint(args)

        input_is_quantity, transform_uses_quantity = self._units_are_present(
            args, transform
        )

        args = self._make_input_units_consistent(
            transform,
            *args,
            frame=self.output_frame,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )
        if transform is not None:
            akwargs = {k: v for k, v in kwargs.items() if k not in _ITER_INV_KWARGS}
            result = transform(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **akwargs,
            )
        else:
            # Always strip units for numerical inverse
            args = self._remove_units_input(args, self.output_frame)
            result = self._numerical_inverse(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **kwargs,
            )

        if with_bounding_box and self.bounding_box is not None:
            result = self.out_of_bounds(result, fill_value=fill_value)

        if not isinstance(self.input_frame, EmptyFrame):
            if self.input_frame.naxes == 1:
                result = (result,)
            result = self._make_output_units_consistent(
                transform,
                *result,
                frame=self.input_frame,
                input_is_quantity=input_is_quantity,
                transform_uses_quantity=transform_uses_quantity,
            )
            if self.input_frame.naxes == 1:
                return result[0]
        return result

    def outside_footprint(self, world_arrays):
        world_arrays = [copy(array) for array in world_arrays]

        axes_types = set(self.output_frame.axes_type)
        axes_phys_types = self.world_axis_physical_types
        footprint = self.footprint()
        not_numerical = False
        if is_high_level(world_arrays[0], low_level_wcs=self):
            not_numerical = True
            world_arrays = high_level_objects_to_values(
                *world_arrays, low_level_wcs=self
            )
        for axtyp in axes_types:
            ind = np.asarray(np.asarray(self.output_frame.axes_type) == axtyp)

            for idim, (coordinate, phys) in enumerate(
                zip(world_arrays, axes_phys_types, strict=False)
            ):
                coord = _tofloat(coordinate)
                if np.asarray(ind).sum() > 1:
                    axis_range = footprint[:, idim]
                else:
                    axis_range = footprint
                min_ax = axis_range.min()
                max_ax = axis_range.max()

                if (
                    axtyp == "SPATIAL"
                    and str(phys).endswith((".ra", ".lon"))
                    and (max_ax - min_ax) > 180
                ):
                    # most likely this coordinate is wrapped at 360
                    d = 0.5 * (min_ax + max_ax)
                    m = axis_range <= d
                    min_ax = axis_range[m].max()
                    max_ax = axis_range[~m].min()
                    outside = (coord > min_ax) & (coord < max_ax)
                else:
                    if len(world_arrays) == 1:
                        coord_ = self._remove_quantity_frame(
                            world_arrays[0], self.output_frame
                        )
                    else:
                        coord_ = self._remove_quantity_frame(
                            world_arrays, self.output_frame
                        )[idim]

                    outside = (coord_ < min_ax) | (coord_ > max_ax)
                if np.any(outside):
                    if np.isscalar(coord):
                        coord = np.nan
                    else:
                        coord[outside] = np.nan
                    world_arrays[idim] = coord
        if not_numerical:
            world_arrays = values_to_high_level_objects(
                *world_arrays, low_level_wcs=self
            )
        return world_arrays

    def out_of_bounds(self, pixel_arrays, fill_value=np.nan):
        if np.isscalar(pixel_arrays) or self.input_frame.naxes == 1:
            pixel_arrays = [pixel_arrays]

        pixel_arrays = list(pixel_arrays)
        bbox = self.bounding_box
        for idim, pix in enumerate(pixel_arrays):
            outside = (pix < bbox[idim][0]) | (pix > bbox[idim][1])
            if np.any(outside):
                if np.isscalar(pix):
                    pixel_arrays[idim] = np.nan
                else:
                    pix_ = pixel_arrays[idim].astype(float, copy=True)
                    pix_[outside] = np.nan
                    pixel_arrays[idim] = pix_
        if self.input_frame.naxes == 1:
            pixel_arrays = pixel_arrays[0]
        return pixel_arrays

    def numerical_inverse(
        self,
        *args,
        tolerance=1e-5,
        maxiter=30,
        adaptive=True,
        detect_divergence=True,
        quiet=True,
        with_bounding_box=True,
        fill_value=np.nan,
        **kwargs,
    ):
        """
        Invert coordinates from output frame to input frame using numerical
        inverse.

        .. note::
            Currently numerical inverse is implemented only for 2D imaging WCS.

        .. note::
            This method uses a combination of vectorized fixed-point
            iterations algorithm and `scipy.optimize.root`. The later is used
            for input coordinates for which vectorized algorithm diverges.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or `~astropy.units.Unit`
            Coordinates to be inverted. The number of arguments must be equal
            to the number of world coordinates given by ``world_n_dim``.

        with_bounding_box : bool, optional
             If `True` (default) values in the result which correspond to any
             of the inputs being outside the bounding_box are set to
             ``fill_value``.

        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is ``np.nan``).

        tolerance : float, optional
            *Absolute tolerance* of solution. Iteration terminates when the
            iterative solver estimates that the "true solution" is
            within this many pixels current estimate, more
            specifically, when the correction to the solution found
            during the previous iteration is smaller
            (in the sense of the L2 norm) than ``tolerance``.
            Default ``tolerance`` is 1.0e-5.

        maxiter : int, optional
            Maximum number of iterations allowed to reach a solution.
            Default is 50.

        quiet : bool, optional
            Do not throw :py:class:`NoConvergence` exceptions when
            the method does not converge to a solution with the
            required accuracy within a specified number of maximum
            iterations set by ``maxiter`` parameter. Instead,
            simply return the found solution. Default is `True`.

        adaptive : bool, optional
            Specifies whether to adaptively select only points that
            did not converge to a solution within the required
            accuracy for the next iteration. Default (`True`) is recommended.

            .. note::
               The :py:meth:`numerical_inverse` uses a vectorized
               implementation of the method of consecutive
               approximations (see ``Notes`` section below) in which it
               iterates over *all* input points *regardless* until
               the required accuracy has been reached for *all* input
               points. In some cases it may be possible that
               *almost all* points have reached the required accuracy
               but there are only a few of input data points for
               which additional iterations may be needed (this
               depends mostly on the characteristics of the geometric
               distortions for a given instrument). In this situation
               it may be advantageous to set ``adaptive`` = `True` in
               which case :py:meth:`numerical_inverse` will continue
               iterating *only* over the points that have not yet
               converged to the required accuracy.

            .. note::
               When ``detect_divergence`` is `True`,
               :py:meth:`numerical_inverse` will automatically switch
               to the adaptive algorithm once divergence has been
               detected.

        detect_divergence : bool, optional
            Specifies whether to perform a more detailed analysis
            of the convergence to a solution. Normally
            :py:meth:`numerical_inverse` may not achieve the required
            accuracy if either the ``tolerance`` or ``maxiter`` arguments
            are too low. However, it may happen that for some
            geometric distortions the conditions of convergence for
            the the method of consecutive approximations used by
            :py:meth:`numerical_inverse` may not be satisfied, in which
            case consecutive approximations to the solution will
            diverge regardless of the ``tolerance`` or ``maxiter``
            settings.

            When ``detect_divergence`` is `False`, these divergent
            points will be detected as not having achieved the
            required accuracy (without further details). In addition,
            if ``adaptive`` is `False` then the algorithm will not
            know that the solution (for specific points) is diverging
            and will continue iterating and trying to "improve"
            diverging solutions. This may result in ``NaN`` or
            ``Inf`` values in the return results (in addition to a
            performance penalties). Even when ``detect_divergence``
            is `False`, :py:meth:`numerical_inverse`, at the end of the
            iterative process, will identify invalid results
            (``NaN`` or ``Inf``) as "diverging" solutions and will
            raise :py:class:`NoConvergence` unless the ``quiet``
            parameter is set to `True`.

            When ``detect_divergence`` is `True` (default),
            :py:meth:`numerical_inverse` will detect points for which
            current correction to the coordinates is larger than
            the correction applied during the previous iteration
            **if** the requested accuracy **has not yet been
            achieved**. In this case, if ``adaptive`` is `True`,
            these points will be excluded from further iterations and
            if ``adaptive`` is `False`, :py:meth:`numerical_inverse` will
            automatically switch to the adaptive algorithm. Thus, the
            reported divergent solution will be the latest converging
            solution computed immediately *before* divergence
            has been detected.

            .. note::
               When accuracy has been achieved, small increases in
               current corrections may be possible due to rounding
               errors (when ``adaptive`` is `False`) and such
               increases will be ignored.

            .. note::
               Based on our testing using JWST NIRCAM images, setting
               ``detect_divergence`` to `True` will incur about 5-10%
               performance penalty with the larger penalty
               corresponding to ``adaptive`` set to `True`.
               Because the benefits of enabling this
               feature outweigh the small performance penalty,
               especially when ``adaptive`` = `False`, it is
               recommended to set ``detect_divergence`` to `True`,
               unless extensive testing of the distortion models for
               images from specific instruments show a good stability
               of the numerical method for a wide range of
               coordinates (even outside the image itself).

            .. note::
               Indices of the diverging inverse solutions will be
               reported in the ``divergent`` attribute of the
               raised :py:class:`NoConvergence` exception object.

        Returns
        -------
        result : tuple
            Returns a tuple of scalar or array values for each axis.

        Raises
        ------
        NoConvergence
            The iterative method did not converge to a
            solution to the required accuracy within a specified
            number of maximum iterations set by the ``maxiter``
            parameter. To turn off this exception, set ``quiet`` to
            `True`. Indices of the points for which the requested
            accuracy was not achieved (if any) will be listed in the
            ``slow_conv`` attribute of the
            raised :py:class:`NoConvergence` exception object.

            See :py:class:`NoConvergence` documentation for
            more details.

        NotImplementedError
            Numerical inverse has not been implemented for this WCS.

        ValueError
            Invalid argument values.

        Examples
        --------
        >>> from astropy.utils.data import get_pkg_data_filename
        >>> from gwcs import NoConvergence
        >>> import asdf
        >>> import numpy as np

        >>> filename = get_pkg_data_filename('data/nircamwcs.asdf', package='gwcs.tests')
        >>> with asdf.open(filename, lazy_load=False, ignore_missing_extensions=True) as af:
        ...    w = af.tree['wcs']

        >>> ra, dec = w([1,2,3], [1,1,1])
        >>> assert np.allclose(ra, [5.927628, 5.92757069, 5.92751337]);
        >>> assert np.allclose(dec, [-72.01341247, -72.01341273, -72.013413])

        >>> x, y = w.numerical_inverse(ra, dec)
        >>> assert np.allclose(x, [1.00000005, 2.00000005, 3.00000006]);
        >>> assert np.allclose(y, [1.00000004, 0.99999979, 1.00000015]);

        >>> x, y = w.numerical_inverse(ra, dec, maxiter=3, tolerance=1.0e-10, quiet=False)
        Traceback (most recent call last):
        ...
        gwcs.wcs._exception.NoConvergence: 'WCS.numerical_inverse' failed to converge to the
        requested accuracy after 3 iterations.

        >>> w.numerical_inverse(
        ...     *w([1, 300000, 3], [2, 1000000, 5], with_bounding_box=False),
        ...     adaptive=False,
        ...     detect_divergence=True,
        ...     quiet=False,
        ...     with_bounding_box=False
        ... )
        Traceback (most recent call last):
        ...
        gwcs.wcs._exception.NoConvergence: 'WCS.numerical_inverse' failed to converge to the
        requested accuracy. After 4 iterations, the solution is diverging at
        least for one input point.

        >>> # Now try to use some diverging data:
        >>> divra, divdec = w([1, 300000, 3], [2, 1000000, 5], with_bounding_box=False)
        >>> assert np.allclose(divra, [5.92762673, 148.21600848, 5.92750827])
        >>> assert np.allclose(divdec, [-72.01339464, -7.80968079, -72.01334172])
        >>> try:  # doctest: +SKIP
        ...     x, y = w.numerical_inverse(divra, divdec, maxiter=20,
        ...                                tolerance=1.0e-4, adaptive=True,
        ...                                detect_divergence=True,
        ...                                quiet=False)
        ... except NoConvergence as e:
        ...     print(f"Indices of diverging points: {e.divergent}")
        ...     print(f"Indices of poorly converging points: {e.slow_conv}")
        ...     print(f"Best solution:\\n{e.best_solution}")
        ...     print(f"Achieved accuracy:\\n{e.accuracy}")
        Indices of diverging points: None
        Indices of poorly converging points: [1]
        Best solution:
        [[1.00000040e+00 1.99999841e+00]
         [6.33507833e+17 3.40118820e+17]
         [3.00000038e+00 4.99999841e+00]]
        Achieved accuracy:
        [[2.75925982e-05 1.18471543e-05]
         [3.65405005e+04 1.31364188e+04]
         [2.76552923e-05 1.14789013e-05]]

        """  # noqa: E501
        return self._numerical_inverse(
            *self._remove_units_input(args, self.output_frame),
            tolerance=tolerance,
            maxiter=maxiter,
            adaptive=adaptive,
            detect_divergence=detect_divergence,
            quiet=quiet,
            with_bounding_box=with_bounding_box,
            fill_value=fill_value,
            **kwargs,
        )

    def _numerical_inverse(
        self,
        *args,
        tolerance=1e-5,
        maxiter=30,
        adaptive=True,
        detect_divergence=True,
        quiet=True,
        with_bounding_box=True,
        fill_value=np.nan,
        **kwargs,
    ):
        args_shape = np.shape(args)
        nargs = args_shape[0]
        arg_dim = len(args_shape) - 1

        if nargs != self.world_n_dim:
            msg = (
                "Number of input coordinates is different from "
                "the number of defined world coordinates in the "
                f"WCS ({self.world_n_dim:d})"
            )
            raise ValueError(msg)

        if self.world_n_dim != self.pixel_n_dim:
            msg = (
                "Support for iterative inverse for transformations with "
                "different number of inputs and outputs was not implemented."
            )
            raise NotImplementedError(msg)

        # initial guess:
        if nargs == 2 and self._approx_inverse is None:
            self._calc_approx_inv(max_inv_pix_error=5, inv_degree=None)

        if self._approx_inverse is None:
            if self.bounding_box is None:
                x0 = np.ones(self.pixel_n_dim)
            else:
                x0 = np.mean(self.bounding_box, axis=-1)

        if arg_dim == 0:
            argsi = args

            if nargs == 2 and self._approx_inverse is not None:
                x0 = self._approx_inverse(*argsi)
                if not np.all(np.isfinite(x0)):
                    return [np.array(np.nan) for _ in range(nargs)]

            result = tuple(
                self._vectorized_fixed_point(
                    x0,
                    argsi,
                    tolerance=tolerance,
                    maxiter=maxiter,
                    adaptive=adaptive,
                    detect_divergence=detect_divergence,
                    quiet=quiet,
                    with_bounding_box=with_bounding_box,
                    fill_value=fill_value,
                )
                .T.ravel()
                .tolist()
            )

        else:
            arg_shape = args_shape[1:]
            nelem = np.prod(arg_shape)

            args = np.reshape(args, (nargs, nelem))

            if self._approx_inverse is None:
                x0 = np.full((nelem, nargs), x0)
            else:
                x0 = np.array(self._approx_inverse(*args)).T

            result = self._vectorized_fixed_point(
                x0,
                args.T,
                tolerance=tolerance,
                maxiter=maxiter,
                adaptive=adaptive,
                detect_divergence=detect_divergence,
                quiet=quiet,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
            ).T

            result = tuple(np.reshape(result, args_shape))

        return result

    def _vectorized_fixed_point(
        self,
        pix0,
        world,
        tolerance,
        maxiter,
        adaptive,
        detect_divergence,
        quiet,
        with_bounding_box,
        fill_value,
    ):
        # ############################################################
        # #            INITIALIZE ITERATIVE PROCESS:                ##
        # ############################################################

        # make a copy of the initial approximation
        pix0 = np.atleast_2d(np.array(pix0))  # 0-order solution
        pix = np.array(pix0)

        world0 = np.atleast_2d(np.array(world))
        world = np.array(world0)

        # estimate pixel scale using approximate algorithm
        # from https://trs.jpl.nasa.gov/handle/2014/40409
        if self.bounding_box is None:
            crpix = np.ones(self.pixel_n_dim)
        else:
            crpix = np.mean(self.bounding_box, axis=-1)

        l1, phi1 = np.deg2rad(self.__call__(*(crpix - 0.5)))
        l2, phi2 = np.deg2rad(self.__call__(*(crpix + [-0.5, 0.5])))  # noqa: RUF005
        l3, phi3 = np.deg2rad(self.__call__(*(crpix + 0.5)))
        l4, phi4 = np.deg2rad(self.__call__(*(crpix + [0.5, -0.5])))  # noqa: RUF005
        area = np.abs(
            0.5
            * (
                (l4 - l2) * (np.sin(phi1) - np.sin(phi3))
                + (l1 - l3) * (np.sin(phi2) - np.sin(phi4))
            )
        )
        inv_pscale = 1 / np.rad2deg(np.sqrt(area))

        # form equation:
        def f(x):
            w = np.array(self.__call__(*(x.T), with_bounding_box=False)).T
            dw = np.mod(np.subtract(w, world) - 180.0, 360.0) - 180.0
            return np.add(inv_pscale * dw, x)

        def froot(x):
            return (
                np.mod(
                    np.subtract(self.__call__(*x, with_bounding_box=False), worldi)
                    - 180.0,
                    360.0,
                )
                - 180.0
            )

        # compute correction:
        def correction(pix):
            p1 = f(pix)
            p2 = f(p1)
            d = p2 - 2.0 * p1 + pix
            idx = np.where(d != 0)
            corr = pix - p2
            corr[idx] = np.square(p1[idx] - pix[idx]) / d[idx]
            return corr

        # initial iteration:
        dpix = correction(pix)

        # Update initial solution:
        pix -= dpix

        # Norm (L2) squared of the correction:
        dn = np.sum(dpix * dpix, axis=1)
        dnprev = dn.copy()  # if adaptive else dn
        tol2 = tolerance**2

        # Prepare for iterative process
        k = 1
        ind = None
        inddiv = None

        # Turn off numpy runtime warnings for 'invalid' and 'over':
        old_invalid = np.geterr()["invalid"]
        old_over = np.geterr()["over"]
        np.seterr(invalid="ignore", over="ignore")

        # ############################################################
        # #                NON-ADAPTIVE ITERATIONS:                 ##
        # ############################################################
        if not adaptive:
            # Fixed-point iterations:
            while np.nanmax(dn) >= tol2 and k < maxiter:
                # Find correction to the previous solution:
                dpix = correction(pix)

                # Compute norm (L2) squared of the correction:
                dn = np.sum(dpix * dpix, axis=1)

                # Check for divergence (we do this in two stages
                # to optimize performance for the most common
                # scenario when successive approximations converge):

                if detect_divergence:
                    divergent = dn >= dnprev
                    if np.any(divergent):
                        # Find solutions that have not yet converged:
                        slowconv = dn >= tol2
                        (inddiv,) = np.where(divergent & slowconv)

                        if inddiv.shape[0] > 0:
                            # Update indices of elements that
                            # still need correction:
                            conv = dn < dnprev
                            iconv = np.where(conv)

                            # Apply correction:
                            dpixgood = dpix[iconv]
                            pix[iconv] -= dpixgood
                            dpix[iconv] = dpixgood

                            # For the next iteration choose
                            # non-divergent points that have not yet
                            # converged to the requested accuracy:
                            (ind,) = np.where(slowconv & conv)
                            world = world[ind]
                            dnprev[ind] = dn[ind]
                            k += 1

                            # Switch to adaptive iterations:
                            adaptive = True
                            break

                    # Save current correction magnitudes for later:
                    dnprev = dn

                # Apply correction:
                pix -= dpix
                k += 1

        # ############################################################
        # #                  ADAPTIVE ITERATIONS:                   ##
        # ############################################################
        if adaptive:
            if ind is None:
                (ind,) = np.where(np.isfinite(pix).all(axis=1))
                world = world[ind]

            # "Adaptive" fixed-point iterations:
            while ind.shape[0] > 0 and k < maxiter:
                # Find correction to the previous solution:
                dpixnew = correction(pix[ind])

                # Compute norm (L2) of the correction:
                dnnew = np.sum(np.square(dpixnew), axis=1)

                # Bookkeeping of corrections:
                dnprev[ind] = dn[ind].copy()
                dn[ind] = dnnew

                if detect_divergence:
                    # Find indices of pixels that are converging:
                    conv = np.logical_or(dnnew < dnprev[ind], dnnew < tol2)
                    if not np.all(conv):
                        conv = np.ones_like(dnnew, dtype=bool)
                    iconv = np.where(conv)
                    iiconv = ind[iconv]

                    # Apply correction:
                    dpixgood = dpixnew[iconv]
                    pix[iiconv] -= dpixgood
                    dpix[iiconv] = dpixgood

                    # Find indices of solutions that have not yet
                    # converged to the requested accuracy
                    # AND that do not diverge:
                    (subind,) = np.where((dnnew >= tol2) & conv)

                else:
                    # Apply correction:
                    pix[ind] -= dpixnew
                    dpix[ind] = dpixnew

                    # Find indices of solutions that have not yet
                    # converged to the requested accuracy:
                    (subind,) = np.where(dnnew >= tol2)

                # Choose solutions that need more iterations:
                ind = ind[subind]
                world = world[subind]

                k += 1

        # ############################################################
        # #         FINAL DETECTION OF INVALID, DIVERGING,          ##
        # #         AND FAILED-TO-CONVERGE POINTS                   ##
        # ############################################################
        # Identify diverging and/or invalid points:
        invalid = (~np.all(np.isfinite(pix), axis=1)) & (
            np.all(np.isfinite(world0), axis=1)
        )

        # When detect_divergence is False, dnprev is outdated
        # (it is the norm of the very first correction).
        # Still better than nothing...
        (inddiv,) = np.where(((dn >= tol2) & (dn >= dnprev)) | invalid)
        if inddiv.shape[0] == 0:
            inddiv = None

        # If there are divergent points, attempt to find a solution using
        # scipy's 'hybr' method:
        if detect_divergence and inddiv is not None and inddiv.size:
            bad = []
            for idx in inddiv:
                worldi = world0[idx]
                result = optimize.root(
                    froot,
                    pix0[idx],
                    method="hybr",
                    tol=tolerance / (np.linalg.norm(pix0[idx]) + 1),
                    options={"maxfev": 2 * maxiter},
                )

                if result["success"]:
                    pix[idx, :] = result["x"]
                    invalid[idx] = False
                else:
                    bad.append(idx)

            inddiv = np.array(bad, dtype=int) if bad else None

        # Identify points that did not converge within 'maxiter'
        # iterations:
        if k >= maxiter:
            (ind,) = np.where((dn >= tol2) & (dn < dnprev) & (~invalid))
            if ind.shape[0] == 0:
                ind = None
        else:
            ind = None

        # Restore previous numpy error settings:
        np.seterr(invalid=old_invalid, over=old_over)

        # ############################################################
        # #  RAISE EXCEPTION IF DIVERGING OR TOO SLOWLY CONVERGING  ##
        # #  DATA POINTS HAVE BEEN DETECTED:                        ##
        # ############################################################
        if (ind is not None or inddiv is not None) and not quiet:
            if inddiv is None:
                msg = (
                    "'WCS.numerical_inverse' failed to "
                    f"converge to the requested accuracy after {k:d} "
                    "iterations."
                )
                raise NoConvergence(
                    msg,
                    best_solution=pix,
                    accuracy=np.abs(dpix),
                    niter=k,
                    slow_conv=ind,
                    divergent=None,
                )
            msg = (
                "'WCS.numerical_inverse' failed to "
                "converge to the requested accuracy.\n"
                f"After {k:d} iterations, the solution is diverging "
                "at least for one input point."
            )
            raise NoConvergence(
                msg,
                best_solution=pix,
                accuracy=np.abs(dpix),
                niter=k,
                slow_conv=ind,
                divergent=inddiv,
            )

        if with_bounding_box and self.bounding_box is not None:
            # find points outside the bounding box and replace their values
            # with fill_value
            valid = np.logical_not(invalid)
            in_bb = np.ones_like(invalid, dtype=np.bool_)

            for c, (x1, x2) in zip(pix[valid].T, self.bounding_box, strict=False):
                in_bb[valid] &= (c >= x1) & (c <= x2)
            pix[np.logical_not(in_bb)] = fill_value

        return pix

    def transform(
        self,
        from_frame: str | CoordinateFrameProtocol,
        to_frame: str | CoordinateFrameProtocol,
        *args: float | np.ndarray,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelArray, ...] | LowLevelArray:
        """
        Transform positions between two frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        args : float or array-like
            Inputs in ``from_frame``, separate inputs for each dimension.
        with_bounding_box : bool, optional
            If True(default) values in the result which correspond to any of
            the inputs being outside the bounding_box are set to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box
            (default is np.nan).
        """
        # Pull the steps and their indices from the pipeline
        # -> this also turns the frame name strings into frame objects
        from_step = self._get_step(from_frame)
        to_step = self._get_step(to_frame)
        transform = self.get_transform(from_step.step.frame, to_step.step.frame)

        if transform is None:
            msg = f"No transformation found from {from_frame} to {to_frame}."
            raise ValueError(msg)

        # Get the frame objects from the wcs pipeline
        from_frame_obj = self.get_frame(from_frame)
        to_frame_obj = self.get_frame(to_frame)

        input_is_quantity, transform_uses_quantity = self._units_are_present(
            args, transform
        )
        args = self._make_input_units_consistent(
            transform,
            *args,
            frame=from_frame_obj,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )

        result = transform(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )
        if to_frame_obj is not None:
            if to_frame_obj.naxes == 1:
                result = (result,)
            result = self._make_output_units_consistent(
                transform,
                *result,
                frame=to_frame_obj,
                input_is_quantity=input_is_quantity,
                transform_uses_quantity=transform_uses_quantity,
            )
        if to_frame_obj is not None and to_frame_obj.naxes == 1:
            return result[0]
        return result

    @property
    def name(self) -> str:
        """Return the name for this WCS."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name for the WCS."""
        self._name = value

    def __str__(self) -> str:
        from astropy.table import Table

        col1 = [str(step.frame) for step in self._pipeline]
        col2: list[str | None] = []
        for item in self._pipeline[:-1]:
            model = item.transform
            if model is None:
                col2.append(None)
            elif model.name is not None:
                col2.append(model.name)
            else:
                col2.append(model.__class__.__name__)
        col2.append(None)
        t = Table([col1, col2], names=["From", "Transform"])
        return str(t)

    def __repr__(self) -> str:
        return (
            f"<WCS(output_frame={self.output_frame}, input_frame={self.input_frame}, "
            f"forward_transform={self.forward_transform})>"
        )

    def footprint(
        self, bounding_box=None, center=False, axis_type: AxisType | str | None = None
    ):
        """
        Return the footprint in world coordinates.

        Parameters
        ----------
        bounding_box : tuple of floats: (start, stop)
            ``prop: bounding_box``
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.
        axis_type : AxisType
            A supported ``output_frame.axes_type`` or ``"all"`` (default).
            One of [``'spatial'``, ``'spectral'``, ``'temporal'``] or a custom type.

        Returns
        -------
        coord : ndarray
            Array of coordinates in the output_frame mapping
            corners to the output frame. For spatial coordinates the order
            is clockwise, starting from the bottom left corner.

        """
        axis_type = AxisType.from_input("all" if axis_type is None else axis_type)

        def _order_clockwise(v):
            return np.asarray(
                [
                    [v[0][0], v[1][0]],
                    [v[0][0], v[1][1]],
                    [v[0][1], v[1][1]],
                    [v[0][1], v[1][0]],
                ]
            ).T

        if bounding_box is None:
            if self.bounding_box is None:
                msg = "Need a valid bounding_box to compute the footprint."
                raise TypeError(msg)
            bb = self.bounding_box.bounding_box(order="F")
        else:
            bb = bounding_box

        if self.output_frame is None:
            msg = "Footprint requires a defined output_frame."
            raise ValueError(msg)

        all_spatial = all(t.lower() == "spatial" for t in self.output_frame.axes_type)
        if self.output_frame.naxes == 1:
            if isinstance(bb[0], u.Quantity):
                bb = np.asarray([b.value for b in bb]) * bb[0].unit
            vertices = (bb,)
        elif all_spatial:
            vertices = _order_clockwise(
                [self._remove_units_input(b, self.input_frame) for b in bb]
            )
        else:
            vertices = np.array(list(itertools.product(*bb))).T  # type: ignore[assignment]

        # workaround an issue with bbox with quantity, interval needs to be a cquantity,
        # not a list of quantities strip units
        if center:
            vertices = to_index(vertices)

        result = np.asarray(self.__call__(*vertices, with_bounding_box=False))

        if axis_type is AxisType.SPATIAL and all_spatial:
            return result.T

        if axis_type != "all":
            axtyp_ind = (
                np.array([AxisType.from_input(t) for t in self.output_frame.axes_type])
                == axis_type
            )
            if not axtyp_ind.any():
                msg = f'This WCS does not have axis of type "{axis_type}".'
                raise ValueError(msg)
            if len(axtyp_ind) > 1:
                result = np.asarray([(r.min(), r.max()) for r in result[axtyp_ind]])

            if axis_type is AxisType.SPATIAL:
                result = _order_clockwise(result)
            else:
                result.sort()
                result = np.squeeze(result)

        if self.output_frame.naxes == 1:
            return np.array([result]).T

        return result.T

    def fix_inputs(
        self, fixed: dict[str | int, LowLevelArray | u.Quantity | float | np.number]
    ) -> Self:
        """
        Return a new unique WCS by fixing inputs to constant values.

        Parameters
        ----------
        fixed : dict
            Keyword arguments with fixed values corresponding to ``self.selector``.

        Returns
        -------
        new_wcs : `WCS`
            A new unique WCS corresponding to the values in ``fixed``.

        Examples
        --------
        >>> w = WCS(pipeline, selector={"spectral_order": [1, 2]}) # doctest: +SKIP
        >>> new_wcs = w.set_inputs(spectral_order=2) # doctest: +SKIP
        >>> new_wcs.inputs # doctest: +SKIP
            ("x", "y")

        """
        return type(self)(
            [
                (self.pipeline[0].frame, fix_inputs(self.pipeline[0].transform, fixed)),
                *self.pipeline[1:],
            ]
        )

    def to_fits_sip(
        self,
        bounding_box=None,
        max_pix_error=0.25,
        degree=None,
        max_inv_pix_error=0.25,
        inv_degree=None,
        npoints=32,
        crpix=None,
        projection="TAN",
        verbose=False,
    ):
        """
        Construct a SIP-based approximation to the WCS for the axes
        corresponding to the `~gwcs.coordinate_frames.CelestialFrame`
        in the form of a FITS header.

        The default mode in using this attempts to achieve roughly 0.25 pixel
        accuracy over the whole image.

        Parameters
        ----------
        bounding_box : tuple, optional
            A pair of tuples, each consisting of two numbers
            Represents the range of pixel values in both dimensions
            ((xmin, xmax), (ymin, ymax))

        max_pix_error : float, optional
            Maximum allowed error over the domain of the pixel array. This
            error is the equivalent pixel error that corresponds to the maximum
            error in the output coordinate resulting from the fit based on
            a nominal plate scale. Ignored when ``degree`` is an integer or
            a list with a single degree.

        degree : int, iterable, None, optional
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_pixel_error`` is ignored.

        max_inv_pix_error : float, optional
            Maximum allowed inverse error over the domain of the pixel array
            in pixel units. If None, no inverse is generated. Ignored when
            ``degree`` is an integer or a list with a single degree.

        inv_degree : int, iterable, None, optional
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_inv_pixel_error`` is ignored.

        npoints : int, optional
            The number of points in each dimension to sample the bounding box
            for use in the SIP fit. Minimum number of points is 3.

        crpix : list of float, None, optional
            Coordinates (1-based) of the reference point for the new FITS WCS.
            When not provided, i.e., when set to `None` (default) the reference
            pixel will be chosen near the center of the bounding box for axes
            corresponding to the celestial frame.

        projection : str, `~astropy.modeling.projections.Pix2SkyProjection`, optional
            Projection to be used for the created FITS WCS. It can be specified
            as a string of three characters specifying a FITS projection code
            from Table 13 in
            `Representations of World Coordinates in FITS \
            <https://doi.org/10.1051/0004-6361:20021326>`_
            (Paper I), Greisen, E. W., and Calabretta, M. R., A & A, 395,
            1061-1075, 2002. Alternatively, it can be an instance of one of the
            `astropy's Pix2Sky_* <https://docs.astropy.org/en/stable/modeling/\
            reference_api.html#module-astropy.modeling.projections>`_
            projection models inherited from
            :py:class:`~astropy.modeling.projections.Pix2SkyProjection`.

        verbose : bool, optional
            Print progress of fits.

        Returns
        -------
        FITS header with all SIP WCS keywords

        Raises
        ------
        ValueError
            If the WCS is not at least 2D, an exception will be raised. If the
            specified accuracy (both forward and inverse, both rms and maximum)
            is not achieved an exception will be raised.

        Notes
        -----

        Use of this requires a judicious choice of required accuracies.
        Attempts to use higher degrees (~7 or higher) will typically fail due
        to floating point problems that arise with high powers.

        """
        _, _, celestial_group = self._separable_groups(detect_celestial=True)
        if celestial_group is None:
            msg = "The to_fits_sip requires an output celestial frame."
            raise ValueError(msg)

        return self._to_fits_sip(
            celestial_group=celestial_group,
            keep_axis_position=False,
            bounding_box=bounding_box,
            max_pix_error=max_pix_error,
            degree=degree,
            max_inv_pix_error=max_inv_pix_error,
            inv_degree=inv_degree,
            npoints=npoints,
            crpix=crpix,
            projection=projection,
            matrix_type="CD",
            verbose=verbose,
        )

    def _to_fits_sip(
        self,
        celestial_group,
        keep_axis_position,
        bounding_box,
        max_pix_error,
        degree,
        max_inv_pix_error,
        inv_degree,
        npoints,
        crpix,
        projection,
        matrix_type,
        verbose,
    ):
        r"""
        Construct a SIP-based approximation to the WCS for the axes
        corresponding to the `~gwcs.coordinate_frames.CelestialFrame`
        in the form of a FITS header.

        The default mode in using this attempts to achieve roughly 0.25 pixel
        accuracy over the whole image.

        Below we describe only parameters additional to the ones explained for
        `to_fits_sip`.

        Other Parameters
        ----------------
        frame : gwcs.coordinate_frames.CelestialFrame
            A celestial frame.

        celestial_group : list of ``_WorldAxisInfo``
            A group of two celestial axes to be represented using standard
            image FITS WCS and maybe ``-SIP`` polynomials.

        keep_axis_position : bool
            This parameter controls whether to keep/preserve output axes
            indices in this WCS object when creating FITS WCS and create a FITS
            header with ``CTYPE`` axes indices preserved from the ``frame``
            object or whether to reset the indices of output celestial axes
            to 1 and 2 with ``CTYPE1``, ``CTYPE2``. Default is `False`.

            .. warning::
                Returned header will have both ``NAXIS`` and ``WCSAXES`` set
                to 2. If ``max(axes_mapping) > 2`` this will lead to an invalid
                WCS. It is caller's responsibility to adjust NAXIS to a valid
                value.

            .. note::
                The ``lon``/``lat`` order is still preserved regardless of this
                setting.

        matrix_type : {'CD', 'PC-CDELT1', 'PC-SUM1', 'PC-DET1', 'PC-SCALE'}
            Specifies formalism (``PC`` or ``CD``) to be used for the linear
            transformation matrix and normalization for the ``PC`` matrix
            *when non-linear polynomial terms are not required to achieve
            requested accuracy*.

            .. note:: ``CD`` matrix is always used when requested SIP
                approximation accuracy requires non-linear terms (when
                ``CTYPE`` ends in ``-SIP``). This parameter is ignored when
                non-linear polynomial terms are used.

            - ``'CD'``: use ``CD`` matrix;

            - ``'PC-CDELT1'``: set ``PC=CD`` and ``CDELTi=1``. This is the
              behavior of `~astropy.wcs.WCS.to_header` method;

            - ``'PC-SUM1'``: normalize ``PC`` matrix such that sum
              of its squared elements is 1: :math:`\Sigma PC_{ij}^2=1`;

            - ``'PC-DET1'``: normalize ``PC`` matrix such that :math:`|\det(PC)|=1`;

            - ``'PC-SCALE'``: normalize ``PC`` matrix such that ``CDELTi``
              are estimates of the linear pixel scales.

        Returns
        -------
        FITS header with all SIP WCS keywords

        Raises
        ------
        ValueError
            If the WCS is not at least 2D, an exception will be raised. If the
            specified accuracy (both forward and inverse, both rms and maximum)
            is not achieved an exception will be raised.

        """
        if isinstance(matrix_type, str):
            matrix_type = matrix_type.upper()

        if matrix_type not in ["CD", "PC-CDELT1", "PC-SUM1", "PC-DET1", "PC-SCALE"]:
            msg = f"Unsupported 'matrix_type' value: {matrix_type!r}."
            raise ValueError(msg)

        if npoints < 8:
            msg = "Number of sampling points is too small. 'npoints' must be >= 8."
            raise ValueError(msg)

        if isinstance(projection, str):
            projection = projection.upper()
            try:
                sky2pix_proj = getattr(projections, f"Sky2Pix_{projection}")(
                    name=projection
                )
            except AttributeError as err:
                msg = f"Unsupported FITS WCS sky projection: {projection}"
                raise ValueError(msg) from err

        elif isinstance(projection, projections.Sky2PixProjection):
            sky2pix_proj = projection
            projection = projection.name
            if (
                not projection
                or not isinstance(projection, str)
                or len(projection) != 3
            ):
                msg = f"Unsupported FITS WCS sky projection: {sky2pix_proj}"
                raise ValueError(msg)
            try:
                getattr(projections, f"Sky2Pix_{projection}")()
            except AttributeError as err:
                msg = f"Unsupported FITS WCS sky projection: {projection}"
                raise ValueError(msg) from err

        else:
            msg = (
                "'projection' must be either a FITS WCS string projection code "
                "or an instance of astropy.modeling.projections.Pix2SkyProjection."
            )
            raise TypeError(msg)

        frame = celestial_group[0].frame

        lon_axis = frame.axes_order[0]
        lat_axis = frame.axes_order[1]

        # identify input axes:
        input_axes = []
        for wax in celestial_group:
            input_axes.extend(wax.input_axes)
        input_axes = sorted(set(input_axes))

        if len(input_axes) != 2:
            msg = "Only CelestialFrame that correspond to two input axes are supported."
            raise ValueError(msg)

        # Axis number for FITS axes.
        # iax? - image axes; nlon, nlat - celestial axes:
        if keep_axis_position:
            nlon = lon_axis + 1
            nlat = lat_axis + 1
            iax1, iax2 = (i + 1 for i in input_axes)
        else:
            nlon, nlat = (1, 2) if lon_axis < lat_axis else (2, 1)
            iax1 = 1
            iax2 = 2

        # Determine reference points.
        if bounding_box is None and self.bounding_box is None:
            msg = "A bounding_box is needed to proceed."
            raise ValueError(msg)
        if bounding_box is None:
            bounding_box = self.bounding_box

        first_bound = bounding_box[0][0]
        if isinstance(first_bound, u.Quantity):
            bounding_box = [
                self._remove_units_input(bb, self.input_frame) for bb in bounding_box
            ]
        bb_center = np.mean(bounding_box, axis=1)

        fixi_dict = {
            k: bb_center[k] for k in set(range(self.pixel_n_dim)).difference(input_axes)
        }

        # Once that bug is fixed, the code below can be replaced with fix_inputs
        # statement commented out immediately above.
        transform = fix_transform_inputs(self.forward_transform, fixi_dict)

        transform = transform | Mapping(
            (lon_axis, lat_axis), n_inputs=self.forward_transform.n_outputs
        )

        (xmin, xmax) = bounding_box[input_axes[0]]
        (ymin, ymax) = bounding_box[input_axes[1]]

        # 0-based crpix:
        if crpix is None:
            crpix1 = round(bb_center[input_axes[0]], 1)
            crpix2 = round(bb_center[input_axes[1]], 1)
        else:
            crpix1 = crpix[0] - 1
            crpix2 = crpix[1] - 1

        if isinstance(first_bound, u.Quantity):
            crpix1 = u.Quantity(crpix1, first_bound.unit)
            crpix2 = u.Quantity(crpix2, first_bound.unit)

        # check that the bounding box has some reasonable size:
        if (xmax - xmin) < 1 or (ymax - ymin) < 1:
            msg = "Bounding box is too small for fitting a SIP polynomial"
            raise ValueError(msg)

        lon, lat = transform(crpix1, crpix2)
        pole = _compute_lon_pole((lon, lat), sky2pix_proj)
        if isinstance(lon, u.Quantity):
            pole = u.Quantity(pole, lon.unit)

        # Now rotate to native system and deproject. Recall that transform
        # expects pixels in the original coordinate system, but the SIP
        # transform is relative to crpix coordinates, thus the initial shift.
        ntransform = (
            (Shift(crpix1) & Shift(crpix2))
            | transform
            | RotateCelestial2Native(lon, lat, pole)
            | sky2pix_proj
        )

        # standard sampling:
        crpix_ = [crpix1, crpix2]
        if isinstance(crpix1, u.Quantity):
            crpix_ = self._remove_units_input(crpix_, self.input_frame)
        u_grid, v_grid = make_sampling_grid(
            npoints, tuple(bounding_box[k] for k in input_axes), crpix=crpix_
        )
        if isinstance(crpix1, u.Quantity):
            u_grid = u.Quantity(u_grid, crpix1.unit)
            v_grid = u.Quantity(v_grid, crpix2.unit)

        undist_x, undist_y = ntransform(u_grid, v_grid)

        # Double sampling to check if sampling is sufficient.
        ud, vd = make_sampling_grid(
            2 * npoints,
            tuple(bounding_box[k] for k in input_axes),
            crpix=crpix_,
        )
        if isinstance(crpix1, u.Quantity):
            ud = u.Quantity(ud, crpix1.unit)
            vd = u.Quantity(vd, crpix2.unit)

        undist_xd, undist_yd = ntransform(ud, vd)

        input_0 = 0
        input_1 = 1
        if isinstance(crpix1, u.Quantity):
            input_0 = u.Quantity(0, crpix1.unit)
            input_1 = u.Quantity(1, crpix1.unit)

        # Determine approximate pixel scale in order to compute error threshold
        # from the specified pixel error. Computed at the center of the array.
        x0, y0 = ntransform(input_0, input_0)
        xx, xy = ntransform(input_1, input_0)
        yx, yy = ntransform(input_0, input_1)
        pixarea = np.abs((xx - x0) * (yy - y0) - (xy - y0) * (yx - x0))
        plate_scale = np.sqrt(pixarea)

        plate_scale = (
            plate_scale.value if isinstance(plate_scale, u.Quantity) else plate_scale
        )
        u_grid = u_grid.value if isinstance(u_grid, u.Quantity) else u_grid
        v_grid = v_grid.value if isinstance(v_grid, u.Quantity) else v_grid
        undist_x = undist_x.value if isinstance(undist_x, u.Quantity) else undist_x
        undist_y = undist_y.value if isinstance(undist_y, u.Quantity) else undist_y
        ud = ud.value if isinstance(ud, u.Quantity) else ud
        vd = vd.value if isinstance(vd, u.Quantity) else vd
        undist_xd = undist_xd.value if isinstance(undist_xd, u.Quantity) else undist_xd
        undist_yd = undist_yd.value if isinstance(undist_yd, u.Quantity) else undist_yd

        # The fitting section.
        if verbose:
            sys.stdout.write("\nFitting forward SIP ...")
        fit_poly_x, fit_poly_y, max_resid = fit_2D_poly(
            degree,
            max_pix_error,
            plate_scale,
            u_grid,
            v_grid,
            undist_x,
            undist_y,
            ud,
            vd,
            undist_xd,
            undist_yd,
            verbose=verbose,
        )

        # The following is necessary to put the fit into the SIP formalism.
        cdmat, sip_poly_x, sip_poly_y = reform_poly_coefficients(fit_poly_x, fit_poly_y)
        # cdmat = np.array([[fit_poly_x.c1_0.value, fit_poly_x.c0_1.value],
        #                   [fit_poly_y.c1_0.value, fit_poly_y.c0_1.value]])
        det = cdmat[0][0] * cdmat[1][1] - cdmat[0][1] * cdmat[1][0]
        U = (cdmat[1][1] * undist_x - cdmat[0][1] * undist_y) / det
        V = (-cdmat[1][0] * undist_x + cdmat[0][0] * undist_y) / det
        detd = cdmat[0][0] * cdmat[1][1] - cdmat[0][1] * cdmat[1][0]
        Ud = (cdmat[1][1] * undist_xd - cdmat[0][1] * undist_yd) / detd
        Vd = (-cdmat[1][0] * undist_xd + cdmat[0][0] * undist_yd) / detd

        if max_inv_pix_error:
            if verbose:
                sys.stdout.write("\nFitting inverse SIP ...")
            fit_inv_poly_u, fit_inv_poly_v, max_inv_resid = fit_2D_poly(
                inv_degree,
                max_inv_pix_error,
                1,
                U,
                V,
                u_grid - U,
                v_grid - V,
                Ud,
                Vd,
                ud - Ud,
                vd - Vd,
                verbose=verbose,
            )

        # create header with WCS info:
        w = celestial_frame_to_wcs(frame.reference_frame, projection=projection)
        w.wcs.crval = [
            lon.value if isinstance(lon, u.Quantity) else lon,
            lat.value if isinstance(lat, u.Quantity) else lat,
        ]
        w.wcs.crpix = [
            crpix1.value if isinstance(crpix1, u.Quantity) else crpix1 + 1,
            crpix2.value if isinstance(crpix2, u.Quantity) else crpix2 + 1,
        ]
        w.wcs.pc = cdmat if nlon < nlat else cdmat[::-1]
        w.wcs.set()
        hdr = w.to_header(True)

        # data array info:
        hdr.insert(0, ("NAXIS", 2, "number of array dimensions"))
        hdr.insert(1, (f"NAXIS{iax1:d}", int(xmax) + 1))
        hdr.insert(2, (f"NAXIS{iax2:d}", int(ymax) + 1))
        if len(hdr["NAXIS*"]) != 3:
            msg = "NAXIS* should have 3 axes"
            raise ValueError(msg)

        # list of celestial axes related keywords:
        cel_kwd = ["CRVAL", "CTYPE", "CUNIT"]

        # Add SIP info:
        if fit_poly_x.degree > 1:
            mat_kind = "CD"
            # CDELT is not used with CD matrix (PC->CD later):
            del hdr["CDELT?"]

            hdr["CTYPE1"] = hdr["CTYPE1"].strip() + "-SIP"
            hdr["CTYPE2"] = hdr["CTYPE2"].strip() + "-SIP"
            hdr["A_ORDER"] = fit_poly_x.degree
            hdr["B_ORDER"] = fit_poly_x.degree
            store_2D_coefficients(hdr, sip_poly_x, "A")
            store_2D_coefficients(hdr, sip_poly_y, "B")
            hdr["sipmxerr"] = (max_resid, "Max diff from GWCS (equiv pix).")

            if max_inv_pix_error:
                hdr["AP_ORDER"] = fit_inv_poly_u.degree
                hdr["BP_ORDER"] = fit_inv_poly_u.degree
                store_2D_coefficients(hdr, fit_inv_poly_u, "AP", keeplinear=True)
                store_2D_coefficients(hdr, fit_inv_poly_v, "BP", keeplinear=True)
                hdr["sipiverr"] = (max_inv_resid, "Max diff for inverse (pixels)")

        else:
            if matrix_type.startswith("PC"):
                mat_kind = "PC"
                cel_kwd.append("CDELT")

                if matrix_type == "PC-CDELT1":
                    cdelt = [1.0, 1.0]

                elif matrix_type == "PC-SUM1":
                    norm = np.sqrt(np.sum(w.wcs.pc**2))
                    cdelt = [norm, norm]

                elif matrix_type == "PC-DET1":
                    det_pc = np.linalg.det(w.wcs.pc)
                    norm = np.sqrt(np.abs(det_pc))
                    cdelt = [norm, np.sign(det_pc) * norm]

                elif matrix_type == "PC-SCALE":
                    cdelt = proj_plane_pixel_scales(w)

                for i in range(1, 3):
                    s = cdelt[i - 1]
                    hdr[f"CDELT{i}"] = s
                    for j in range(1, 3):
                        pc_kwd = f"PC{i}_{j}"
                        if pc_kwd in hdr:
                            hdr[pc_kwd] = w.wcs.pc[i - 1, j - 1] / s

            else:
                mat_kind = "CD"
                del hdr["CDELT?"]

            hdr["sipmxerr"] = (max_resid, "Max diff from GWCS (equiv pix).")

        # Construct CD matrix while remapping input axes.
        # We do not update comments to typical comments for CD matrix elements
        # (such as 'partial of second axis coordinate w.r.t. y'), because
        # when input frame has number of axes > 2, then imaging
        # axes arbitrary.
        old_nlon, old_nlat = (1, 2) if nlon < nlat else (2, 1)

        # Remap input axes (CRPIX) and output axes-related parameters
        # (CRVAL, CUNIT, CTYPE, CD/PC). This has to be done in two steps to avoid
        # name conflicts (i.e., swapping CRPIX1<->CRPIX2).

        # remap input axes:
        axis_rename = {}
        if iax1 != 1:
            axis_rename["CRPIX1"] = f"CRPIX{iax1}"
        if iax2 != 2:
            axis_rename["CRPIX2"] = f"CRPIX{iax2}"

        # CP/PC matrix:
        axis_rename[f"PC{old_nlon}_1"] = f"{mat_kind}{nlon}_{iax1}"
        axis_rename[f"PC{old_nlon}_2"] = f"{mat_kind}{nlon}_{iax2}"
        axis_rename[f"PC{old_nlat}_1"] = f"{mat_kind}{nlat}_{iax1}"
        axis_rename[f"PC{old_nlat}_2"] = f"{mat_kind}{nlat}_{iax2}"

        # remap celestial axes keywords:
        for kwd in cel_kwd:
            for iold, inew in [(1, nlon), (2, nlat)]:
                if iold != inew:
                    axis_rename[f"{kwd:s}{iold:d}"] = f"{kwd:s}{inew:d}"

        # construct new header cards with remapped axes:
        new_cards = [
            fits.Card(keyword=axis_rename[c.keyword], value=c.value, comment=c.comment)
            if c[0] in axis_rename
            else c
            for c in hdr.cards
        ]

        hdr = fits.Header(new_cards)
        hdr["WCSAXES"] = 2
        hdr.insert("WCSAXES", ("WCSNAME", f"{self.output_frame.name}"), after=True)

        # for PC matrix formalism, set diagonal elements to 0 if necessary
        # (by default, in PC formalism, diagonal matrix elements by default
        # are 0):
        if mat_kind == "PC":
            if nlon not in [iax1, iax2]:
                hdr.insert(
                    f"{mat_kind}{nlon}_{iax2}",
                    (
                        f"{mat_kind}{nlon}_{nlon}",
                        0.0,
                        "Coordinate transformation matrix element",
                    ),
                )
            if nlat not in [iax1, iax2]:
                hdr.insert(
                    f"{mat_kind}{nlat}_{iax2}",
                    (
                        f"{mat_kind}{nlat}_{nlat}",
                        0.0,
                        "Coordinate transformation matrix element",
                    ),
                )

        return hdr

    def _separable_groups(self, detect_celestial):
        """
        This method finds sets (groups) of separable axes - axes that are
        dependent on other axes within a set/group but do not depend on
        axes from other groups. In other words, axes from different
        groups are separable.

        Parameters
        ----------
        detect_celestial : bool
            If `True`, will return, as the third return value, the group of
            celestial axes separately from all other (groups of) axes. If
            no celestial frame is detected, then return value for the
            celestial axes group will be set to `None`.

        Returns
        -------
        axes_groups : list of lists of ``_WorldAxisInfo``
            Each inner list represents a group of non-separable (among
            themselves) axes and each axis in a group is independent of axes
            in *other* groups. Each axis in a group is represented through
            the `_WorldAxisInfo` class used to store relevant information about
            an axis. When ``detect_celestial`` is set to `True`, celestial axes
            group is not included in this list.

        world_axes : list of ``_WorldAxisInfo``
            A flattened version of ``axes_groups``. Even though it is not
            difficult to flatten ``axes_groups``, this list is a by-product
            of other checks and returned here for efficiency. When
            ``detect_celestial`` is set to `True`, celestial axes
            group is not included in this list.

        celestial_group : list of ``_WorldAxisInfo``
            A group of two celestial axes. This group is returned *only when*
            ``detect_celestial`` is set to `True`.

        """

        def find_frame(axis_number):
            for frame in frames:
                if axis_number in frame.axes_order:
                    return frame
            msg = (
                "Encountered an output axes that does not "
                "belong to any output coordinate frames."
            )
            raise RuntimeError(msg)

        # use correlation matrix to find separable axes:
        corr_mat = self.axis_correlation_matrix
        axes_sets = [set(np.flatnonzero(r)) for r in corr_mat.T]

        k = 0
        while len(axes_sets) - 1 > k:
            for m in range(len(axes_sets) - 1, k, -1):
                if axes_sets[k].isdisjoint(axes_sets[m]):
                    continue
                axes_sets[k] = axes_sets[k].union(axes_sets[m])
                del axes_sets[m]
            k += 1

        # create a mapping of output axes to input/image axes groups:
        mapping = {k: tuple(np.flatnonzero(r)) for k, r in enumerate(corr_mat)}

        axes_groups = []
        world_axes = []  # flattened version of axes_groups
        input_axes = []  # all input axes

        if isinstance(self.output_frame, CompositeFrame):
            frames = self.output_frame.frames
        else:
            frames = [self.output_frame]

        celestial_group = None

        # identify which separable group of axes belong
        for s in axes_sets:
            axis_info_group = []  # group of separable output axes info

            # Find the frame to which the first axis in the group belongs.
            # Most likely this frame will be the frame of all other axes in
            # this group; if not, we will update it later.
            axis = sorted(s)
            frame = find_frame(axis[0])

            celestial = (
                detect_celestial
                and len(axis) == 2
                and len(frame.axes_order) == 2
                and isinstance(frame, CelestialFrame)
            )

            for axno in axis:
                if axno not in frame.axes_order:
                    frame = find_frame(axno)
                    celestial = False  # Celestial axes must belong to the same frame

                # index of the axis in this frame's
                fidx = frame.axes_order.index(axno)

                axis_info = _WorldAxisInfo(
                    axis=axno,
                    frame=frame,
                    world_axis_order=self.output_frame.axes_order.index(axno),
                    cunit=frame.unit[fidx].to_string("fits", fraction=True).upper(),
                    ctype=get_ctype_from_ucd(self.world_axis_physical_types[axno]),
                    input_axes=mapping[axno],
                )
                axis_info_group.append(axis_info)
                input_axes.extend(mapping[axno])

            world_axes.extend(axis_info_group)
            if celestial:
                celestial_group = axis_info_group
            else:
                axes_groups.append(axis_info_group)

        # sanity check:
        input_axes = set(
            sum(
                (ax.input_axes for ax in world_axes),
                world_axes[0].input_axes.__class__(),
            )
        )
        n_inputs = len(input_axes)

        if (
            n_inputs != self.pixel_n_dim
            or max(input_axes) + 1 != n_inputs
            or min(input_axes) < 0
        ):
            msg = "Input axes indices are inconsistent with the forward transformation."
            raise ValueError(msg)

        if detect_celestial:
            return axes_groups, world_axes, celestial_group
        return axes_groups, world_axes

    def to_fits_tab(
        self,
        bounding_box=None,
        bin_ext_name="WCS-TABLE",
        coord_col_name="coordinates",
        sampling=1,
    ):
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ .

        Parameters
        ----------
        bounding_box : tuple, optional
            Specifies the range of acceptable values for each input axis.
            The order of the axes is
            `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
            For two image axes ``bounding_box`` is of the form
            ``((xmin, xmax), (ymin, ymax))``.

        bin_ext_name : str, optional
            Extension name for the `~astropy.io.fits.BinTableHDU` HDU for those
            axes groups that will be converted using FITW WCS' ``-TAB``
            algorithm. Extension version will be determined automatically
            based on the number of separable group of axes.

        coord_col_name : str, optional
            Field name of the coordinate array in the structured array
            stored in `~astropy.io.fits.BinTableHDU` data. This corresponds to
            ``TTYPEi`` field in the FITS header of the binary table extension.

        sampling : float, tuple, optional
            The target "density" of grid nodes per pixel to be used when
            creating the coordinate array for the ``-TAB`` FITS WCS convention.
            It is equal to ``1/step`` where ``step`` is the distance between
            grid nodes in pixels. ``sampling`` can be specified as a single
            number to be used for all axes or as a `tuple` of numbers
            that specify the sampling for each image axis.

        Returns
        -------
        hdr : `~astropy.io.fits.Header`
            Header with WCS-TAB information associated (to be used) with image
            data.

        bin_table_hdu : `~astropy.io.fits.BinTableHDU`
            Binary table extension containing the coordinate array.

        Raises
        ------
        ValueError
            When ``bounding_box`` is not defined either through the input
            ``bounding_box`` parameter or this object's ``bounding_box``
            property.

        ValueError
            When ``sampling`` is a `tuple` of length larger than 1 that
            does not match the number of image axes.

        RuntimeError
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

        """
        if bounding_box is None:
            if self.bounding_box is None:
                msg = "Need a valid bounding_box to compute the footprint."
                raise ValueError(msg)
            bounding_box = self.bounding_box

        else:
            # validate user-supplied bounding box:
            frames = self.available_frames
            transform_0 = self.get_transform(frames[0], frames[1])
            Bbox.validate(transform_0, bounding_box)

        if self.forward_transform.n_inputs == 1:
            bounding_box = [bounding_box]

        if self.pixel_n_dim > self.world_n_dim:
            msg = (
                "The case when the number of input axes is larger than the "
                "number of output axes is not supported."
            )
            raise RuntimeError(msg)

        try:
            sampling = np.broadcast_to(sampling, (self.pixel_n_dim,))
        except ValueError as err:
            msg = (
                "Number of sampling values either must be 1 "
                "or it must match the number of pixel axes."
            )
            raise ValueError(msg) from err

        _, world_axes = self._separable_groups(detect_celestial=False)

        hdr, bin_table_hdu = self._to_fits_tab(
            hdr=None,
            world_axes_group=world_axes,
            use_cd=False,
            bounding_box=bounding_box,
            bin_ext=bin_ext_name,
            coord_col_name=coord_col_name,
            sampling=sampling,
        )

        return hdr, bin_table_hdu

    def to_fits(
        self,
        bounding_box=None,
        max_pix_error=0.25,
        degree=None,
        max_inv_pix_error=0.25,
        inv_degree=None,
        npoints=32,
        crpix=None,
        projection="TAN",
        bin_ext_name="WCS-TABLE",
        coord_col_name="coordinates",
        sampling=1,
        verbose=False,
    ):
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ . If WCS contains
        celestial frame, PC/CD formalism will be used for the celestial axes.

        .. note::
            SIP distortion fitting requires that the WCS object has only two
            celestial axes. When WCS does not contain celestial axes,
            SIP fitting parameters (``max_pix_error``, ``degree``,
            ``max_inv_pix_error``, ``inv_degree``, and ``projection``)
            are ignored. When a WCS, in addition to celestial
            frame, contains other types of axes, SIP distortion fitting is
            disabled (only linear terms are fitted for celestial frame).

        Parameters
        ----------
        bounding_box : tuple, optional
            Specifies the range of acceptable values for each input axis.
            The order of the axes is
            `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
            For two image axes ``bounding_box`` is of the form
            ``((xmin, xmax), (ymin, ymax))``.

        max_pix_error : float, optional
            Maximum allowed error over the domain of the pixel array. This
            error is the equivalent pixel error that corresponds to the maximum
            error in the output coordinate resulting from the fit based on
            a nominal plate scale.

        degree : int, iterable, None, optional
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_pixel_error`` is ignored.

            .. note::
                When WCS object has When ``degree`` is `None` and the WCS object has

        max_inv_pix_error : float, optional
            Maximum allowed inverse error over the domain of the pixel array
            in pixel units. If None, no inverse is generated.

        inv_degree : int, iterable, None, optional
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_inv_pixel_error`` is ignored.

        npoints : int, optional
            The number of points in each dimension to sample the bounding box
            for use in the SIP fit. Minimum number of points is 3.

        crpix : list of float, None, optional
            Coordinates (1-based) of the reference point for the new FITS WCS.
            When not provided, i.e., when set to `None` (default) the reference
            pixel will be chosen near the center of the bounding box for axes
            corresponding to the celestial frame.

        projection : str, `~astropy.modeling.projections.Pix2SkyProjection`, optional
            Projection to be used for the created FITS WCS. It can be specified
            as a string of three characters specifying a FITS projection code
            from Table 13 in
            `Representations of World Coordinates in FITS \
            <https://doi.org/10.1051/0004-6361:20021326>`_
            (Paper I), Greisen, E. W., and Calabretta, M. R., A & A, 395,
            1061-1075, 2002. Alternatively, it can be an instance of one of the
            `astropy's Pix2Sky_* <https://docs.astropy.org/en/stable/modeling/\
            reference_api.html#module-astropy.modeling.projections>`_
            projection models inherited from
            :py:class:`~astropy.modeling.projections.Pix2SkyProjection`.

        bin_ext_name : str, optional
            Extension name for the `~astropy.io.fits.BinTableHDU` HDU for those
            axes groups that will be converted using FITW WCS' ``-TAB``
            algorithm. Extension version will be determined automatically
            based on the number of separable group of axes.

        coord_col_name : str, optional
            Field name of the coordinate array in the structured array
            stored in `~astropy.io.fits.BinTableHDU` data. This corresponds to
            ``TTYPEi`` field in the FITS header of the binary table extension.

        sampling : float, tuple, optional
            The target "density" of grid nodes per pixel to be used when
            creating the coordinate array for the ``-TAB`` FITS WCS convention.
            It is equal to ``1/step`` where ``step`` is the distance between
            grid nodes in pixels. ``sampling`` can be specified as a single
            number to be used for all axes or as a `tuple` of numbers
            that specify the sampling for each image axis.

        verbose : bool, optional
            Print progress of fits.

        Returns
        -------
        hdr : `~astropy.io.fits.Header`
            Header with WCS-TAB information associated (to be used) with image
            data.

        hdulist : a list of `~astropy.io.fits.BinTableHDU`
            A Python list of binary table extensions containing the coordinate
            array for TAB extensions; one extension per separable axes group.

        Raises
        ------
        ValueError
            When ``bounding_box`` is not defined either through the input
            ``bounding_box`` parameter or this object's ``bounding_box``
            property.

        ValueError
            When ``sampling`` is a `tuple` of length larger than 1 that
            does not match the number of image axes.

        RuntimeError
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

        """
        if bounding_box is None:
            if self.bounding_box is None:
                msg = "Need a valid bounding_box to compute the footprint."
                raise ValueError(msg)
            bounding_box = self.bounding_box

        else:
            # validate user-supplied bounding box:
            frames = self.available_frames
            transform_0 = self.get_transform(frames[0], frames[1])
            Bbox.validate(transform_0, bounding_box)

        if self.forward_transform.n_inputs == 1:
            bounding_box = [bounding_box]

        if self.pixel_n_dim > self.world_n_dim:
            msg = (
                "The case when the number of input axes is larger than the "
                "number of output axes is not supported."
            )
            raise RuntimeError(msg)

        try:
            sampling = np.broadcast_to(sampling, (self.pixel_n_dim,))
        except ValueError as err:
            msg = (
                "Number of sampling values either must be 1 "
                "or it must match the number of pixel axes."
            )
            raise ValueError(msg) from err

        world_axes_groups, _, celestial_group = self._separable_groups(
            detect_celestial=True
        )

        # Find celestial axes group and treat it separately from other axes:
        if celestial_group:
            # if world_axes_groups is empty, then we have only celestial axes
            # and so we can allow arbitrary degree for SIP. When there are
            # other axes types present, issue a warning and set 'degree' to 1
            # because use of SIP when world_n_dim > 2 currently is not supported by
            # astropy.wcs.WCS - see https://github.com/astropy/astropy/pull/11452
            if world_axes_groups and (degree is None or np.max(degree) != 2):
                if degree is not None:
                    warnings.warn(
                        "SIP distortion is not supported when the number\n"
                        "of axes in WCS is larger than 2. Setting 'degree'\n"
                        "to 1 and 'max_inv_pix_error' to None.",
                        stacklevel=2,
                    )
                degree = 1
                max_inv_pix_error = None

            hdr = self._to_fits_sip(
                celestial_group=celestial_group,
                keep_axis_position=True,
                bounding_box=bounding_box,
                max_pix_error=max_pix_error,
                degree=degree,
                max_inv_pix_error=max_inv_pix_error,
                inv_degree=inv_degree,
                npoints=npoints,
                crpix=crpix,
                projection=projection,
                matrix_type="PC-CDELT1",
                verbose=verbose,
            )
            use_cd = "A_ORDER" in hdr

        else:
            use_cd = False
            hdr = fits.Header()
            hdr["NAXIS"] = 0
            hdr["WCSAXES"] = 0

        # now handle non-celestial axes using -TAB convention for each
        # separable axes group:
        hdulist = []
        for extver0, world_axes_group in enumerate(world_axes_groups):
            # For each subset of separable axes call _to_fits_tab to
            # convert that group to a single Bin TableHDU with a
            # coordinate array for this group of axes:
            hdr, bin_table_hdu = self._to_fits_tab(
                hdr=hdr,
                world_axes_group=world_axes_group,
                use_cd=use_cd,
                bounding_box=bounding_box,
                bin_ext=(bin_ext_name, extver0 + 1),
                coord_col_name=coord_col_name,
                sampling=sampling,
            )
            hdulist.append(bin_table_hdu)

        hdr.add_comment("FITS WCS created by approximating a gWCS")

        return hdr, hdulist

    def _to_fits_tab(
        self,
        hdr,
        world_axes_group,
        use_cd,
        bounding_box,
        bin_ext,
        coord_col_name,
        sampling,
    ):
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ .

        Below we describe only parameters additional to the ones explained for
        `to_fits_tab`.

        .. warn::
            For this helper function, parameters ``bounding_box`` and
            ``sampling`` (when provided as a tuple) are expected to have
            the same length as the number of input axes in the *full* WCS
            object. That is, the number of elements in ``bounding_box`` and
            ``sampling`` is not be affected by ``ignore_axes``.

        Other Parameters
        ----------------
        hdr : astropy.io.fits.Header, None
            The first time this function is called, ``hdr`` should be set to
            `None` or be an empty :py:class:`~astropy.io.fits.Header` object.
            On subsequent calls, updated header from the previous iteration
            should be provided.

        world_axes_group : tuple of dict
            A list of world axes to represent through FITS' -TAB convention.
            This is a list of dictionaries with each dicti

        axes_mapping : dict
            A dictionary that maps output axis index to a tuple of input
            axis indices. In a typical scenario of two input image axes
            and two output celestial axes for a FITS-like WCS,
            this dictionary would look like ``{0: (0, 1), 1: (0, 1)}``
            with the two non-separable input axes.

        fix_axes : dict
            A dictionary containing as keys image axes' indices to be
            fixed and as values - the values to which inputs should be kept
            fixed. For example, this dictionary may be used to indicate the
            celestial axes that should not be included into -TAB approximation
            because they will be approximated using -SIP.

        use_cd : bool
            When `True` - CD-matrix formalism will be used instead of the
            PC-matrix formalism.

        bin_ext : str, tuple of str and int
            Extension name  and optionally version for the
            `~astropy.io.fits.BinTableHDU` HDU. When only a string extension
            name is provided, extension version will be set to 1.
            When ``bin_ext`` is a tuple, first element should be extension
            name and the second element is a positive integer extension version
            number.

        Returns
        -------
        hdr : `~astropy.io.fits.Header`
            Header with WCS-TAB information associated (to be used) with image
            data.

        bin_table_hdu : `~astropy.io.fits.BinTableHDU`
            Binary table extension containing the coordinate array.

        Raises
        ------
        ValueError
            When ``bounding_box`` is not defined either through the input
            ``bounding_box`` parameter or this object's ``bounding_box``
            property.

        ValueError
            When ``sampling`` is a `tuple` of length larger than 1 that
            does not match the number of image axes.

        ValueError
            When extension version is smaller than 1.

        TypeError

        RuntimeError
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

        """
        if isinstance(bin_ext, str):
            bin_ext = (bin_ext, 1)

        if isinstance(bounding_box, Bbox):
            bounding_box = bounding_box.bounding_box(order="F")
        if isinstance(bounding_box, list):
            for index, bbox in enumerate(bounding_box):
                if isinstance(bbox, Bbox):
                    bounding_box[index] = bbox.bounding_box(order="F")

        # identify input axes:
        input_axes = []
        world_axes_idx = []
        for ax in world_axes_group:
            world_axes_idx.append(ax.axis)
            input_axes.extend(ax.input_axes)
        input_axes = sorted(set(input_axes))
        n_inputs = len(input_axes)
        n_outputs = len(world_axes_group)
        world_axes_idx.sort()

        # Create initial header and deal with non-degenerate axes
        if hdr is None:
            hdr = fits.Header()
            hdr["NAXIS"] = n_inputs, "number of array dimensions"
            hdr["WCSAXES"] = n_outputs
            hdr.insert("WCSAXES", ("WCSNAME", f"{self.output_frame.name}"), after=True)

        else:
            hdr["NAXIS"] += n_inputs
            hdr["WCSAXES"] += n_outputs

        # see what axes have been already populated in the header:
        used_hdr_axes = []
        for v in hdr["naxis*"]:
            value = v.split("NAXIS")[1]
            if not value:
                continue

            used_hdr_axes.append(int(value) - 1)

        degenerate_axis_start = max(
            self.pixel_n_dim + 1, max(used_hdr_axes) + 1 if used_hdr_axes else 1
        )

        # Deal with non-degenerate axes and add NAXISi to the header:
        offset = hdr.index("NAXIS")

        for iax in input_axes:
            iiax = int(np.searchsorted(used_hdr_axes, iax))
            hdr.insert(
                iiax + offset + 1,
                (f"NAXIS{iax + 1:d}", int(max(bounding_box[iiax])) + 1),
            )

        # 1D grid coordinates:
        gcrds = []
        cdelt = []
        bb = [bounding_box[k] for k in input_axes]
        for (xmin, xmax), s in zip(bb, sampling, strict=False):
            npix = max(2, 1 + int(np.ceil(abs((xmax - xmin) / s))))
            gcrds.append(np.linspace(xmin, xmax, npix))
            cdelt.append((npix - 1) / (xmax - xmin) if xmin != xmax else 1)

        # In the forward transformation, select only inputs and outputs
        # that we need given world_axes_group parameter:
        bb_center = np.mean(bounding_box, axis=1)

        fixi_dict = {
            k: bb_center[k] for k in set(range(self.pixel_n_dim)).difference(input_axes)
        }

        transform = fix_transform_inputs(self.forward_transform, fixi_dict)
        transform = transform | Mapping(
            world_axes_idx, n_inputs=self.forward_transform.n_outputs
        )

        xyz = np.meshgrid(*gcrds[::-1], indexing="ij")[::-1]

        shape = xyz[0].shape
        xyz = [v.ravel() for v in xyz]

        coord = np.stack(transform(*xyz), axis=-1)

        coord = coord.reshape(
            (
                *shape,
                len(world_axes_group),
            )
        )

        # create header with WCS info:
        if hdr is None:
            hdr = fits.Header()

        for axis_info in world_axes_group:
            k = axis_info.axis
            widx = world_axes_idx.index(k)
            k1 = k + 1
            ct = get_ctype_from_ucd(self.world_axis_physical_types[k])
            if len(ct) > 4:
                msg = "Axis type name too long."
                raise ValueError(msg)

            hdr[f"CTYPE{k1:d}"] = ct + (4 - len(ct)) * "-" + "-TAB"
            hdr[f"CUNIT{k1:d}"] = self.world_axis_units[k]
            hdr[f"PS{k1:d}_0"] = bin_ext[0]
            hdr[f"PV{k1:d}_1"] = bin_ext[1]
            hdr[f"PS{k1:d}_1"] = coord_col_name
            hdr[f"PV{k1:d}_3"] = widx + 1
            hdr[f"CRVAL{k1:d}"] = 1

            if widx < n_inputs:
                m1 = input_axes[widx] + 1
                hdr[f"CRPIX{m1:d}"] = gcrds[widx][0] + 1
                if use_cd:
                    hdr[f"CD{k1:d}_{m1:d}"] = cdelt[widx]
                else:
                    if k1 != m1:
                        hdr[f"PC{k1:d}_{k1:d}"] = 0.0
                    hdr[f"PC{k1:d}_{m1:d}"] = 1.0
                    hdr[f"CDELT{k1:d}"] = cdelt[widx]
            else:
                m1 = degenerate_axis_start
                degenerate_axis_start += 1

                hdr[f"CRPIX{m1:d}"] = 1
                if use_cd:
                    hdr[f"CD{k1:d}_{m1:d}"] = 1.0
                else:
                    if k1 != m1:
                        hdr[f"PC{k1:d}_{k1:d}"] = 0.0
                    hdr[f"PC{k1:d}_{m1:d}"] = 1.0
                    hdr[f"CDELT{k1:d}"] = 1

                coord = coord[None, :]

        # structured array (data) for binary table HDU:
        arr = np.array(
            [(coord,)],
            dtype=[
                (coord_col_name, np.float64, coord.shape),
            ],
        )

        # create binary table HDU:
        bin_table_hdu = fits.BinTableHDU(arr, name=bin_ext[0], ver=bin_ext[1])

        return hdr, bin_table_hdu

    def _calc_approx_inv(self, max_inv_pix_error=5, inv_degree=None, npoints=16):
        """
        Compute polynomial fit for the inverse transformation to be used as
        initial approximation/guess for the iterative solution.
        """
        self._approx_inverse = None

        try:
            # try to use analytic inverse if available:
            self._approx_inverse = functools.partial(
                self.backward_transform, with_bounding_box=False
            )
        except (NotImplementedError, KeyError):
            pass
        else:
            return

        if not isinstance(self.output_frame, CelestialFrame):
            # The _calc_approx_inv method only works with celestial frame transforms
            return

        # Determine reference points.
        if self.bounding_box is None:
            # A bounding_box is needed to proceed.
            return

        crpix = np.mean(self.bounding_box, axis=1)

        crval1, crval2 = self.forward_transform(*crpix)

        # Rotate to native system and deproject. Set center of the projection
        # transformation to the middle of the bounding box ("image") in order
        # to minimize projection effects across the entire image,
        # thus the initial shift.
        sky2pix_proj = Sky2Pix_TAN()

        for transform in self.forward_transform:
            if isinstance(transform, projections.Projection):
                sky2pix_proj = transform
                break
        if sky2pix_proj.__name__.startswith("Pix2Sky"):
            sky2pix_proj = sky2pix_proj.inverse
        lon_pole = _compute_lon_pole((crval1, crval2), sky2pix_proj)
        ntransform = (
            (Shift(crpix[0]) & Shift(crpix[1]))
            | self.forward_transform
            | RotateCelestial2Native(crval1, crval2, lon_pole)
            | sky2pix_proj()
        )

        # standard sampling:
        u, v = make_sampling_grid(npoints, self.bounding_box, crpix=crpix)
        undist_x, undist_y = ntransform(u, v)

        # Double sampling to check if sampling is sufficient.
        ud, vd = make_sampling_grid(2 * npoints, self.bounding_box, crpix=crpix)
        undist_xd, undist_yd = ntransform(ud, vd)

        fit_inv_poly_u, fit_inv_poly_v, max_inv_resid = fit_2D_poly(
            None,
            max_inv_pix_error,
            1,
            undist_x,
            undist_y,
            u,
            v,
            undist_xd,
            undist_yd,
            ud,
            vd,
            verbose=True,
        )

        self._approx_inverse = (
            RotateCelestial2Native(crval1, crval2, lon_pole)
            | sky2pix_proj
            | Mapping((0, 1, 0, 1))
            | (fit_inv_poly_u & fit_inv_poly_v)
            | (Shift(crpix[0]) & Shift(crpix[1]))
        )
