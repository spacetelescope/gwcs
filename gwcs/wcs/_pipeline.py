import warnings
from functools import partial, reduce
from typing import TypeAlias, Union

import numpy as np
from astropy.modeling import Model
from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox
from astropy.modeling.models import (
    Mapping,
    RotateCelestial2Native,
    Shift,
    Sky2Pix_TAN,
)
from astropy.units import Unit

from gwcs.coordinate_frames import CelestialFrame, CoordinateFrame, EmptyFrame
from gwcs.utils import CoordinateFrameError

from ._exception import GwcsBoundingBoxWarning, GwcsFrameExistsError
from ._fixed_point import vectorized_fixed_point
from ._step import IndexedStep, Step, StepTuple
from ._utils import fit_2D_poly, make_sampling_grid

__all__ = ["ForwardTransform", "Pipeline"]

# Type aliases due to the use of the `|` for type hints not working with Model
ForwardTransform: TypeAlias = Union[Model, list[Step | StepTuple], None]  # noqa: UP007
Mdl: TypeAlias = Union[Model, None]  # noqa: UP007


class Pipeline:
    """
    Class to handle a sequence of WCS transformations.

    This is intended to act line a list of steps, but with built in protections
    for things like duplicate frames. In addition, this handles all the logic
    for handling steps and their frames/transforms.
    """

    def __init__(
        self,
        forward_transform: ForwardTransform = None,
        input_frame: str | CoordinateFrame | None = None,
        output_frame: str | CoordinateFrame | None = None,
    ) -> None:
        self._pipeline: list[Step] = []
        self._initialize_pipeline(forward_transform, input_frame, output_frame)

    def _initialize_pipeline(
        self,
        forward_transform: ForwardTransform,
        input_frame: str | CoordinateFrame | None,
        output_frame: str | CoordinateFrame | None,
    ) -> None:
        """
        Initialize a pipeline from a forward transform specification.

        Parameters
        ----------
        forward_transform " `~astropy.modeling.Model`, list of `~gwcs.wcs.Step`, or None
            The forward transform to initialize the pipeline with.
            - Can be a single model which acts as the entire transform.
            - List of steps for the pipeline
            - List of tuples[CoordinateFrame, Model] for the pipeline
            - None for an empty pipeline
        input_frame : `~gwcs.coordinate_frames.CoordinateFrame` or None
            The input frame of the pipeline.
        output_frame : `~gwcs.coordinate_frames.CoordinateFrame` or None
            The output frame of the pipeline. This must be specified if
            forward_transform is not a list of steps.

        Returns
        -------
        An initialized pipeline.
        """
        self._approx_inverse = None

        if forward_transform is None:
            # Initialize a WCS without a forward_transform - allows building a
            # WCS programmatically.
            if output_frame is None:
                msg = "An output_frame must be specified if forward_transform is None."
                raise CoordinateFrameError(msg)

            forward_transform = [
                Step(input_frame, None),
                Step(output_frame, None),
            ]

        if isinstance(forward_transform, Model):
            if output_frame is None:
                msg = (
                    "An output_frame must be specified if forward_transform is a model."
                )
                raise CoordinateFrameError(msg)

            forward_transform = [
                Step(input_frame, forward_transform.copy()),
                Step(output_frame, None),
            ]

        if not isinstance(forward_transform, list):
            msg = (
                "Expected forward_transform to be a None, model, or a "
                f"(frame, transform) list, got {type(forward_transform)}"
            )
            raise TypeError(msg)

        self._extend(forward_transform)

    @property
    def pipeline(self) -> list[Step]:
        """
        Allow direct access to the raw pipeline steps.
        """

        # TODO: This can still allow direct modification of the pipeline list
        #       without any of the checks and handling that have been put in
        #       place in order to ensure the pipeline is functional.
        #       -> Maybe we should return a copy?
        return self._pipeline

    @property
    def available_frames(self) -> list[str]:
        """
        List of all the frame names in this WCS in their order in the pipeline
        """
        return [step.frame.name for step in self._pipeline]

    def _wrap_step(
        self, step: Step | StepTuple, *, replace_index: int | None = None
    ) -> Step:
        """
        Wrap the step in a Step object if it is not already, and
        check that the frame is not already in the pipeline.

        Parameters
        ----------
        step : `~gwcs.wcs.Step` or tuple
            The step to wrap in a Step object and check.
        replace_index : int or None
            The index of the step to replace in the pipeline, this ensures that
            we can inplace replace a step using the same frame as the one being
            replaced. This frame will be removed from the frames to check against
            If None (default), do not remove any frames for checking.
        """
        # Copy externally created steps to ensure they are not modified outside
        # the control of the pipeline
        value = step.copy() if isinstance(step, Step) else Step(*step)

        frames = self.available_frames

        # If we are replacing a step, remove it from the list of frames as we will
        # not be duplicating it in that case
        if replace_index is not None:
            frames.pop(replace_index)

        if value.frame.name in frames:
            msg = f"Frame {value.frame.name} is already in the pipeline."
            raise GwcsFrameExistsError(msg)

        # Add the frame as an attribute of the pipeline
        super().__setattr__(value.frame.name, value.frame)

        return value

    def _check_last_step(self) -> None:
        """
        Check the last frame in the pipeline has a None transform
        -> The last frame in the pipeline must have a None transform.
        """
        if self._pipeline[-1].transform is not None:
            msg = "The last step in the pipeline must have a None transform."
            raise ValueError(msg)

    def _insert(self, index: int, value: Step | StepTuple) -> None:
        """
        Handle insertion of a step into the pipeline.
        """
        self._pipeline.insert(index, self._wrap_step(value))
        self._check_last_step()

    def _extend(self, values: list[Step]) -> None:
        """
        Handle extending the pipeline with a list of steps
        """
        for value in values:
            self._pipeline.append(self._wrap_step(value))

        self._check_last_step()

    @staticmethod
    def _handle_empty_frame(frame: CoordinateFrame) -> CoordinateFrame | None:
        """
        Handle the case where the frame is an EmptyFrame.
        """
        return None if isinstance(frame, EmptyFrame) else frame

    @property
    def input_frame(self) -> CoordinateFrame | None:
        """
        Return the input frame name of the pipeline.
        """
        return self._handle_empty_frame(
            self._pipeline[0].frame if self._pipeline else None
        )

    @property
    def output_frame(self) -> CoordinateFrame | None:
        """
        Return the output frame name of the pipeline.
        """
        return self._handle_empty_frame(
            self._pipeline[-1].frame if self._pipeline else None
        )

    @property
    def unit(self) -> Unit | None:
        """The unit of the coordinates in the output coordinate system."""
        return self._pipeline[-1].frame.unit if self._pipeline else None

    @staticmethod
    def _combine_transforms(transforms: list[Model]) -> Model:
        """
        Combine a list of transforms into a single transform.
        """
        return reduce(lambda x, y: x | y, transforms)

    @staticmethod
    def _frame_name(frame: str | CoordinateFrame) -> str:
        """
        Return the name of the frame.

        Parameters
        ----------
        frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Name of the frame or the frame object.

        Returns
        -------
        Name of the frame.
        """
        return frame.name if isinstance(frame, CoordinateFrame) else frame

    def _frame_index(self, frame: str | CoordinateFrame) -> int:
        """
        Return the index of the given frame in the pipeline.

        Parameters
        ----------
        frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Name of the frame or the frame object.

        Returns
        -------
        Index of the frame in the pipeline.
        """
        try:
            return self.available_frames.index(self._frame_name(frame))
        except ValueError as err:
            msg = f"Frame {self._frame_name(frame)} is not in the available frames"
            raise CoordinateFrameError(msg) from err

    def _get_step(self, frame: str | CoordinateFrame) -> IndexedStep:
        """
        Get the index and step corresponding to the given frame.
        """
        index = self._frame_index(frame)

        return IndexedStep(index, self._pipeline[index])

    def get_transform(
        self, from_frame: str | CoordinateFrame, to_frame: str | CoordinateFrame
    ) -> Mdl:
        """
        Return a transform between two coordinate frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame name of object.
        to_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            End coordinate frame name or object.

        Returns
        -------
        transform : `~astropy.modeling.Model`
            Transform between two frames.
        """
        from_index = self._frame_index(from_frame)
        to_index = self._frame_index(to_frame)

        # Moving backwards over the pipeline
        if to_index < from_index:
            transforms = [
                step.transform.inverse
                for step in self._pipeline[to_index:from_index][::-1]
            ]

        # from and to are the same
        elif to_index == from_index:
            return None

        # Moving forwards over the pipeline
        else:
            transforms = [
                step.transform for step in self._pipeline[from_index:to_index]
            ]

        return self._combine_transforms(transforms)

    def set_transform(
        self,
        from_frame: str | CoordinateFrame,
        to_frame: str | CoordinateFrame,
        transform: Model,
    ) -> None:
        """
        Set/replace the transform between two coordinate frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.coordinate_frames.CoordinateFrame`
            End coordinate frame.
        transform : `~astropy.modeling.Model`
            Transform between ``from_frame`` and ``to_frame``.
        """
        from_index = self._frame_index(from_frame)
        to_index = self._frame_index(to_frame)

        if from_index + 1 != to_index:
            msg = (
                f"Frames {self._frame_name(from_frame)} and "
                f"{self._frame_name(to_frame)} "
                "are not in sequence"
            )
            raise ValueError(msg)

        self._pipeline[from_index].transform = transform

    def insert_transform(
        self, frame: str | CoordinateFrame, transform: Model, after: bool = False
    ) -> None:
        """
        Insert a transform before (default) or after a coordinate frame.

        Append (or prepend) a transform to the transform connected to frame.

        Parameters
        ----------
        frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame which sets the point of insertion.
        transform : `~astropy.modeling.Model`
            New transform to be inserted in the pipeline
        after : bool
            If True, the new transform is inserted in the pipeline
            immediately after ``frame``.
        """

        index = self._frame_index(frame)
        if not after:
            index -= 1

        current_transform = self._pipeline[index].transform
        transform = (
            transform | current_transform if after else current_transform | transform
        )

        self._pipeline[index].transform = transform

        self._check_last_step()

    def insert_frame(
        self,
        input_frame: str | CoordinateFrame,
        transform: Model,
        output_frame: str | CoordinateFrame,
    ) -> None:
        """
        Insert a new frame into an existing pipeline. This frame must be
        anchored to a frame already in the pipeline by a transform. This
        existing frame is identified solely by its name, although an entire
        `~gwcs.coordinate_frames.CoordinateFrame` can be passed (e.g., the
        `input_frame` or `output_frame` attribute). This frame is never
        modified.

        Parameters
        ----------
        input_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame at start of new transform
        transform : `~astropy.modeling.Model`
            New transform to be inserted in the pipeline
        output_frame: str or `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame at end of new transform
        """

        def get_index(frame: str | CoordinateFrame) -> int | None:
            try:
                index = self._frame_index(frame)
            except CoordinateFrameError as err:
                index = None
                if not isinstance(frame, CoordinateFrame):
                    msg = (
                        f"New coordinate frame {self._frame_name(frame)} "
                        "must be defined"
                    )
                    raise ValueError(msg) from err  # noqa: TRY004

            return index

        input_index = get_index(input_frame)
        output_index = get_index(output_frame)

        new_frames = [input_index, output_index].count(None)

        match new_frames:
            case 0:
                msg = (
                    "Could not insert frame as both frames "
                    f"{self._frame_name(input_frame)} and "
                    f"{self._frame_name(output_frame)} already exist"
                )
                raise ValueError(msg)
            case 2:
                msg = (
                    "Could not insert frame as neither frame "
                    f"{self._frame_name(input_frame)} and "
                    f"{self._frame_name(output_frame)} exists"
                )
                raise ValueError(msg)

        # so input_index is None or output_index is None
        if input_index is None:
            self._insert(output_index, Step(input_frame, transform))
        else:
            current = self._pipeline[input_index].transform
            self._pipeline[input_index].transform = transform
            self._insert(input_index + 1, Step(output_frame, current))

    @property
    def bounding_box(self) -> ModelBoundingBox | CompoundBoundingBox | None:
        """
        Return the bounding box of the pipeline.
        """
        # Pull the first transform of the pipeline which is what controls the
        # bounding_box
        frames = self.available_frames
        transform = self.get_transform(frames[0], frames[1])

        if transform is None:
            return None

        try:
            bounding_box = transform.bounding_box
        except NotImplementedError:
            return None

        if (
            # Check that the bounding_box was set on the instance (not a default)
            transform._user_bounding_box is not None
            # Check the order of that bounding_box is C
            and bounding_box.order == "C"
            # Check that the bounding_box is not a single value
            and (isinstance(bounding_box, CompoundBoundingBox) or len(bounding_box) > 1)
        ):
            warnings.warn(
                "The bounding_box was set in C order on the transform prior to "
                "being used in the gwcs!\n"
                "Check that you intended that ordering for the bounding_box, "
                "and consider setting it in F order.\n"
                "The bounding_box will remain meaning the same but will be "
                "converted to F order for consistency in the GWCS.",
                GwcsBoundingBoxWarning,
                stacklevel=2,
            )
            self.bounding_box = bounding_box.bounding_box(order="F")
            bounding_box = self.bounding_box

        return bounding_box

    @bounding_box.setter
    def bounding_box(
        self, value: tuple | ModelBoundingBox | CompoundBoundingBox | None
    ) -> None:
        """
        Set the range of acceptable values for each input axis.

        The order of the axes is `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
        For two inputs and axes_order(0, 1) the bounding box is
        ((xlow, xhigh), (ylow, yhigh)).

        Parameters
        ----------
        value : tuple or None
            Tuple of tuples with ("low", high") values for the range.
        """
        frames = self.available_frames
        transform_0 = self.get_transform(frames[0], frames[1])
        if value is None:
            transform_0.bounding_box = value
        else:
            # Make sure the dimensions of the new bbox are correct.
            if isinstance(value, CompoundBoundingBox):
                bbox = CompoundBoundingBox.validate(transform_0, value, order="F")
            else:
                bbox = ModelBoundingBox.validate(transform_0, value, order="F")

            transform_0.bounding_box = bbox

        self.set_transform(frames[0], frames[1], transform_0)

    def attach_compound_bounding_box(
        self, cbbox: dict[tuple[str], tuple], selector_args: tuple[str]
    ):
        """
        Attach a compound bounding box dictionary to the pipeline.

        Parameters
        ----------
        cbbox
            Dictionary of the bounding box tuples (F order) for each input set
                keys: selector argument
                values: bounding box tuple in F order
        selector_args:
            Argument names to the model that are used to select the bounding box
        """
        frames = self.available_frames
        transform_0 = self.get_transform(frames[0], frames[1])

        self.bounding_box = CompoundBoundingBox.validate(
            transform_0, cbbox, selector_args=selector_args, order="F"
        )

    @property
    def forward_transform(self) -> Model:
        """
        Return the forward transform of the pipeline.
        """
        transform = self._combine_transforms(
            [step.transform for step in self._pipeline[:-1]]
        )

        if self.bounding_box is not None:
            # Currently compound models do not attempt to combine individual model
            # bounding boxes. Get the forward transform and assign the bounding_box
            # to it before evaluating it. The order Model.bounding_box is reversed.
            transform.bounding_box = self.bounding_box

        return transform

    @property
    def backward_transform(self):
        """
        Return the total backward transform if available - from output to input
        coordinate system.

        Raises
        ------
        NotImplementedError :
            An analytical inverse does not exist.

        """
        try:
            backward = self.forward_transform.inverse
        except NotImplementedError as err:
            msg = f"Could not construct backward transform. \n{err}"
            raise NotImplementedError(msg) from err
        try:
            _ = backward.inverse
        except NotImplementedError:  # means "hasattr" won't work
            backward.inverse = self.forward_transform
        return backward

    @property
    def input_n_dim(self) -> int:
        if self.input_frame is None:
            return self.forward_transform.n_inputs
        return self.input_frame.naxes

    @property
    def output_n_dim(self) -> int:
        if self.output_frame is None:
            return self.forward_transform.n_outputs
        return self.output_frame.naxes

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
        if kwargs.pop("with_units", False):
            msg = (
                "Support for with_units in numerical_inverse has been removed, "
                "use inverse"
            )
            raise ValueError(msg)

        args_shape = np.shape(args)
        nargs = args_shape[0]
        arg_dim = len(args_shape) - 1

        if nargs != self.output_n_dim:
            msg = (
                "Number of input coordinates is different from "
                "the number of defined world coordinates in the "
                f"WCS ({self.output_n_dim:d})"
            )
            raise ValueError(msg)

        if self.output_n_dim != self.input_n_dim:
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
                x0 = np.ones(self.input_n_dim)
            else:
                x0 = np.mean(self.bounding_box, axis=-1)

        if arg_dim == 0:
            argsi = args

            if nargs == 2 and self._approx_inverse is not None:
                x0 = self._approx_inverse(*argsi)
                if not np.all(np.isfinite(x0)):
                    return [np.array(np.nan) for _ in range(nargs)]

            result = tuple(
                vectorized_fixed_point(
                    self.forward_transform,
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

            result = vectorized_fixed_point(
                self.forward_transform,
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

    def _calc_approx_inv(self, max_inv_pix_error=5, inv_degree=None, npoints=16):
        """
        Compute polynomial fit for the inverse transformation to be used as
        initial approximation/guess for the iterative solution.
        """
        self._approx_inverse = None

        try:
            # try to use analytic inverse if available:
            self._approx_inverse = partial(
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
        ntransform = (
            (Shift(crpix[0]) & Shift(crpix[1]))
            | self.forward_transform
            | RotateCelestial2Native(crval1, crval2, 180)
            | Sky2Pix_TAN()
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
            RotateCelestial2Native(crval1, crval2, 180)
            | Sky2Pix_TAN()
            | Mapping((0, 1, 0, 1))
            | (fit_inv_poly_u & fit_inv_poly_v)
            | (Shift(crpix[0]) & Shift(crpix[1]))
        )
