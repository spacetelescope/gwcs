import warnings
from functools import reduce
from typing import Any, TypeAlias, Union

from astropy.modeling import Model
from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox
from astropy.units import Unit

from gwcs.coordinate_frames import CoordinateFrame, EmptyFrame
from gwcs.utils import CoordinateFrameError

from ._exception import GwcsBoundingBoxWarning, GwcsFrameExistsError
from ._step import Step, StepTuple

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
                    "An output_frame must be specified "
                    "if forward_transform is a model."
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

    def __getattr__(self, name: str) -> Any:
        """
        Modify the behavior of the getattr method to allow accessing frames
        via their name.
        -> Enable dot access to frames in the pipeline based on their name
        """

        if name in self.available_frames:
            return self._pipeline[self._frame_index(name)].frame

        msg = f"Pipeline has no attribute {name}"
        raise AttributeError(msg)

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

        return value

    def _check_last_step(self) -> None:
        """
        Check the last frame in the pipeline has a None transform
        -> The last frame in the pipeline must have a None transform.
        """
        if self._pipeline[-1].transform is not None:
            msg = "The last frame in the pipeline must have a None transform."
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
