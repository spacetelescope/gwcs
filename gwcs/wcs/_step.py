import warnings
from typing import NamedTuple, TypeAlias, Union

from astropy.modeling.core import Model

from gwcs.coordinate_frames import CoordinateFrame, EmptyFrame

__all__ = [
    "IndexedStep",
    "Step",
    "StepTuple",
]


StepTuple: TypeAlias = tuple[CoordinateFrame, Union[Model, None]]  # noqa: UP007
# Define a type alias for Model or None, the union is necessary because the Model class
#    does not support the | operator for type hinting
Mdl: TypeAlias = Union[Model, None]  # noqa: UP007


class Step:
    """
    Represents a ``step`` in the WCS pipeline.

    Parameters
    ----------
    frame : `~gwcs.coordinate_frames.CoordinateFrame`
        A gwcs coordinate frame object.
    transform : `~astropy.modeling.Model` or None
        A transform from this step's frame to next step's frame.
        The transform of the last step should be `None`.
    """

    class StepAxisWarning(DeprecationWarning):
        """
        Warning raised when the number of axes in the step does not match the
        the number of inputs/outputs of the transform.
        """

    def __init__(self, frame: str | CoordinateFrame | None, transform: Mdl = None):
        # Allow for a string to be passed in for the frame but be turned into a
        # frame object
        self.frame = (
            frame if isinstance(frame, CoordinateFrame) else EmptyFrame(name=frame)
        )
        self.transform = transform

    @property
    def frame(self) -> CoordinateFrame:
        return self._frame

    @frame.setter
    def frame(self, val):
        if not isinstance(val, CoordinateFrame | str):
            msg = '"frame" should be an instance of CoordinateFrame or a string.'
            raise TypeError(msg)

        self._frame = val

    @property
    def transform(self) -> Mdl:
        return self._transform

    @transform.setter
    def transform(self, val):
        if val is not None:
            if not isinstance(val, Model):
                msg = (
                    '"transform" should be an instance of astropy.modeling.Model '
                    "or None."
                )
                raise TypeError(msg)
            if (
                not isinstance(self.frame, EmptyFrame)
                and self.frame.naxes != val.n_inputs
            ):
                warnings.warn(
                    f"Number of inputs ({val.n_inputs}) does not match the number "
                    f"of axes ({self.frame.naxes}) in the frame."
                    "This may lead to unexpected behavior.\n"
                    "This will be an error in a future version.",
                    self.StepAxisWarning,
                    stacklevel=2,
                )
        self._transform = val

    @property
    def frame_name(self):
        if isinstance(self.frame, str):
            return self.frame
        return self.frame.name

    def __getitem__(self, ind):
        warnings.warn(
            "Indexing a WCS.pipeline step is deprecated. "
            "Use the `frame` and `transform` attributes instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if ind not in (0, 1):
            msg = "Allowed inices are 0 (frame) and 1 (transform)."
            raise IndexError(msg)
        if ind == 0:
            return self.frame
        return self.transform

    def __str__(self):
        return (
            f"{self.frame_name}\t "
            f"{getattr(self.transform, 'name', 'None') or type(self.transform).__name__}"  # noqa: E501
        )

    def __repr__(self):
        return (
            f"Step(frame={self.frame_name}, "
            f"transform={getattr(self.transform, 'name', 'None') or type(self.transform).__name__})"  # noqa: E501
        )

    def copy(self):
        return Step(self.frame, self.transform)


class IndexedStep(NamedTuple):
    """
    Class to handle a step and its index in the pipeline.
    """

    index: int
    step: Step
