import sys
import warnings
from typing import NamedTuple, Self, TypeAlias, Union

from astropy.modeling.core import Model

from gwcs.coordinate_frames import (
    BaseCoordinateFrame,
    CoordinateFrame,
    CoordinateFrameProtocol,
    EmptyFrame,
)

__all__ = [
    "IndexedStep",
    "Mdl",
    "Step",
    "StepTuple",
]


Mdl: TypeAlias = Union[Model, None]  # noqa: UP007
StepTuple: TypeAlias = tuple[CoordinateFrameProtocol, Union[Model, None]]  # noqa: UP007


# Runtime checkable isinstance check evaluates the actual properties of the object
#    in Python 3.11, so EmptyFrame causes an error to be raised if we attempt to
#    check if it is a CoordinateFrameProtocol. In Python 3.12+, the check does not
#    evaluate the properties of the object, so it does not cause an error.
if sys.version_info >= (3, 12):

    def _is_coordinate_frame(frame: str | CoordinateFrameProtocol) -> bool:
        return isinstance(frame, CoordinateFrameProtocol)
else:

    def _is_coordinate_frame(frame: str | CoordinateFrameProtocol) -> bool:
        return isinstance(frame, BaseCoordinateFrame | CoordinateFrame)


class Step:
    """
    Represents a ``step`` in the WCS pipeline.

    Parameters
    ----------
    frame : `~gwcs.coordinate_frames.CoordinateFrameProtocol` or str
        A gwcs coordinate frame object.
    transform : `~astropy.modeling.Model` or None
        A transform from this step's frame to next step's frame.
        The transform of the last step should be `None`.
    """

    def __init__(
        self, frame: str | CoordinateFrameProtocol, transform: Mdl = None
    ) -> None:
        # Allow for a string to be passed in for the frame but be turned into a
        # frame object
        # This is correct type-wise, but the Python 3.11 bugfix causes a MyPy error
        self.frame = (
            frame
            if _is_coordinate_frame(frame)
            else EmptyFrame.from_transform(frame, transform)  # type: ignore[assignment, arg-type]
        )
        self.transform = transform

    @property
    def frame(self) -> CoordinateFrameProtocol:
        return self._frame

    @frame.setter
    def frame(self, val: CoordinateFrameProtocol) -> None:
        if not _is_coordinate_frame(val):
            msg = '"frame" should be an instance of CoordinateFrameProtocol.'
            raise TypeError(msg)

        self._frame = val

    @property
    def transform(self) -> Mdl:
        return self._transform

    @transform.setter
    def transform(self, val: Mdl) -> None:
        if val is not None and not isinstance(val, Model):
            msg = '"transform" should be an instance of astropy.modeling.Model.'
            raise TypeError(msg)
        self._transform = val

    @property
    def frame_name(self) -> str:
        return self.frame.name

    @property
    def inverse(self) -> Mdl:
        if self.transform is None:
            return None

        try:
            return self.transform.inverse
        except NotImplementedError:
            return None

    def __str__(self) -> str:
        return (
            f"{self.frame_name}\t "
            f"{getattr(self.transform, 'name', 'None') or type(self.transform).__name__}"  # noqa: E501
        )

    def __repr__(self) -> str:
        return (
            f"Step(frame={self.frame_name}, "
            f"transform={getattr(self.transform, 'name', 'None') or type(self.transform).__name__})"  # noqa: E501
        )

    def copy(self) -> Self:
        return type(self)(self.frame, self.transform)

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


class IndexedStep(NamedTuple):
    """
    Class to handle a step and its index in the pipeline.
    """

    idx: int
    step: Step
