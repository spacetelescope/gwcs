import warnings

from astropy.modeling.core import Model

from gwcs.coordinate_frames import (
    CoordinateFrame,
)

__all__ = ["Step"]


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

    def __init__(self, frame, transform=None):
        self.frame = frame
        self.transform = transform

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, val):
        if not isinstance(val, CoordinateFrame | str):
            msg = '"frame" should be an instance of CoordinateFrame or a string.'
            raise TypeError(msg)

        self._frame = val

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, val):
        if val is not None and not isinstance(val, (Model)):
            msg = '"transform" should be an instance of astropy.modeling.Model.'
            raise TypeError(msg)
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
