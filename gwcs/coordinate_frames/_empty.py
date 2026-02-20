import warnings

from ._core import CoordinateFrame

__all__ = ["EmptyFrame"]


class EmptyFrameDeprecationWarning(DeprecationWarning):
    pass


class EmptyFrame(CoordinateFrame):
    """
    Represents a "default" detector frame. This is for use as the default value
    for input frame by the WCS object.
    """

    def __init__(self, name=None):
        self._name = "detector" if name is None else name
        msg = (
            "The use of strings in place of a proper CoordinateFrame has been "
            "deprecated."
        )
        warnings.warn(msg, EmptyFrameDeprecationWarning, stacklevel=2)

    def __repr__(self):
        return f'<{type(self).__name__}(name="{self.name}")>'

    def __str__(self):
        if self._name is not None:
            return self._name
        return type(self).__name__

    @property
    def name(self):
        """A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val):
        """A custom name of this frame."""
        self._name = val

    def _raise_error(self) -> None:
        msg = "EmptyFrame does not have any information"
        raise NotImplementedError(msg)

    @property
    def naxes(self):
        self._raise_error()

    @property
    def unit(self):
        return None

    @property
    def axes_names(self):
        self._raise_error()

    @property
    def axes_order(self):
        self._raise_error()

    @property
    def reference_frame(self):
        self._raise_error()

    @property
    def axes_type(self):
        self._raise_error()

    @property
    def axis_physical_types(self):
        self._raise_error()

    @property
    def world_axis_object_classes(self):
        self._raise_error()

    @property
    def _native_world_axis_object_components(self):
        self._raise_error()

    def to_high_level_coordinates(self, *values):
        self._raise_error()

    def from_high_level_coordinates(self, *high_level_coords):
        self._raise_error()
