from ._axis import AxisType
from ._core import CoordinateFrame

__all__ = ["DiscreteFrame"]


class DiscreteFrame(CoordinateFrame):
    """
    A discrete coordinate frame.
    """

    def __init__(self, axis_index, discrete_set=None, axes_names=None, name=None):
        super().__init__(
            naxes=1,
            axes_type=(AxisType.DISCRETE,),
            axes_order=(axis_index,),
            name=name,
            axes_names=axes_names,
        )
        self._discrete_set = discrete_set
