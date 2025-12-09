from ._axis import AxisType
from ._core import CoordinateFrame

__all__ = ["DiscreteFrame"]


class DiscreteFrame(CoordinateFrame):
    """
    A discrete coordinate frame.
    """

    def __init__(
        self,
        axis_index: int,
        discrete_set: set[str | int] | None = None,
        name: str | None = None,
        axes_names: list[str] | None = None,
        axis_physical_types: list[str] | None = None,
    ):
        super().__init__(
            naxes=1,
            axes_type=(AxisType.DISCRETE,),
            axes_order=(axis_index,),
            name=name,
            axes_names=[name] if axes_names is None else axes_names,
            axis_physical_types=axis_physical_types,
        )
        self._discrete_set = discrete_set

    @property
    def discrete_set(self) -> set[str | int] | None:
        """
        The discrete set of values for this frame.
        """
        return self._discrete_set
