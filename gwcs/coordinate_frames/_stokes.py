from astropy import units as u
from astropy.coordinates import StokesCoord

from ._axis import AxisType
from ._base import WorldAxisObjectClass, WorldAxisObjectComponent
from ._core import CoordinateFrame

__all__ = ["StokesFrame"]


class StokesFrame(CoordinateFrame):
    """
    A coordinate frame for representing Stokes polarisation states.

    Parameters
    ----------
    name : str
        Name of this frame.
    axes_order : tuple
        A dimension in the data that corresponds to this axis.
    """

    def __init__(
        self,
        axes_order: tuple[int] = (0,),
        axes_names: tuple[str] = ("stokes",),
        name: str | None = None,
        axis_physical_types: tuple[str | None] | None = None,
    ) -> None:
        super().__init__(
            naxes=1,
            axes_type=(AxisType.STOKES,),
            axes_order=axes_order,
            name=name,
            axes_names=axes_names,
            unit=(u.one,),
            axis_physical_types=axis_physical_types,
        )

    def _default_axis_physical_types(
        self, axes_type: tuple[AxisType | str, ...]
    ) -> tuple[str, ...]:
        return ("phys.polarization.stokes",)

    @property
    def world_axis_object_classes(self) -> dict[str, WorldAxisObjectClass]:
        return {
            "stokes": WorldAxisObjectClass(
                StokesCoord,
                (),
                {},
            )
        }

    @property
    def _native_world_axis_object_components(self) -> list[WorldAxisObjectComponent]:
        return [WorldAxisObjectComponent("stokes", 0, "value")]
