from astropy import units as u
from astropy.coordinates import StokesCoord

from gwcs._typing import AxisPhysicalTypes
from gwcs.api import (
    WorldAxisClass,
    WorldAxisClasses,
    WorldAxisComponent,
    WorldAxisComponents,
)

from ._axis import AxisType
from ._core import CoordinateFrame
from ._properties import FrameProperties

__all__ = ["StokesFrame"]


class StokesFrame(CoordinateFrame):
    """
    A coordinate frame for representing Stokes polarisation states.

    Parameters
    ----------
    name
        Name of this frame.
    axes_order
        A dimension in the data that corresponds to this axis.
    """

    def __init__(
        self,
        axes_order: tuple[int, ...] = (0,),
        axes_names: tuple[str, ...] = ("stokes",),
        name: str | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        super().__init__(
            1,
            (AxisType.STOKES,),
            axes_order,
            name=name,
            axes_names=axes_names,
            unit=(u.one,),  # type: ignore[arg-type]
            axis_physical_types=axis_physical_types,
        )

    def _default_axis_physical_types(
        self, properties: FrameProperties
    ) -> AxisPhysicalTypes:
        return ("phys.polarization.stokes",)

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        return {
            "stokes": WorldAxisClass(
                StokesCoord,
                (),
                {},
            )
        }

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        return [WorldAxisComponent("stokes", 0, "value")]
