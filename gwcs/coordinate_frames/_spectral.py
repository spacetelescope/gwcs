from astropy import units as u
from astropy.coordinates import BaseCoordinateFrame, SpectralCoord

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

__all__ = ["SpectralFrame"]


class SpectralFrame(CoordinateFrame):
    """
    Represents Spectral Frame

    Parameters
    ----------
    axes_order
        A dimension in the input data that corresponds to this axis.
    reference_frame : astropy.coordinates.builtin_frames
        Reference frame (usually used with output_frame to convert to world
        coordinate objects).
    unit
        Spectral unit.
    axes_names
        Spectral axis name.
    name
        Name for this frame.

    """

    def __init__(
        self,
        axes_order: tuple[int, ...] = (0,),
        reference_frame: BaseCoordinateFrame | None = None,
        unit: tuple[u.Unit, ...] | None = None,
        axes_names: tuple[str, ...] | None = None,
        name: str | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        super().__init__(
            naxes=1,
            axes_type=AxisType.SPECTRAL,
            axes_order=axes_order,
            axes_names=axes_names,
            reference_frame=reference_frame,
            unit=unit,
            name=name,
            axis_physical_types=axis_physical_types,
        )

    def _default_axis_physical_types(
        self, properties: FrameProperties
    ) -> AxisPhysicalTypes:
        if properties.unit[0].physical_type == "frequency":
            return ("em.freq",)
        if properties.unit[0].physical_type == "length":
            return ("em.wl",)
        if properties.unit[0].physical_type == "energy":
            return ("em.energy",)
        if properties.unit[0].physical_type == "speed":
            return ("spect.dopplerVeloc",)
        return (f"custom:{properties.unit[0].physical_type}",)

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        return {"spectral": WorldAxisClass(SpectralCoord, (), {"unit": self.unit[0]})}

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        return [WorldAxisComponent("spectral", 0, lambda sc: sc.to_value(self.unit[0]))]
