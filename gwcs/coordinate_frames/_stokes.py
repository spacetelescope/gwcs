from astropy import units as u
from astropy.coordinates import StokesCoord

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
        axes_order=(0,),
        axes_names=("stokes",),
        name=None,
        axis_physical_types=None,
    ):
        pht = axis_physical_types or self._default_axis_physical_types()

        super().__init__(
            1,
            ["STOKES"],
            axes_order,
            name=name,
            axes_names=axes_names,
            unit=u.one,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self):
        return ("phys.polarization.stokes",)

    @property
    def world_axis_object_classes(self):
        return {
            "stokes": (
                StokesCoord,
                (),
                {},
            )
        }

    @property
    def _native_world_axis_object_components(self):
        return [("stokes", 0, "value")]
