from astropy import units as u
from astropy.coordinates import (
    GCRS,
    BaseCoordinateFrame,
    Galactic,
    GeocentricTrueEcliptic,
    PrecessedGeocentric,
    SkyCoord,
    builtin_frames,
)

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

__all__ = ["CelestialFrame"]

STANDARD_REFERENCE_FRAMES = [frame.upper() for frame in builtin_frames.__all__]


class CelestialFrame(CoordinateFrame):
    """
    Representation of a Celesital coordinate system.

    This class has a native order of longitude then latitude, meaning
    ``axes_names``, ``unit`` and ``axis_physical_types`` should be lon, lat
    ordered. If your transform is in a different order this should be specified
    with ``axes_order``.

    Parameters
    ----------
    axes_order
        A dimension in the input data that corresponds to this axis.
    reference_frame
        A reference frame.
    unit
        Units on axes.
    axes_names
        Names of the axes in this frame.
    name
        Name of this frame.
    axis_physical_types
        The UCD 1+ physical types for the axes, in frame order (lon, lat).
    """

    def __init__(
        self,
        axes_order: tuple[int, ...] | None = None,
        reference_frame: BaseCoordinateFrame | None = None,
        unit: tuple[u.Unit, ...] | None = None,
        axes_names: tuple[str, ...] | None = None,
        name: str | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        naxes = 2
        if (
            reference_frame is not None
            and not isinstance(reference_frame, str)
            and reference_frame.name.upper() in STANDARD_REFERENCE_FRAMES
        ):
            _axes_names = (
                tuple(
                    n
                    for n in reference_frame.representation_component_names.values()
                    if n != "distance"
                )
                if axes_names is None
                else axes_names
            )
            naxes = len(_axes_names)

        self.native_axes_order = tuple(range(naxes))
        if axes_order is None:
            axes_order = self.native_axes_order
        if unit is None:
            # Astropy dynamically creates some units, so MyPy can't find them
            unit = tuple([u.degree] * naxes)  # type: ignore[attr-defined]
        axes_type = (AxisType.SPATIAL,) * naxes

        super().__init__(
            naxes=naxes,
            axes_type=axes_type,
            axes_order=axes_order,
            reference_frame=reference_frame,
            unit=unit,
            axes_names=_axes_names,
            name=name,
            axis_physical_types=axis_physical_types,
        )

    def _default_axis_physical_types(
        self, properties: FrameProperties
    ) -> AxisPhysicalTypes:
        if isinstance(self.reference_frame, Galactic):
            return "pos.galactic.lon", "pos.galactic.lat"
        if isinstance(
            self.reference_frame,
            GeocentricTrueEcliptic | GCRS | PrecessedGeocentric,
        ):
            return "pos.bodyrc.lon", "pos.bodyrc.lat"
        if isinstance(self.reference_frame, builtin_frames.BaseRADecFrame):
            return "pos.eq.ra", "pos.eq.dec"
        if isinstance(self.reference_frame, builtin_frames.BaseEclipticFrame):
            return "pos.ecliptic.lon", "pos.ecliptic.lat"
        return tuple(f"custom:{t}" for t in properties.axes_names)

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        return {
            "celestial": WorldAxisClass(
                SkyCoord,
                (),
                {"frame": self.reference_frame, "unit": self._prop.unit},
            )
        }

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        return [
            WorldAxisComponent(
                "celestial", 0, lambda sc: sc.spherical.lon.to_value(self._prop.unit[0])
            ),
            WorldAxisComponent(
                "celestial", 1, lambda sc: sc.spherical.lat.to_value(self._prop.unit[1])
            ),
        ]
