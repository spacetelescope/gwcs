from dataclasses import InitVar, dataclass

import numpy as np
from astropy import units as u
from astropy.wcs.wcsapi.low_level_api import VALID_UCDS, validate_physical_types

__all__ = ["FrameProperties"]


@dataclass
class FrameProperties:
    naxes: InitVar[int]
    axes_type: tuple[str]
    unit: tuple[u.Unit] = None
    axes_names: tuple[str] = None
    axis_physical_types: list[str] = None

    def __post_init__(self, naxes):
        if isinstance(self.axes_type, str):
            self.axes_type = (self.axes_type,)
        else:
            self.axes_type = tuple(self.axes_type)

        if len(self.axes_type) != naxes:
            msg = "Length of axes_type does not match number of axes."
            raise ValueError(msg)

        if self.unit is not None:
            unit = tuple(self.unit) if np.iterable(self.unit) else (self.unit,)
            if len(unit) != naxes:
                msg = "Number of units does not match number of axes."
                raise ValueError(msg)
            self.unit = tuple(u.Unit(au) for au in unit)
        else:
            self.unit = tuple(u.dimensionless_unscaled for na in range(naxes))

        if self.axes_names is not None:
            if isinstance(self.axes_names, str):
                self.axes_names = (self.axes_names,)
            else:
                self.axes_names = tuple(self.axes_names)
            if len(self.axes_names) != naxes:
                msg = "Number of axes names does not match number of axes."
                raise ValueError(msg)
        else:
            self.axes_names = tuple([""] * naxes)

        if self.axis_physical_types is not None:
            if isinstance(self.axis_physical_types, str):
                self.axis_physical_types = (self.axis_physical_types,)
            elif not np.iterable(self.axis_physical_types):
                msg = (
                    "axis_physical_types must be of type string or iterable of strings"
                )
                raise TypeError(msg)
            if len(self.axis_physical_types) != naxes:
                msg = f'"axis_physical_types" must be of length {naxes}'
                raise ValueError(msg)
            ph_type = []
            for axt in self.axis_physical_types:
                if axt not in VALID_UCDS and not axt.startswith("custom:"):
                    ph_type.append(f"custom:{axt}")
                else:
                    ph_type.append(axt)

            validate_physical_types(ph_type)
            self.axis_physical_types = tuple(ph_type)

    @property
    def _default_axis_physical_types(self):
        """
        The default physical types to use for this frame if none are specified
        by the user.
        """
        return tuple(f"custom:{t}" for t in self.axes_type)
