# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models for generating FITS WCS standard transforms.
"""

import numbers

import numpy as np
from astropy import units as u
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter
from astropy.modeling.models import Shift, Scale, AffineTransformation2D, RotateNative2Celestial

from .utils import _compute_lon_pole


__all__ = [
    "FITSImagingWCSTransform",
]


class FITSImagingWCSTransform(Model):
    """
    FITS WCS standard imaging transform.

    Parameters
    ----------
    projection : `~astropy.modeling.projections.Projection`
      Celestial projection.
    crpix : tuple or iterable of size 2
      Pixel coordinate of the reference point. Default is [0, 0]
    crval : tuple or iterable of size 2
      Celestial longitude and latitude of the fiducial point in deg. Default is [0, 0]
    cdelt : tuple or iterable of size 2
      Coordinate scale factors. Default is [1, 1]
    pc : ndarray of shape (2,2)
      Linear transformation matrix. Default is [[1,0][0, 1]]


    Returns
    -------
    transform : `~astropy.modeling.core.CompoundModel`
      The transform between a detector and a celestial coordinate frames.

    Examples
    --------
    >>> from gwcs import FITSImagingWCSTransform
    >>> from astropy.modeling.models import Pix2Sky_Gnomonic
    >>> tan = Pix2Sky_Gnomonic()
    >>> fwcs = FITSImagingWCSTransform(tan, crpix=[10,10], crval=[5.36, -72.5])
    >>> print(fwcs)
    Model: FITSImagingWCSTransform
    Inputs: ('x', 'y')
    Outputs: ('lon', 'lat')
    Model set size: 1
    Parameters:
       crpix         crval       cdelt        pc
    ------------ ------------- ---------- ----------
    10.0 .. 10.0 5.36 .. -72.5 1.0 .. 1.0 1.0 .. 1.0

    """

    _separable = False

    standard_broadcasting = False
    fittable = False

    n_inputs = 2
    n_outputs = 2

    crpix = Parameter(default=[0.0, 0.0], description="crpix")
    crval = Parameter(default=[0.0, 0.0], description="crval")
    cdelt = Parameter(default=[1.0, 1.0], description="cdelt")
    pc = Parameter(default=[[1.0, 0.0], [0.0, 1.0]], description="pc")


    def __init__(self, projection, crpix=crpix,
                 crval=crval, cdelt=cdelt, pc=pc, **kwargs):
        super().__init__(crpix=crpix,
                         crval=crval, cdelt=cdelt, pc=pc, **kwargs)
        self.projection = projection
        self.inputs = ("x", "y")
        self.outputs = ("lon", "lat")
        self.shift = Shift(-crpix[0]) & Shift(-crpix[1])
        self.scale = Scale(cdelt[0]) & Scale(cdelt[1])
        self.rotation = AffineTransformation2D(matrix=pc)
        lon_pole = _compute_lon_pole(self.crval, projection)
        self.sky_rot = RotateNative2Celestial(crval[0], crval[1], lon_pole)
        self.forward = self.shift | self.rotation | self.scale | self.projection | self.sky_rot

    def evaluate(self, x, y, crpix, crval, cdelt, pc):
        return self.forward(x, y)
