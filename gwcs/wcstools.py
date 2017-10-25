# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import functools
import numpy as np
from astropy.modeling.core import Model
from astropy.modeling import projections
from astropy.modeling import models
from astropy import coordinates as coord

from .wcs import WCS
from .coordinate_frames import *
from .utils import UnsupportedTransformError, UnsupportedProjectionError
from .utils import _compute_lon_pole, _get_slice, _toindex, axis_domain_to_slice

import warnings
from astropy.utils.decorators import deprecated
from .utils import _domain_to_bounding_box


__all__ = ['wcs_from_fiducial', 'grid_from_domain', 'grid_from_bounding_box']


def wcs_from_fiducial(fiducial, coordinate_frame=None, projection=None,
                      transform=None, name='', bounding_box=None, domain=None):
    """
    Create a WCS object from a fiducial point in a coordinate frame.

    If an additional transform is supplied it is prepended to the projection.

    Parameters
    ----------
    fiducial : `~astropy.coordinates.SkyCoord` or tuple of float
        One of:
            A location on the sky in some standard coordinate system.
            A Quantity with spectral units.
            A list of the above.
    coordinate_frame : ~gwcs.coordinate_frames.CoordinateFrame`
        The output coordinate frame.
        If fiducial is not an instance of `~astropy.coordinates.SkyCoord`,
        ``coordinate_frame`` is required.
    projection : `~astropy.modeling.projections.Projection`
        Projection instance - required if there is a celestial component in
        the fiducial.
    transform : `~astropy.modeling.Model` (optional)
        An optional tranform to be prepended to the transform constructed by
        the fiducial point. The number of outputs of this transform must equal
        the number of axes in the coordinate frame.
    name : str
        Name of this WCS.
    bounding_box : tuple
        Domain of this WCS. The format is a list of dictionaries for each
        axis in the input frame
        [{'lower': float, 'upper': float, 'includes_lower': bool,
        'includes_upper': bool, 'step': float}]
    """
    if transform is not None:
        if not isinstance(transform, Model):
            raise UnsupportedTransformError("Expected transform to be an instance"
                                            "of astropy.modeling.Model")
    if domain is not None:
        warnings.warning("'domain' was deprecated in 0.8 and will be removed from next"
                         "version. Use 'bounding_box' instead.")
        bounding_box = _domain_to_bounding_box(domain)
    # transform_outputs = transform.n_outputs
    if isinstance(fiducial, coord.SkyCoord):
        coordinate_frame = CelestialFrame(reference_frame=fiducial.frame,
                                          unit=(fiducial.spherical.lon.unit,
                                                fiducial.spherical.lat.unit))
        fiducial_transform = _sky_transform(fiducial, projection)
    elif isinstance(coordinate_frame, CompositeFrame):
        trans_from_fiducial = []
        for item in coordinate_frame.frames:
            ind = coordinate_frame.frames.index(item)
            try:
                model = frame2transform[item.__class__](fiducial[ind], projection=projection)
            except KeyError:
                raise TypeError("Coordinate frame {0} is not supported".format(item))
            trans_from_fiducial.append(model)
        fiducial_transform = functools.reduce(lambda x, y: x & y,
                                              [tr for tr in trans_from_fiducial])
    else:
        # The case of one coordinate frame with more than 1 axes.
        try:
            fiducial_transform = frame2transform[coordinate_frame.__class__](fiducial,
                                                                             projection=projection)
        except KeyError:
            raise TypeError("Coordinate frame {0} is not supported".format(coordinate_frame))

    if transform is not None:
        forward_transform = transform | fiducial_transform
    else:
        forward_transform = fiducial_transform
    if bounding_box is not None:
        if len(bounding_box) != forward_transform.n_outputs:
            raise ValueError("Expected the number of items in 'bounding_box' to be equal to the "
                             "number of outputs of the forawrd transform.")
        forward_transform.bounding_box = bonding_box[::-1]
    return WCS(output_frame=coordinate_frame, forward_transform=forward_transform,
               name=name)


def _verify_projection(projection):
    if projection is None:
        raise ValueError("Celestial coordinate frame requires a projection to be specified.")
    if not isinstance(projection, projections.Projection):
        raise UnsupportedProjectionError(projection)


def _sky_transform(skycoord, projection):
    """
    A sky transform is a projection, followed by a rotation on the sky.
    """
    _verify_projection(projection)
    lon_pole = _compute_lon_pole(skycoord, projection)
    if isinstance(skycoord, coord.SkyCoord):
        lon, lat = skycoord.spherical.lon, skycoord.spherical.lat
    else:
        lon, lat = skycoord
    sky_rotation = models.RotateNative2Celestial(lon, lat, lon_pole)
    return projection | sky_rotation


def _spectral_transform(fiducial, **kwargs):
    """
    A spectral transform is a shift by the fiducial.
    """
    return models.Shift(fiducial)


def _frame2D_transform(fiducial, **kwargs):
    fiducial_transform = functools.reduce(lambda x, y: x & y,
                                          [models.Shift(val) for val in fiducial])
    return fiducial_transform

frame2transform = {CelestialFrame: _sky_transform,
                   SpectralFrame: _spectral_transform,
                   Frame2D: _frame2D_transform
                   }


@deprecated("0.8", alternative="grid_from_bounding_box")
def grid_from_domain(domain):
    """
    Create a grid of input points from the WCS domain.

    Parameters
    ----------
    domain : list of dicts
        Domain of this WCS. The format is a list of dictionaries for each
        axis in the input frame.
        [{'lower': float, 'upper': float,
        'includes_lower': bool, 'includes_upper': bool, 'step': float}]

    The assumption is the list is in order of X, Y [, Z] and the output will be in the same order.

    For example, if the domain is
    [{'lower': 0, 'upper': 1623}, {'lower': 785, 'upper': 835}] then the output will be:

    array([[[   0,    1,    2, ..., 1620, 1621, 1622],
        [   0,    1,    2, ..., 1620, 1621, 1622],
        [   0,    1,    2, ..., 1620, 1621, 1622],
        ...,
        [   0,    1,    2, ..., 1620, 1621, 1622],
        [   0,    1,    2, ..., 1620, 1621, 1622],
        [   0,    1,    2, ..., 1620, 1621, 1622]],

       [[ 785,  785,  785, ...,  785,  785,  785],
        [ 786,  786,  786, ...,  786,  786,  786],
        [ 787,  787,  787, ...,  787,  787,  787],
        ...,
        [ 832,  832,  832, ...,  832,  832,  832],
        [ 833,  833,  833, ...,  833,  833,  833],
        [ 834,  834,  834, ...,  834,  834,  834]]])

    Returns
    -------
    x, y : ndarray
        Input points.
    """
    slices = [_get_slice(d) for d in domain]
    return np.mgrid[slices[::-1]][::-1]


def grid_from_bounding_box(bounding_box, step=1, center=True):
    """
    Create a grid of input points from the WCS bounding_box.

    Parameters
    ----------
    bounding_box : tuple
        `ref: prop: bounding_box`
    step : scalar or tuple
        Step size for grid in each dimension.  Scalar applies to all dimensions.
    center : bool

    The bounding_box is in order of X, Y [, Z] and the output will be in the
    same order.

    Examples
    --------
    >>> bb = ((-1, 2.9), (6, 7.5))
    >>> grid_from_bounding_box(bb, step=(1, .5))
        [[[-1. ,  0. ,  1. ,  2. ],
         [-1. ,  0. ,  1. ,  2. ],
         [-1. ,  0. ,  1. ,  2. ],
         [-1. ,  0. ,  1. ,  2. ]],

        [[ 6. ,  6. ,  6. ,  6. ],
         [ 6.5,  6.5,  6.5,  6.5],
         [ 7. ,  7. ,  7. ,  7. ],
         [ 7.5,  7.5,  7.5,  7.5]]])


    Returns
    -------
    x, y [, z]: ndarray
        Grid of points.
    """
    if center:
        bb = tuple([(np.floor(b[0] + 0.5), np.ceil(b[1] - .5)) for b in bounding_box])
    else:
        bb = bounding_box

    step = np.atleast_1d(step)
    if len(bb) > 1 and len(step) == 1:
        step = np.repeat(step, len(bb))

    if len(step) != len(bb):
        raise ValueError('`step` must be a scalar, or tuple with length '
                         'matching `bounding_box`')

    slices = []
    for d, s in zip(bb, step):
        slices.append(slice(d[0], d[1] + s, s))
    return np.mgrid[slices[::-1]][::-1]
