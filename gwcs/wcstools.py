# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools
import warnings
import numpy as np
from astropy.modeling.core import Model
from astropy.modeling import projections
from astropy.modeling import models, fitting
from astropy import coordinates as coord
from astropy import units as u

from .coordinate_frames import *  # noqa
from .utils import UnsupportedTransformError, UnsupportedProjectionError
from .utils import _compute_lon_pole


__all__ = ['wcs_from_fiducial', 'grid_from_bounding_box', 'wcs_from_points']


def wcs_from_fiducial(fiducial, coordinate_frame=None, projection=None,
                      transform=None, name='', bounding_box=None, input_frame=None):
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
        The bounding box over which the WCS is valid.
        It is a tuple of tuples of size 2 where each tuple
        represents a range of (low, high) values. The ``bounding_box`` is in the order of
        the axes, `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
        For two inputs and axes_order(0, 1) the bounding box is ((xlow, xhigh), (ylow, yhigh)).
    input_frame : ~gwcs.coordinate_frames.CoordinateFrame`
        The input coordinate frame.
    """
    from .wcs import WCS

    if transform is not None:
        if not isinstance(transform, Model):
            raise UnsupportedTransformError("Expected transform to be an instance"
                                            "of astropy.modeling.Model")

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
        forward_transform.bounding_box = bounding_box[::-1]
    if input_frame is None:
        input_frame = 'detector'
    return WCS(output_frame=coordinate_frame, input_frame=input_frame,
               forward_transform=forward_transform, name=name)


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


def grid_from_bounding_box(bounding_box, step=1, center=True):
    """
    Create a grid of input points from the WCS bounding_box.

    Note: If ``bbox`` is a tuple describing the range of an axis in ``bounding_box``,
          ``x.5`` is considered part of the next pixel in ``bbox[0]``
          and part of the previous pixel in ``bbox[1]``. In this way if
          ``bbox`` describes the edges of an image the indexing includes
          only pixels within the image.

    Parameters
    ----------
    bounding_box : tuple
        The bounding_box of a WCS object, `~gwcs.wcs.WCS.bounding_box`.
    step : scalar or tuple
        Step size for grid in each dimension.  Scalar applies to all dimensions.
    center : bool

    The bounding_box is in order of X, Y [, Z] and the output will be in the
    same order.

    Examples
    --------
    >>> bb = ((-1, 2.9), (6, 7.5))
    >>> grid_from_bounding_box(bb, step=(1, .5), center=False)
    array([[[-1. ,  0. ,  1. ,  2. ,  3. ],
            [-1. ,  0. ,  1. ,  2. ,  3. ],
            [-1. ,  0. ,  1. ,  2. ,  3. ],
            [-1. ,  0. ,  1. ,  2. ,  3. ]],
           [[ 6. ,  6. ,  6. ,  6. ,  6. ],
            [ 6.5,  6.5,  6.5,  6.5,  6.5],
            [ 7. ,  7. ,  7. ,  7. ,  7. ],
            [ 7.5,  7.5,  7.5,  7.5,  7.5]]])

    >>> bb = ((-1, 2.9), (6, 7.5))
    >>> grid_from_bounding_box(bb)
    array([[[-1.,  0.,  1.,  2.,  3.],
            [-1.,  0.,  1.,  2.,  3.]],
           [[ 6.,  6.,  6.,  6.,  6.],
            [ 7.,  7.,  7.,  7.,  7.]]])

    Returns
    -------
    x, y [, z]: ndarray
        Grid of points.
    """
    def _bbox_to_pixel(bbox):
        return (np.floor(bbox[0] + 0.5), np.ceil(bbox[1] - 0.5))
    # 1D case
    if np.isscalar(bounding_box[0]):
        nd = 1
        bounding_box = (bounding_box, )
    else:
        nd = len(bounding_box)
    if center:
        bb = tuple([_bbox_to_pixel(bb) for bb in bounding_box])
    else:
        bb = bounding_box

    step = np.atleast_1d(step)
    if nd > 1 and len(step) == 1:
        step = np.repeat(step, nd)

    if len(step) != len(bb):
        raise ValueError('`step` must be a scalar, or tuple with length '
                         'matching `bounding_box`')

    slices = []
    for d, s in zip(bb, step):
        slices.append(slice(d[0], d[1] + s, s))
    grid = np.mgrid[slices[::-1]][::-1]
    if nd == 1:
        return grid[0]
    return grid


def wcs_from_points(xy, world_coords, proj_point='center',
                    projection=projections.Sky2Pix_TAN(), poly_degree=4,
                    polynomial_type='polynomial'):
    """
    Given two matching sets of coordinates on detector and sky, compute the WCS.

    Notes
    -----
    This function implements the following algorithm:
    ``world_coords`` are transformed to a projection plane using the specified
    projection. A polynomial fits ``xy`` and the projected coordinates.
    The fitted polynomials and the projection transforms are combined into a
    tranform from detector to sky. The input coordinate frame is set to
    ``detector``. The output coordinate frame is initialized based on the frame
    in the fiducial.


    Parameters
    ----------
    xy : tuple of 2 ndarrays
        Points in the input cooridnate frame - x, y inputs.
    world_coords : `~astropy.coordinates.SkyCoord`
        Points in the output coordinate frame.
        The order matches the order of ``xy``.
    proj_point : `~astropy.coordinates.SkyCoord`
        A fiducial point in the output coordinate frame. If set to 'center'
        (default), the geometric center of input world
        coordinates will be used as the projection point. To specify an exact
        point for the projection, a Skycoord object with a coordinate pair can
        be passed in.
    projection : `~astropy.modeling.projections.Projection`
        A projection type. One of the projections in
        `~astropy.modeling.projections.projcodes`. Defaults to TAN projection
        (`astropy.modeling.projections.Sky2Pix_TAN`).
    poly_degree : int
        Degree of polynomial model to be fit to data. Defaults to 4.
    polynomial_type : str
        one of "polynomial", "chebyshev", "legendre". Defaults to "polynomial".

    Returns
    -------
    wcsobj : `~gwcs.wcs.WCS`
        a WCS object for this observation.
    """
    from .wcs import WCS

    supported_poly_types = {"polynomial": models.Polynomial2D,
                            "chebyshev": models.Chebyshev2D,
                            "legendre": models.Legendre2D
                            }

    x, y = xy

    if not isinstance(world_coords, coord.SkyCoord):
        raise TypeError('`world_coords` must be an `~astropy.coordinates.SkyCoord`')
    try:
        lon, lat = world_coords.data.lon.deg, world_coords.data.lat.deg
    except AttributeError:
        unit_sph = world_coords.unit_spherical
        lon, lat = unit_sph.lon.deg, unit_sph.lat.deg

    if isinstance(proj_point, coord.SkyCoord):
        assert proj_point.size == 1
        proj_point.transform_to(world_coords)
        crval = (proj_point.data.lon, proj_point.data.lat)
        frame = proj_point.frame
    elif proj_point == 'center':  # use center of input points
        sc1 = SkyCoord(lon.min()*u.deg, lat.max()*u.deg)
        sc2 = SkyCoord(lon.max()*u.deg, lat.min()*u.deg)
        pa = sc1.position_angle(sc2)
        sep = sc1.separation(sc2)
        midpoint_sc = sc1.directional_offset_by(pa, sep/2)
        crval = (midpoint_sc.data.lon, midpoint_sc.data.lat)
        frame = sc1.frame
    else:
        raise ValueError("`proj_point` must be set to 'center', or an" +
                         "`~astropy.coordinates.SkyCoord` object with " +
                         "a pair of points.")

    if not isinstance(projection, projections.Projection):
        raise UnsupportedProjectionError("Unsupported projection code {0}".format(projection))

    if polynomial_type not in supported_poly_types.keys():
        raise ValueError("Unsupported polynomial_type: {}. "
                         "Only one of {} is supported.".format(polynomial_type,
                                                               supported_poly_types.keys()))

    skyrot = models.RotateCelestial2Native(crval[0], crval[1], 180*u.deg)
    trans = (skyrot | projection)
    projection_x, projection_y = trans(lon, lat)
    poly = supported_poly_types[polynomial_type](poly_degree)

    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poly_x = fitter(poly, x, y, projection_x)
        poly_y = fitter(poly, x, y, projection_y)
        distortion = models.Mapping((0, 1, 0, 1)) | poly_x & poly_y

        poly_x_inverse = fitter(poly, projection_x, projection_y, x)
        poly_y_inverse = fitter(poly, projection_x, projection_y, y)
        distortion.inverse = models.Mapping((0, 1, 0, 1)) | poly_x_inverse & poly_y_inverse

    transform = distortion | projection.inverse | skyrot.inverse

    skyframe = CelestialFrame(reference_frame=frame)
    detector = Frame2D(name="detector")
    pipeline = [(detector, transform),
                (skyframe, None)]

    return WCS(pipeline)
