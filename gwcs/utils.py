# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility function for WCS

"""
from __future__ import absolute_import, division, unicode_literals, print_function

import re
import functools
import numpy as np
from astropy.modeling import models as astmodels
from astropy.modeling.models import Mapping
from astropy.modeling import core, projections
from astropy.io import fits
from astropy import coordinates as coords
from astropy import units as u

from astropy.utils.decorators import deprecated


# these ctype values do not include yzLN and yzLT pairs
sky_pairs = {"equatorial": ["RA", "DEC"],
             "ecliptic": ["ELON", "ELAT"],
             "galactic": ["GLON", "GLAT"],
             "helioecliptic": ["HLON", "HLAT"],
             "supergalactic": ["SLON", "SLAT"],
             # "spec": specsystems
             }

radesys = ['ICRS', 'FK5', 'FK4', 'FK4-NO-E', 'GAPPT', 'GALACTIC']


class UnsupportedTransformError(Exception):

    def __init__(self, message):
        super(UnsupportedTransformError, self).__init__(message)


class UnsupportedProjectionError(Exception):
    def __init__(self, code):
        message = "Unsupported projection: {0}".format(code)
        super(UnsupportedProjectionError, self).__init__(message)


class DimensionalityError(Exception):

    def __init__(self, message):
        super(DimensionalityError, self).__init__(message)


class RegionError(Exception):

    def __init__(self, message):
        super(RegionError, self).__init__(message)


class CoordinateFrameError(Exception):

    def __init__(self, message):
        super(CoordinateFrameError, self).__init__(message)


def _domain_to_bounding_box(domain):
    bb = tuple([(item['lower'], item['upper']) for item in domain])
    if len(bb) == 1:
        bb = bb[0]
    return bb

def _toindex(value):
    """
    Convert value to an int or an int array.

    Input coordinates converted to integers
    corresponding to the center of the pixel.
    The convention is that the center of the pixel is
    (0, 0), while the lower left corner is (-0.5, -0.5).
    The outputs are used to index the mask.

    Examples
    --------
    >>> _toindex(np.array([-0.5, 0.49999]))
    array([0, 0])
    >>> _toindex(np.array([0.5, 1.49999]))
    array([1, 1])
    >>> _toindex(np.array([1.5, 2.49999]))
    array([2, 2])
    """
    indx = np.asarray(np.floor(np.asarray(value) + 0.5), dtype=np.int)
    return indx


@deprecated("0.8", message="_domain has been deprecated in 0.8 and will be"
            "removed in the next version", alternative="bounding_box")
def _domain_to_bounds(domain):
    def _get_bounds(axis_domain):
        step = axis_domain.get('step', 1)
        x = axis_domain['lower'] if axis_domain.get('includes_lower', True) \
            else axis_domain['lower'] + step
        y = axis_domain['upper'] - 1 if not axis_domain.get('includes_upper', False) \
            else axis_domain['upper']
        return (x, y)

    bounds = [_get_bounds(d) for d in domain]
    return bounds


def _get_slice(axis_domain):
    """ TODO: Remove when domain is removed"""
    step = axis_domain.get('step', 1)
    x = axis_domain['lower'] if axis_domain.get('includes_lower', True) \
        else axis_domain['lower'] + step
    y = axis_domain['upper'] if not axis_domain.get('includes_upper', False) \
        else axis_domain['upper'] + step
    return slice(x, y, step)


def axis_domain_to_slice(axis_domain, step):
    """
    Return a slice from the bounding_box for an axis.

    Parameters
    ----------
    axis_domain : tuple
        The range of acceptable input values for an axis, usually from bounding_box.
    step : int
        A step to use in the slice.
    """
    x, y = axis_domain
    return slice(x, y, step)


def _get_values(units, *args):
    """
    Return the values of SkyCoord or Quantity objects.

    Parameters
    ----------
    units : str or `~astropy.units.Unit`
        Units of the wcs object.
        The input values are converted to ``units`` before the values are returned.
    """
    val = []
    values = []
    for arg in args:
        if isinstance(arg, coords.SkyCoord):
            try:
                lon = arg.data.lon
                lat = arg.data.lat
            except AttributeError:
                lon = arg.spherical.lon
                lat = arg.spherical.lat
            val.extend([lon, lat])
        elif isinstance(arg, u.Quantity):
            val.append(arg)
        else:
            raise TypeError("Unsupported coordinate type {}".format(arg))
    for va, un in zip(val, units):
        values.append(va.to(un).value)
    return values


def _compute_lon_pole(skycoord, projection):
    """
    Compute the longitude of the celestial pole of a standard frame in the
    native frame.

    This angle then can be used as one of the Euler angles (the other two being skyccord)
    to rotate the native frame into the standard frame ``skycoord.frame``.

    Parameters
    ----------
    skycoord : `astropy.coordinates.SkyCoord`, or
               sequence of floats or `~astropy.units.Quantity` of length 2
        The fiducial point of the native coordinate system.
        If tuple, its length is 2
    projection : `astropy.modeling.projections.Projection`
        A Projection instance.

    Returns
    -------
    lon_pole : float or `~astropy/units.Quantity`
        Native longitude of the celestial pole [deg].

    TODO: Implement all projections
        Currently this only supports Zenithal and Cylindrical.
    """
    if isinstance(skycoord, coords.SkyCoord):
        lat = skycoord.spherical.lat
        unit = u.deg
    else:
        lon, lat = skycoord
        if isinstance(lat, u.Quantity):
            unit = u.deg
        else:
            unit = None
    if isinstance(projection, projections.Zenithal):
        lon_pole = 180
    elif isinstance(projection, projections.Cylindrical):
        if lat >= 0:
            lon_pole = 0
        else:
            lon_pole = 180
    else:
        raise UnsupportedProjectionError("Projection {0} is not supported.".format(projection))
    if unit is not None:
        lon_pole = lon_pole * unit
    return lon_pole


def get_projcode(wcs_info):
    # CTYPE here is only the imaging CTYPE keywords
    sky_axes, _, _ = get_axes(wcs_info)
    if not sky_axes:
        return None
    projcode = wcs_info['CTYPE'][sky_axes[0]][5:8].upper()
    if projcode not in projections.projcodes:
        raise UnsupportedProjectionError('Projection code %s, not recognized' % projcode)
        #projcode = None
    return projcode


def read_wcs_from_header(header):
    """
    Extract basic FITS WCS keywords from a FITS Header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS Header with WCS information.

    Returns
    -------
    wcs_info : dict
        A dictionary with WCS keywords.
    """
    wcs_info = {}

    try:
        wcs_info['WCSAXES'] = header['WCSAXES']
    except KeyError:
        p = re.compile('ctype[\d]*', re.IGNORECASE)
        ctypes = header['CTYPE*']
        keys = list(ctypes.keys())
        for key in keys[::-1]:
            if p.split(key)[-1] != "":
                keys.remove(key)
        wcs_info['WCSAXES'] = len(keys)
    wcsaxes = wcs_info['WCSAXES']
    # if not present call get_csystem
    wcs_info['RADESYS'] = header.get('RADESYS', 'ICRS')
    wcs_info['VAFACTOR'] = header.get('VAFACTOR', 1)
    wcs_info['NAXIS'] = header.get('NAXIS', 0)
    # date keyword?
    # wcs_info['DATEOBS'] = header.get('DATE-OBS', 'DATEOBS')
    wcs_info['EQUINOX'] = header.get("EQUINOX", None)
    wcs_info['EPOCH'] = header.get("EPOCH", None)
    wcs_info['DATEOBS'] = header.get("MJD-OBS", header.get("DATE-OBS", None))

    ctype = []
    cunit = []
    crpix = []
    crval = []
    cdelt = []
    for i in range(1, wcsaxes + 1):
        ctype.append(header['CTYPE{0}'.format(i)])
        cunit.append(header.get('CUNIT{0}'.format(i), None))
        crpix.append(header.get('CRPIX{0}'.format(i), 0.0))
        crval.append(header.get('CRVAL{0}'.format(i), 0.0))
        cdelt.append(header.get('CDELT{0}'.format(i), 1.0))

    if 'CD1_1' in header:
        wcs_info['has_cd'] = True
    else:
        wcs_info['has_cd'] = False
    pc = np.zeros((wcsaxes, wcsaxes))
    for i in range(1, wcsaxes + 1):
        for j in range(1, wcsaxes + 1):
            try:
                if wcs_info['has_cd']:
                    pc[i-1, j-1] = header['CD{0}_{1}'.format(i, j)]
                else:
                    pc[i-1, j-1] = header['PC{0}_{1}'.format(i, j)]
            except KeyError:
                if i == j:
                    pc[i-1, j-1] = 1.
                else:
                    pc[i-1, j-1] = 0.
    wcs_info['CTYPE'] = ctype
    wcs_info['CUNIT'] = cunit
    wcs_info['CRPIX'] = crpix
    wcs_info['CRVAL'] = crval
    wcs_info['CDELT'] = cdelt
    wcs_info['PC'] = pc
    return wcs_info


def get_axes(header):
    """
    Matches input with spectral and sky coordinate axes.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header (or dict) with basic WCS information.

    Returns
    -------
    sky_inmap, spectral_inmap, unknown : lists
        indices in the input representing sky and spectral cordinates.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    # Split each CTYPE value at "-" and take the first part.
    # This should represent the coordinate system.
    ctype = [ax.split('-')[0].upper() for ax in wcs_info['CTYPE']]
    sky_inmap = []
    spec_inmap = []
    unknown = []
    skysystems = np.array(list(sky_pairs.values())).flatten()
    for ax in ctype:
        ind = ctype.index(ax)
        if ax in specsystems:
            spec_inmap.append(ind)
        elif ax in skysystems:
            sky_inmap.append(ind)
        else:
            unknown.append(ind)

    if sky_inmap:
        _is_skysys_consistent(ctype, sky_inmap)

    return sky_inmap, spec_inmap, unknown


def _is_skysys_consistent(ctype, sky_inmap):
    """ Determine if the sky axes in CTYPE mathch to form a standard celestial system."""

    for item in sky_pairs.values():
        if ctype[sky_inmap[0]] == item[0]:
            if ctype[sky_inmap[1]] != item[1]:
                raise ValueError(
                    "Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            break
        elif ctype[sky_inmap[1]] == item[0]:
            if ctype[sky_inmap[0]] != item[1]:
                raise ValueError(
                    "Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            sky_inmap = sky_inmap[::-1]
            break


specsystems = ["WAVE", "FREQ", "ENER", "WAVEN", "AWAV",
               "VRAD", "VOPT", "ZOPT", "BETA", "VELO"]

sky_systems_map = {'ICRS': coords.ICRS,
                   'FK5': coords.FK5,
                   'FK4': coords.FK4,
                   'FK4NOE': coords.FK4NoETerms,
                   'GAL': coords.Galactic,
                   'HOR': coords.AltAz
                   }


def make_fitswcs_transform(header):
    """
    Create a basic FITS WCS transform.
    It does not include distortions.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header (or dict) with basic WCS information

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")
    transforms = []
    wcs_linear = fitswcs_linear(wcs_info)
    transforms.append(wcs_linear)
    wcs_nonlinear = fitswcs_nonlinear(wcs_info)
    if wcs_nonlinear is not None:
        transforms.append(wcs_nonlinear)
    return functools.reduce(core._model_oper('|'), transforms)


def fitswcs_linear(header):
    """
    Create a WCS linear transform from a FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header or dict with basic FITS WCS keywords.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    pc = wcs_info['PC']
    # get the part of the PC matrix corresponding to the imaging axes
    sky_axes, spec_axes, unknown = get_axes(wcs_info)
    if pc.shape != (2, 2):
        if sky_axes:
            i, j = sky_axes
        elif unknown and len(unknown) == 2:
            i, j = unknown
        sky_pc = np.zeros((2, 2))
        sky_pc[0, 0] = pc[i, i]
        sky_pc[0, 1] = pc[i, j]
        sky_pc[1, 0] = pc[j, i]
        sky_pc[1, 1] = pc[j, j]
        pc = sky_pc.copy()

    sky_axes.extend(unknown)
    if sky_axes:
        crpix = []
        cdelt = []
        for i in sky_axes:
            crpix.append(wcs_info['CRPIX'][i])
            cdelt.append(wcs_info['CDELT'][i])
        #crpix = wcs_info['CRPIX'][sky_axes]
        #cdelt = wcs_info['CDELT'][sky_axes]
    else:
        cdelt = wcs_info['CDELT']
        crpix = wcs_info['CRPIX']

    # if wcsaxes == 2:
    rotation = astmodels.AffineTransformation2D(matrix=pc, name='pc_matrix')
    # elif wcsaxes == 3 :
    # rotation = AffineTransformation3D(matrix=matrix)
    # else:
    # raise DimensionsError("WCSLinearTransform supports only 2 or 3 dimensions, "
    # "{0} given".format(wcsaxes))

    translation_models = [astmodels.Shift(-shift, name='crpix' + str(i + 1))
                          for i, shift in enumerate(crpix)]
    translation = functools.reduce(lambda x, y: x & y, translation_models)

    if not wcs_info['has_cd']:
        # Do not compute scaling since CDELT* = 1 if CD is present.
        scaling_models = [astmodels.Scale(scale, name='cdelt' + str(i + 1)) \
                          for i, scale in enumerate(cdelt)]

        scaling = functools.reduce(lambda x, y: x & y, scaling_models)
        wcs_linear = translation | rotation | scaling
    else:
        wcs_linear = translation | rotation

    return wcs_linear


def fitswcs_nonlinear(header):
    """
    Create a WCS linear transform from a FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header or dict with basic FITS WCS keywords.
    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    transforms = []
    projcode = get_projcode(wcs_info)
    if projcode is not None:
        projection = create_projection_transform(projcode).rename(projcode)
        transforms.append(projection)
    # Create the sky rotation transform
    sky_axes, _, _ = get_axes(wcs_info)
    if sky_axes:
        phip, lonp = [wcs_info['CRVAL'][i] for i in sky_axes]
        # TODO: write "def compute_lonpole(projcode, l)"
        # Set a defaul tvalue for now
        thetap = 180
        n2c = astmodels.RotateNative2Celestial(phip, lonp, thetap, name="crval")
        transforms.append(n2c)
    if transforms:
        return functools.reduce(core._model_oper('|'), transforms)
    else:
        return None


def create_projection_transform(projcode):
    """
    Create the non-linear projection transform.

    Parameters
    ----------
    projcode : str
        FITS WCS projection code.

    Returns
    -------
    transform : astropy.modeling.Model
        Projection transform.
    """

    projklassname = 'Pix2Sky_' + projcode
    try:
        projklass = getattr(projections, projklassname)
    except AttributeError:
        raise UnsupportedProjectionError(projcode)

    projparams = {}
    return projklass(**projparams)


def isnumerical(val):
    """
    Determine if a value is numerical (number or np.array of numbers).
    """
    dtypes = ['uint64', 'float64', 'int8', 'int64', 'int16', 'uint16', 'uint8',
              'float32', 'int32', 'uint32']
    isnum = True
    if isinstance(val, coords.SkyCoord):
        isnum = False
    elif isinstance(val, u.Quantity):
        isnum = False
    elif isinstance(val, np.ndarray) and val.dtype not in dtypes:
        isnum = False
    return isnum


# ######### axis separability #########
# Functions to determine axis separability
# The interface will change most likely


def _compute_n_outputs(left, right):
    """
    Compute the number of outputs of two models.

    The two models are the left and right model to an operation in
    the expression tree of a compound model.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    """
    if isinstance(left, core.Model):
        lnout = left.n_outputs
    else:
        lnout = left.shape[0]
    if isinstance(right, core.Model):
        rnout = right.n_outputs
    else:
        rnout = right.shape[0]
    noutp = lnout + rnout
    return noutp


def _arith_oper(left, right):
    """
    Function corresponding to one of the arithmetic operators ['+', '-'. '*', '/', '**'].

    This always returns a nonseparable outputs.


    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    # models have the same number of outputs
    if isinstance(left, core.Model):
        noutp = left.n_outputs
    else:
        noutp = left.shape[0]
    if isinstance(left, core.Model):
        ninp = left.n_inputs
    else:
        ninp = left.shape[1]
    result = np.ones((noutp, ninp))
    return result


def _coord_matrix(model, pos, noutp):
    """
    Create an array representing inputs and outputs of a simple model.

    The array has a shape (noutp, model.n_inputs).

    Parameters
    ----------
    model : `astropy.modeling.Model`
        model
    pos : str
        Position of this model in the expression tree.
        One of ['left', 'right'].
    noutp : int
        Number of outputs of the compound model of which the input model
        is a left or right child.

    Examples
    --------
    >>> _coord_matrix(Shift(1), 'left', 2)
        array([[ 1.],
        [ 0.]])
    >>> _coord_matrix(Shift(1), 'right', 2)
        array([[ 0.],
               [ 1.]])
    >>> _coord_matrix(Rotation2D, 'right', 4)
        array([[ 0.,  0.],
            [ 0.,  0.],
            [ 1.,  1.],
            [ 1.,  1.]])
    """
    if isinstance(model, Mapping):
        axes = []
        for i in model.mapping:
            axis = np.zeros((model.n_inputs,))
            axis[i] = 1
            axes.append(axis)
        m = np.vstack(axes)
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[: model.n_outputs, :model.n_inputs] = m
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = m
        return mat
    if not model.separable:
        # this does not work for more than 2 coordinates
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[:model.n_outputs, : model.n_inputs] = 1
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = 1
    else:
        mat = np.zeros((noutp, model.n_inputs))

        for i in range(model.n_inputs):
            mat[i, i] = 1
        if pos == 'right':
            mat = np.roll(mat, (noutp - model.n_outputs))
    return mat


def _cstack(left, right):
    """
    Function corresponding to '&' operation.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.

    """
    noutp = _compute_n_outputs(left, right)

    if isinstance(left, core.Model):
        cleft = _coord_matrix(left, 'left', noutp)
    else:
        cleft = np.zeros((noutp, left.shape[1]))
        cleft[: left.shape[0], :left.shape[1]] = left
    if isinstance(right, core.Model):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
        cright[-right.shape[0]:, -right.shape[1]:] = 1

    return np.hstack([cleft, cright])


def _cdot(left, right):
    """
    Function corresponding to "|" operation.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    left, right = right, left
    if isinstance(right, core.Model):
        cright = _coord_matrix(right, 'right', right.n_outputs)
    else:
        cright = right
    if isinstance(left, core.Model):
        cleft = _coord_matrix(left, 'left', left.n_outputs)
    else:
        cleft = left
    result = np.dot(cleft, cright)
    return result


def _separable(transform):
    """
    Calculate the separability of outputs.

    Parameters
    ----------
    transform : `astropy.modeling.Model`
        A transform (usually a compound model).

    Returns
    -------
    is_separable : ndarray of dtype np.bool
        An array of shape (transform.n_outputs,) of boolean type
        Each element represents the separablity of the corresponding output.

    Examples
    --------
    >>> separable(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([ True,  True], dtype=bool)
    >>> separable(Shift(1) & Shift(2) | Rotation2D(2))
        array([False, False], dtype=bool)
    >>> separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | Polynomial2D(1) & Polynomial2D(2))
        array([False, False], dtype=bool)
    >>> separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))
        array([ True,  True,  True,  True], dtype=bool)

    """
    if isinstance(transform, core._CompoundModel):
        is_separable = transform._tree.evaluate(_operators)
    elif isinstance(transform, core.Model):
        is_separable = _coord_matrix(transform, 'left', transform.n_outputs)
    return is_separable


def is_separable(transform):
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        is_separable = np.array([False] * transform.n_outputs)
        return is_separable
    separable_matrix = _separable(transform)
    is_separable = separable_matrix.sum(1)
    is_separable = np.where(is_separable != 1, False, True)
    return is_separable


def separable_axes(wcsobj, start_frame=None, end_frame=None):
        """
        Computes the separability of axes in ``end_frame``.

        Returns a 1D boolean array of size frame.naxes where True means
        the axis is completely separable and False means the axis is nonseparable
        from at least one other axis.

        Parameters
        ----------
        wcsobj : `~gwcs.wcs.WCS`
            WCS object
        start_frame : `~gwcs.coordinate_frames.CoordinateFrame`
            A frame in the WCS pipeline.
            The transform between start_frame and the end frame is used to compute the
            mapping inputs: outputs.
            If None the input_frame is used as start_frame.
        end_frame : `~gwcs.coordinate_frames.CoordinateFrame`
            A frame in the WCS pipeline.
            The transform between start_frame and the end frame is used to compute the
            mapping inputs: outputs.
            If None wcsobj.output_frame is used.

        See Also
        --------
        input_axes : For each output axis return the input axes contributing to it.

        """
        if wcsobj is not None:
            if start_frame is None:
                start_frame = wcsobj.input_frame
            else:
                if start_frame not in wcsobj.available_frames:
                    raise ValueError("Unrecognized frame {0}".format(start_frame))
            if end_frame is None:
                end_frame = wcsobj.output_frame
            else:
                if end_frame not in wcsobj.available_frames:
                    raise ValueError("Unrecognized frame {0}".format(end_frame))
            transform = wcsobj.get_transform(start_frame, end_frame)
        else:
            raise ValueError("A starting frame is needed to determine separability of axes.")

        sep = is_separable(transform)
        return [sep[ax] for ax in end_frame.axes_order]


_operators = {'&': _cstack, '|': _cdot, '+': _arith_oper, '-': _arith_oper,
              '*': _arith_oper, '/': _arith_oper, '**': _arith_oper}
