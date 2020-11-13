# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility function for WCS

"""
import re
import functools
import numpy as np
from astropy.modeling import models as astmodels
from astropy.modeling import core, projections
from astropy.io import fits
from astropy import coordinates as coords
from astropy import units as u
from astropy.time import Time, TimeDelta


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


class RegionError(Exception):

    def __init__(self, message):
        super(RegionError, self).__init__(message)


class CoordinateFrameError(Exception):

    def __init__(self, message):
        super(CoordinateFrameError, self).__init__(message)


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


def get_values(units, *args):
    """
    Return the values of Quantity objects after optionally converting to units.

    Parameters
    ----------
    units : str or `~astropy.units.Unit` or None
        Units to convert to. The input values are converted to ``units``
        before the values are returned.
    args : `~astropy.units.Quantity`
        Quantity inputs.
    """
    if units is not None:
        result = [a.to_value(unit) for a, unit in zip(args, units)]
    else:
        result = [a.value for a in args]
    return result


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
        p = re.compile(r'ctype[\d]*', re.IGNORECASE)
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
                    pc[i - 1, j - 1] = header['CD{0}_{1}'.format(i, j)]
                else:
                    pc[i - 1, j - 1] = header['PC{0}_{1}'.format(i, j)]
            except KeyError:
                if i == j:
                    pc[i - 1, j - 1] = 1.
                else:
                    pc[i - 1, j - 1] = 0.
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

    translation_models = [astmodels.Shift(-(shift - 1), name='crpix' + str(i + 1))
                          for i, shift in enumerate(crpix)]
    translation = functools.reduce(lambda x, y: x & y, translation_models)

    if not wcs_info['has_cd']:
        # Do not compute scaling since CDELT* = 1 if CD is present.
        scaling_models = [astmodels.Scale(scale, name='cdelt' + str(i + 1))
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
    isnum = True
    if isinstance(val, coords.SkyCoord):
        isnum = False
    elif isinstance(val, u.Quantity):
        isnum = False
    elif isinstance(val, (Time, TimeDelta)):
        isnum = False
    elif (isinstance(val, np.ndarray)
          and not np.issubdtype(val.dtype, np.floating)
          and not np.issubdtype(val.dtype, np.integer)):
        isnum = False
    return isnum
