# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility function for WCS

"""
from __future__ import absolute_import, division, unicode_literals, print_function


import numpy as np
from astropy.modeling import projections
from astropy.modeling import models as astmodels
from astropy.modeling import core
from astropy.io import fits
import functools

try:
    from astropy import time
    HAS_TIME = True
except ImportError:
    HAS_TIME = False
from astropy import coordinates
from astropy import coordinates as coord


# these ctype values do not include yzLN and yzLT pairs
sky_pairs = {"equatorial": ["RA--", "DEC-"],
             "ecliptic": ["ELON", "ELAT"],
             "galactic": ["GLON", "GLAT"],
             "helioecliptic": ["HLON", "HLAT"],
             "supergalactic": ["SLON", "SLAT"],
             #"spec": specsystems
             }

radesys = ['ICRS', 'FK5', 'FK4', 'FK4-NO-E', 'GAPPT', 'GALACTIC']


class UnsupportedTransformError(Exception):

    def __init__(self, message):
        super(UnsupportedTransformError, self).__init__(message)


class UnsupportedProjectionError(Exception):
    def __init__(self, code):
        message = "Unsupported projection: {0}".format(code)
        super(UnsupportedProjectionError, self).__init__(message)


class ModelDimensionalityError(Exception):

    def __init__(self, message):
        super(ModelDimensionalityError, self).__init__(message)


class RegionError(Exception):

    def __init__(self, message):
        super(RegionError, self).__init__(message)


class CoordinateFrameError(Exception):

    def __init__(self, message):
        super(CoordinateFrameError, self).__init__(message)


def get_projcode(ctype):
    # CTYPE here is only the imaging CTYPE keywords
    projcode = ctype[0][5:8].upper()
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
        p = re.compile('ctype[\d]*', re.IGNORECASE)
        ctypes = header['CTYPE*']
        keys = ctype.keys()
        for key in keys[::-1]:
            if p.split(k)[-1] != "":
                keys.remove(k)
        wcs_info['WCSAXES'] = len(keys)
    wcsaxes = wcs_info['WCSAXES']
    # if not present call get_csystem
    wcs_info['RADESYS'] = header.get('RADESYS', 'ICRS')
    wcs_info['VAFACTOR'] = header.get('VAFACTOR', 1)
    wcs_info['NAXIS'] = header.get('NAXIS', 0)
    # date keyword?
    #wcs_info['DATEOBS'] = header.get('DATE-OBS', 'DATEOBS')
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
    for i in range(1, wcsaxes+1):
        for j in range(1, wcsaxes+1):
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
    sky_inmap, spectral_inmap : tuples
        indices in the input representing sky and spectral cordinates.

    """
    if isinstance(header, fits.Header):
        wcs_info = util.read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    ctype = [ax[:4] for ax in wcs_info['CTYPE']]
    sky_inmap = []
    spec_inmap = []
    for ax in ctype:
        if ax.upper() in specsystems:
            spec_inmap.append(ctype.index(ax))
        else:
            sky_inmap.append(ctype.index(ax))
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
    return sky_inmap, spec_inmap


specsystems = ["WAVE", "FREQ", "ENER", "WAVEN", "AWAV",
               "VRAD", "VOPT", "ZOPT", "BETA", "VELO"]

sky_systems_map = {'ICRS': coord.ICRS,
                   'FK5': coord.FK5,
                   'FK4': coord.FK4,
                   'FK4NOE': coord.FK4NoETerms,
                   'GAL': coord.Galactic,
                   'HOR': coord.AltAz
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
    wcs_linear = fitswcs_linear(wcs_info)
    projcode = get_projcode(wcs_info['CTYPE'])
    projection = create_projection_transform(projcode)

    # Create the sky rotation transform
    phip, lonp = wcs_info['CRVAL']
    # TODO: write "def compute_lonpole(projcode, l)"
    # Set a defaul tvalue for now
    thetap = 180
    n2c = astmodels.RotateNative2Celestial(phip, lonp, thetap)

    return functools.reduce(core._model_oper('|'), [wcs_linear, projection, n2c])


def fitswcs_linear(header):
    """
    Create a WCS linear transform from a FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header or dict with basic FITS WCS keywords.

    """
    if isinstance(header, fits.Header):
        wcs_info = util.read_wcs_from_header(wcs_info)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    wcsaxes = wcs_info['WCSAXES']

    pc = wcs_info['PC']
    # get the part of the PC matrix corresponding to the imaging axes
    sky_axes = None
    if pc.shape != (2, 2):
        sky_axes, _ = util.get_axes(wcs_info)
        i, j = sky_axes
        sky_pc = np.zeros((2,2))
        sky_pc[0, 0] = pc[i, i]
        sky_pc[0, 1] = pc[i, j]
        sky_pc[1, 0] = pc[j, i]
        sky_pc[1, 1] = pc[j, j]
        pc = sky_pc.copy()


    if sky_axes is not None:
        crpix = []
        cdelt = []
        for i in sky_axes:
            crpix.append(wcs_info['CRPIX'][i])
            cdelt.append(wcs_info['CDELT'][i])
    else:
        cdelt = wcs_info['CDELT']
        crpix = wcs_info['CRPIX']

    if wcsaxes == 2:
        rotation = astmodels.AffineTransformation2D(matrix=pc)
    #elif wcsaxes == 3 :
        #rotation = AffineTransformation3D(matrix=matrix)
    #else:
        #raise DimensionsError("WCSLinearTransform supports only 2 or 3 dimensions, "
                          #"{0} given".format(wcsaxes))

    translation_models = [astmodels.Shift(-shift) for shift in crpix]
    translation = functools.reduce(core._model_oper('&'), translation_models)

    if not wcs_info['has_cd']:
        # Do not compute scaling since CDELT* = 1 if CD is present.
        scaling_models = [astmodels.Scale(scale) for scale in cdelt]
        scaling = functools.reduce(core._model_oper('&'), scaling_models)
        wcs_linear = translation | rotation | scaling
    else:
        wcs_linear = translation | rotation

    return wcs_linear


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

    projparams={}
    return projklass(**projparams)
