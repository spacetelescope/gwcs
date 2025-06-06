# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility function for WCS

"""

import functools
import re

import astropy.units as u
import numpy as np
from astropy import coordinates as coords
from astropy.io import fits
from astropy.modeling import core, projections
from astropy.modeling import models as astmodels
from astropy.wcs import Celprm

# these ctype values do not include yzLN and yzLT pairs
sky_pairs = {
    "equatorial": ["RA", "DEC"],
    "ecliptic": ["ELON", "ELAT"],
    "galactic": ["GLON", "GLAT"],
    "helioecliptic": ["HLON", "HLAT"],
    "supergalactic": ["SLON", "SLAT"],
    # "spec": specsystems
}

radesys = ["ICRS", "FK5", "FK4", "FK4-NO-E", "GAPPT", "GALACTIC"]


class UnsupportedTransformError(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsupportedProjectionError(Exception):
    def __init__(self, code):
        message = f"Unsupported projection: {code}"
        super().__init__(message)


class RegionError(Exception):
    def __init__(self, message):
        super().__init__(message)


class CoordinateFrameError(Exception):
    def __init__(self, message):
        super().__init__(message)


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
    return np.asarray(np.floor(np.asarray(value) + 0.5), dtype=int)


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
        result = [a.to_value(unit) for a, unit in zip(args, units, strict=False)]
    else:
        result = [a.value for a in args]
    return result


def _compute_lon_pole(skycoord, projection):
    """
    Compute the longitude of the celestial pole of a standard
    frame in the native frame.

    This angle then can be used as one of the Euler angles
    (the other two being skycoord) to rotate the native frame into the
    standard frame ``skycoord.frame``.

    Parameters
    ----------
    skycoord : `astropy.coordinates.SkyCoord`, or
               sequence of floats or `~astropy.units.Quantity` of length 2
        The celestial longitude and latitude of the fiducial point - typically
        right ascension and declination. These are given by the ``CRVALia``
        keywords in ``FITS``.

    projection : `astropy.modeling.projections.Projection`
        A `~astropy.modeling.projections.Projection` model instance.

    Returns
    -------
    lonpole : float or `~astropy/units.Quantity`
        Native longitude of the celestial pole in degrees.

    """
    if isinstance(skycoord, coords.SkyCoord):
        lon = skycoord.spherical.lon.value
        lat = skycoord.spherical.lat.value
        unit = u.deg

    else:
        lon, lat = skycoord
        unit = None
        if isinstance(lon, u.Quantity):
            lon = lon.to(u.deg).to_value()
            unit = u.deg

        if isinstance(lat, u.Quantity):
            lat = lat.to(u.deg).to_value()
            unit = u.deg

    cel = Celprm()
    cel.ref = [lon, lat]
    cel.prj.code = projection.prjprm.code
    pvrange = projection.prjprm.pvrange
    if pvrange:
        i1 = pvrange // 100
        i2 = i1 + (pvrange % 100) + 1
        cel.prj.pv = i1 * [None] + list(projection.prjprm.pv[i1:i2])
    cel.set()

    lonpole = cel.ref[2]
    if unit is not None:
        lonpole = lonpole * unit

    return lonpole


def get_projcode(wcs_info):
    # CTYPE here is only the imaging CTYPE keywords
    sky_axes, _, _ = get_axes(wcs_info)
    if not sky_axes:
        return None
    projcode = wcs_info["CTYPE"][sky_axes[0]][5:8].upper()
    if projcode not in projections.projcodes:
        msg = f"Projection code {projcode}, not recognized"
        raise UnsupportedProjectionError(msg)
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
        wcs_info["WCSAXES"] = header["WCSAXES"]
    except KeyError:
        p = re.compile(r"ctype[\d]*", re.IGNORECASE)
        ctypes = header["CTYPE*"]
        keys = list(ctypes.keys())
        for key in keys[::-1]:
            if p.split(key)[-1] != "":
                keys.remove(key)
        wcs_info["WCSAXES"] = len(keys)
    wcsaxes = wcs_info["WCSAXES"]
    # if not present call get_csystem
    wcs_info["RADESYS"] = header.get("RADESYS", "ICRS")
    wcs_info["VAFACTOR"] = header.get("VAFACTOR", 1)
    wcs_info["NAXIS"] = header.get("NAXIS", 0)
    wcs_info["EQUINOX"] = header.get("EQUINOX", None)
    wcs_info["EPOCH"] = header.get("EPOCH", None)
    wcs_info["DATEOBS"] = header.get("MJD-OBS", header.get("DATE-OBS", None))

    ctype = []
    cunit = []
    crpix = []
    crval = []
    cdelt = []
    for i in range(1, wcsaxes + 1):
        ctype.append(header[f"CTYPE{i}"])
        cunit.append(header.get(f"CUNIT{i}", None))
        crpix.append(header.get(f"CRPIX{i}", 0.0))
        crval.append(header.get(f"CRVAL{i}", 0.0))
        cdelt.append(header.get(f"CDELT{i}", 1.0))

    if "CD1_1" in header:
        wcs_info["has_cd"] = True
    else:
        wcs_info["has_cd"] = False
    pc = np.zeros((wcsaxes, wcsaxes))
    for i in range(1, wcsaxes + 1):
        for j in range(1, wcsaxes + 1):
            key = f"CD{i}_{j}" if wcs_info["has_cd"] else f"PC{i}_{j}"
            if key in header:
                pc[i - 1, j - 1] = header[key]
            elif i == j:
                pc[i - 1, j - 1] = 1.0
            else:
                pc[i - 1, j - 1] = 0.0
    wcs_info["CTYPE"] = ctype
    wcs_info["CUNIT"] = cunit
    wcs_info["CRPIX"] = crpix
    wcs_info["CRVAL"] = crval
    wcs_info["CDELT"] = cdelt
    wcs_info["PC"] = pc
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
        indices in the input representing sky and spectral coordinates.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        msg = "Expected a FITS Header or a dict."
        raise TypeError(msg)

    # Split each CTYPE value at "-" and take the first part.
    # This should represent the coordinate system.
    ctype = [ax.split("-")[0].upper() for ax in wcs_info["CTYPE"]]
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
    """Determine if the sky axes in CTYPE match to form a standard celestial system."""

    for item in sky_pairs.values():
        if ctype[sky_inmap[0]] == item[0]:
            if ctype[sky_inmap[1]] != item[1]:
                msg = "Inconsistent ctype for sky coordinates {} and {}".format(*ctype)
                raise ValueError(msg)
            break
        if ctype[sky_inmap[1]] == item[0]:
            if ctype[sky_inmap[0]] != item[1]:
                msg = "Inconsistent ctype for sky coordinates {} and {}".format(*ctype)
                raise ValueError(msg)
            sky_inmap = sky_inmap[::-1]
            break


specsystems = [
    "WAVE",
    "FREQ",
    "ENER",
    "WAVEN",
    "AWAV",
    "VRAD",
    "VOPT",
    "ZOPT",
    "BETA",
    "VELO",
]

sky_systems_map = {
    "ICRS": coords.ICRS,
    "FK5": coords.FK5,
    "FK4": coords.FK4,
    "FK4NOE": coords.FK4NoETerms,
    "GAL": coords.Galactic,
    "HOR": coords.AltAz,
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
        msg = "Expected a FITS Header or a dict."
        raise TypeError(msg)
    transforms = []
    wcs_linear = fitswcs_linear(wcs_info)
    transforms.append(wcs_linear)
    wcs_nonlinear = fitswcs_nonlinear(wcs_info)
    if wcs_nonlinear is not None:
        transforms.append(wcs_nonlinear)
    return functools.reduce(core._model_oper("|"), transforms)


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
        msg = "Expected a FITS Header or a dict."
        raise TypeError(msg)

    pc = wcs_info["PC"]
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
            crpix.append(wcs_info["CRPIX"][i])
            cdelt.append(wcs_info["CDELT"][i])
    else:
        cdelt = wcs_info["CDELT"]
        crpix = wcs_info["CRPIX"]

    # if wcsaxes == 2:
    rotation = astmodels.AffineTransformation2D(matrix=pc, name="pc_matrix")

    translation_models = [
        astmodels.Shift(-(shift - 1), name="crpix" + str(i + 1))
        for i, shift in enumerate(crpix)
    ]
    translation = functools.reduce(lambda x, y: x & y, translation_models)

    if not wcs_info["has_cd"]:
        # Do not compute scaling since CDELT* = 1 if CD is present.
        scaling_models = [
            astmodels.Scale(scale, name="cdelt" + str(i + 1))
            for i, scale in enumerate(cdelt)
        ]

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
        msg = "Expected a FITS Header or a dict."
        raise TypeError(msg)

    transforms = []
    projcode = get_projcode(wcs_info)
    if projcode is not None:
        projection = create_projection_transform(projcode).rename(projcode)
        transforms.append(projection)
    # Create the sky rotation transform
    sky_axes, _, _ = get_axes(wcs_info)
    if sky_axes:
        phip, lonp = (wcs_info["CRVAL"][i] for i in sky_axes)
        thetap = _compute_lon_pole((phip, lonp), projection)
        n2c = astmodels.RotateNative2Celestial(phip, lonp, thetap, name="crval")
        transforms.append(n2c)
    if transforms:
        return functools.reduce(core._model_oper("|"), transforms)
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

    projklassname = f"Pix2Sky_{projcode}"
    try:
        projklass = getattr(projections, projklassname)
    except AttributeError as err:
        raise UnsupportedProjectionError(projcode) from err

    projparams = {}
    return projklass(**projparams)


def is_high_level(*args, low_level_wcs):
    """
    Determine if args matches the high level classes as defined by
    ``low_level_wcs``.
    """
    if len(args) != len(low_level_wcs.world_axis_object_classes):
        return False

    type_match = [
        (type(arg), waoc[0])
        for arg, waoc in zip(
            args, low_level_wcs.world_axis_object_classes.values(), strict=False
        )
    ]

    types_are_high_level = [argt is t for argt, t in type_match]

    if all(types_are_high_level):
        return True

    if any(types_are_high_level):
        msg = (
            "Invalid types were passed, got "
            f"({', '.join(tm[0].__name__ for tm in type_match)}) expected "
            f"({', '.join(tm[1].__name__ for tm in type_match)})."
        )
        raise TypeError(msg)

    return False
