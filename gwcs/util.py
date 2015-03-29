"""
Utility function for WCS

"""
from __future__ import division, print_function

import numpy as np
from astropy.modeling.projections import projcodes
try:
    from astropy import time
    HAS_TIME = True
except ImportError:
    HAS_TIME = False
#from astropy import coordinates

#these ctype values do not include yzLN and yzLT pairs
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
        self.message = message
        super(UnsupportedTransformError, self).__init__()


class ModelDimensionalityError(Exception):
    def __init__(self, message):
        self.message = message
        super(ModelDimensionalityError, self).__init__()


class RegionError(Exception):
    def __init__(self, message):
        self.message = message
        super(UnknownRegionError, self).__init__()


class CoordinateFrameError(Exception):
    def __init__(self, message):
        self.message = message
        super(CoordinateFrameError, self).__init__()


def get_projcode(ctype):
    # CTYPE here is only the imaging CTYPE keywords
    projcode = ctype[0][5:8].upper()
    if projcode not in projcodes:
        raise ValueError('Projection code %s, not recognized' %projcode)
    return projcode


def read_wcs_from_header(header, mode=None):
        """
        Read basic WCS info from the Primary header of the data file.
        """
        wcs_info = {}
        try:
            wcsaxes = header['WCSAXES']
        except KeyError:
            if mode == 'imaging':
                wcsaxes = 2
            elif mode == 'spectroscopic':
                wcsaxes = 3
                #wcs_info['SPECSYS'] = header['SPECSYS']
            elif mode == 'ifu':
                wcsaxes = 3
            elif mode == 'multislit':
                wcsaxes = 3
            else:
                raise ValueError('Unrecognized mode')
            # or perhaps do as wcslib
            # wcsaxes = max(len(dict(**header["ctype*"])), header['NAXES'])
        wcs_info['WCSAXES'] = wcsaxes
        #if not present call get_csystem
        wcs_info['RADESYS'] = header.get('RADESYS', 'ICRS')
        wcs_info['VAFACTOR'] = header.get('VAFACTOR', 1)
        wcs_info['NAXIS'] = header.get('NAXIS', 0)
        # date keyword?
        wcs_info['DATEOBS'] = header.get('DATE-OBS', 'DATEOBS')

        ctype = []
        cunit = []
        crpix = []
        crval = []
        cdelt = []
        for i in range(1, wcsaxes+1):
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


def get_axes(wcs_info):
    """
    Determines the axes in the input with spectral and sky coordinates.

    Returns
    -------
    sky_inmap, spectral_inmap : tuples
        indices in the input with sky and spectral cordinates
        These can be used directly in composite transforms

    """
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
                raise ValueError("Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            break
        elif ctype[sky_inmap[1]] == item[0]:
            if ctype[sky_inmap[0]] != item[1]:
                raise ValueError("Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            sky_inmap = sky_inmap[::-1]
            break
    return sky_inmap, spec_inmap


def get_sky_wcs_info(wcsinfo):
    sky_wcsinfo = {}
    sky_axes, _, = get_axes(wcsinfo)
    sky_wcsinfo_keys = ['CDELT', 'CRPIX', 'CRVAL', 'CTYPE', 'CUNIT']
    other_keys = ['RADESYS', 'VAFACTOR', 'has_cd']
    for kw in sky_wcsinfo_keys:
        sky_wcsinfo[kw] = np.asarray(wcsinfo[kw])[sky_axes].tolist()
    for kw in other_keys:
        sky_wcsinfo[kw] = wcsinfo[kw]
    pc = np.zeros((2,2))
    wpc = wcsinfo['PC']
    pc[0, 0] = wpc[sky_axes[0], sky_axes[0]]
    pc[0, 1] = wpc[sky_axes[0], sky_axes[1]]
    pc[1, 0] = wpc[sky_axes[1], sky_axes[0]]
    pc[1, 1] = wpc[sky_axes[1], sky_axes[1]]
    sky_wcsinfo['PC'] = pc
    sky_wcsinfo['WCSAXES'] = 2
    return sky_wcsinfo

def get_csystem(header=None, radesys=None, equinox=None, dateobs=None):
    ctypesys = "UNKNOWN"
    if header is not None:
        ctype =  header["CTYPE*"]
        radesys = header.get("RADESYS", None)
        equinox = header.get("EQUINOX", None)
        epoch = header.get("EPOCH", None)
        dateobs = header.get("MJD-OBS", header.get("DATE-OBS", None))
    cs = [ctype[0][:4], ctype[1][:4]]

    for item in sky_pairs.items():
        if cs[0] in item[1]:
            assert cs[1] in item[1], "Inconsistent coordinate system type in CTYPE"
            ctypesys = item[0]
    if ctypesys == 'spec':
        #try to get the rest of the kw that define a spectral system from the header
        return csystems.SpectralCoordSystem(cs, **kwargs)
    if ctypesys not in ['equatorial', 'ecliptic']:
        return coordinates.__getattribute__(sky_systems_map[ctypesys])(0., 0., equinox=equinox,
                        obstime=dateobs, unit=units)
        #return ctypesys, radesys, equinox
    else:
        if radesys is None:
            if equinox is None:
                radesys = "ICRS"
            else:
                if equinox < 1984.0:
                    radesys = 'FK4'
                else:
                    radesys = 'FK5'
        if radesys in ['FK4', 'FK4-NO-E']:
            if radesys == 'FK4-NO-E':
                assert ctypesys != "ecliptic", (
                    " Inconsistent coordinate systems: 'ecliptic' and 'FK4-NO-E' ")
            if equinox is None:
                if epoch is None:
                    equinox = "1950.0"
                else:
                    equinox = epoch
        elif radesys == 'FK5':
            if equinox is None:
                if epoch is None:
                    equinox = "2000.0"
                else:
                    equinox = epoch
        elif radesys == 'GAPPT':
            assert dateobs is not None, "Either 'DATE-OBS' or 'MJD-OBS' is required"
            equinox = dateobs
    if HAS_TIME:
        if equinox is not None:
            equinox = time.Time(equinox, scale='utc')
        if dateobs is not None:
            dateobs = time.Time(dateobs, scale='utc')
    return ctypesys, radesys, equinox, dateobs
    #units = header.get('CUNIT*', ['deg', 'deg'])
    #return coordinates.__getattribute__(sky_systems_map[radesys])(0., 0., equinox=equinox,
    #                obstime=dateobs, unit=units)


def populate_meta(header, regions, regionsschema):
    fits_keywords = []
    if regions is None:
        fits_keywords.append('WREGIONS')
    if regionsschema is None:
        fits_keywords.append('REGSCHEM')

    meta = {}
    meta['NAXIS'] = header.get('NAXIS', 0)
    for i in range(1, meta['NAXIS']+1):
        name = 'NAXIS'+str(i)
        meta[name] = header.get(name)
        name = 'CUNIT'+str(i)
        meta[name] = header.get(name, "deg")

    for key in fits_keywords:
        meta[key] = header.get(key, None)
    return meta

specsystems = ["WAVE", "FREQ", "ENER", "WAVEN", "AWAV",
               "VRAD", "VOPT", "ZOPT", "BETA", "VELO"]

sky_systems_map = {'ICRS': 'ICRSCoordinates',
                                    'FK5': 'FK5Coordinates',
                                    'FK4': 'FK4Coordinates',
                                    'FK4NOE': 'FK4NoETermCoordinates',
                                    'GAL': 'GalacticCorrdinates',
                                    'HOR': 'HorizontalCoordinates'
                                }
