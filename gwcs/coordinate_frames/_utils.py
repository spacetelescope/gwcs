import warnings

from astropy.wcs.wcsapi.fitswcs import CTYPE_TO_UCD1

__all__ = ["get_ctype_from_ucd"]


def _ucd1_to_ctype_name_mapping(ctype_to_ucd, allowed_ucd_duplicates):
    inv_map = {}
    new_ucd = set()

    for kwd, ucd in ctype_to_ucd.items():
        if ucd in inv_map:
            if ucd not in allowed_ucd_duplicates:
                new_ucd.add(ucd)
            continue
        inv_map[ucd] = allowed_ucd_duplicates.get(ucd, kwd)

    if new_ucd:
        warnings.warn(
            "Found unsupported duplicate physical type in 'astropy' mapping to CTYPE.\n"
            "Update 'gwcs' to the latest version or notify 'gwcs' developer.\n"
            "Duplicate physical types will be mapped to the following CTYPEs:\n"
            + "\n".join([f"{ucd!r:s} --> {inv_map[ucd]!r:s}" for ucd in new_ucd])
        )

    return inv_map


# List below allowed physical type duplicates and a corresponding CTYPE
# to which all duplicates will be mapped to:
_ALLOWED_UCD_DUPLICATES = {
    "time": "TIME",
    "em.wl": "WAVE",
}

UCD1_TO_CTYPE = _ucd1_to_ctype_name_mapping(
    ctype_to_ucd=CTYPE_TO_UCD1, allowed_ucd_duplicates=_ALLOWED_UCD_DUPLICATES
)


def get_ctype_from_ucd(ucd):
    """
    Return the FITS ``CTYPE`` corresponding to a UCD1 value.

    Parameters
    ----------
    ucd : str
        UCD string, for example one of ```WCS.world_axis_physical_types``.

    Returns
    -------
    CTYPE : str
        The corresponding FITS ``CTYPE`` value or an empty string.
    """
    return UCD1_TO_CTYPE.get(ucd, "")
